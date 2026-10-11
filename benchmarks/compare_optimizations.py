"""Compare the frozen corrected sampler against workspace optimizations.

Example (single BLAS thread, 5 measured calls and one warmup per variant):
    python benchmarks/compare_optimizations.py
Fast smoke run:
    python benchmarks/compare_optimizations.py --quick --cases strong_signal

Timing and allocation measurements run in separate fresh subprocesses. A result
describes only the selected seeded workloads and this machine; it is not a
mixing-time test or evidence of a universal performance improvement.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "benchmarks" / "baselines" / "before_optimizations.zip"
BASELINE_SHA256 = "256a63a9b5e8c73881ecba644003a270dd964e0bd7c72de5e0b3d2cb3c5d25ec"
THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)
CASE_NAMES = ("strong_signal", "weak_signal", "correlated_competitors", "trace_heavy")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(variant):
    if variant == "baseline":
        with zipfile.ZipFile(BASELINE) as archive:
            return {
                name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in sorted(archive.namelist()) if name.endswith(".py")
            }
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in sorted((ROOT / "BVS").glob("*.py"))
    }


def git_output(*args):
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True,
        timeout=15, check=False,
    )
    return {"exit_code": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()}


def case_config(name, quick=False, tiny=False):
    config = dict(
        name=name, n=250, p=800, M=1500, burn_in=500, s0=30, s_initial=5,
        kappa1=0.5, kappa2=0.5, online=True, RB=True, Sample_beta=False,
        data_seed=2871, sampler_seed=9137, design="independent", signal_scale=1.0,
    )
    if name == "weak_signal":
        config.update(signal_scale=0.12, RB=False, kappa2=0.0,
                      description="Weak signals with a lighter sparsity penalty; plain online sampling.")
    elif name == "correlated_competitors":
        config.update(design="correlated_blocks", signal_scale=0.65,
                      description="Five signal groups have four strongly correlated competing predictors (rho=0.95); no claim of verified multimodality.")
    elif name == "trace_heavy":
        config.update(p=2000, M=2000, burn_in=2000, online=False, Sample_beta=True,
                      description="Stored gamma, beta, sigma2, and Rao-Blackwell traces; equal retained and burn-in lengths expose retained backing allocations.")
    else:
        config["description"] = "Five strong signals in an independent design; online Rao-Blackwell summaries."
    if quick:
        config.update(n=80, p=120, M=120, burn_in=40, s0=15)
        if name == "trace_heavy":
            config.update(p=300, M=200, burn_in=200)
    if tiny:
        config.update(n=32, p=24, M=8, burn_in=4, s0=12, s_initial=3)
    return config


def make_problem(np, config):
    rng = np.random.default_rng(config["data_seed"])
    n, p = config["n"], config["p"]
    X = rng.standard_normal((n, p))
    signal_columns = np.arange(5)
    if config["design"] == "correlated_blocks":
        covariance = np.full((4, 4), 0.95)
        np.fill_diagonal(covariance, 1.0)
        factor = np.linalg.cholesky(covariance)
        for start in range(0, 20, 4):
            X[:, start:start + 4] = X[:, start:start + 4] @ factor.T
        signal_columns = np.arange(0, 20, 4)
    beta = np.zeros(p)
    beta[signal_columns] = config["signal_scale"] * np.array([2.5, -2., 2., -1.5, 1.5])
    Y = (X @ beta + rng.standard_normal(n))[:, None]
    return X, Y, signal_columns


def array_digest(np, array):
    """Hash in C order without allocating a full copy of a strided trace."""
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    if array.flags.c_contiguous:
        digest.update(memoryview(array).cast("B"))
    else:
        for chunk in np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                               order="C", buffersize=65536):
            digest.update(np.ascontiguousarray(chunk).tobytes())
    return digest.hexdigest()


def summarize_result(np, result, config, signal_columns):
    arrays = []

    def describe(value, name):
        if isinstance(value, np.ndarray):
            arrays.append((name, value))
            return {"shape": list(value.shape), "dtype": str(value.dtype),
                    "sha256": array_digest(np, value)}
        if isinstance(value, dict):
            # Model-count dictionary keys are tuples; stringify in a stable order.
            return {str(key): describe(value[key], f"{name}.{key}")
                    for key in sorted(value, key=str)}
        if isinstance(value, (list, tuple)):
            return [describe(item, f"{name}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, np.generic):
            return value.item()
        return value

    structure = describe(result, "result")
    backing = {}
    seen_arrays = set()
    logical_bytes = 0
    details = []
    for name, array in arrays:
        if id(array) not in seen_arrays:
            logical_bytes += array.nbytes
            seen_arrays.add(id(array))
        owner = array
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        backing[id(owner)] = owner.nbytes
        details.append(dict(name=name, shape=list(array.shape),
                            logical_bytes=int(array.nbytes),
                            backing_bytes=int(owner.nbytes),
                            owns_data=bool(array.flags.owndata)))
    if isinstance(result, dict):
        counts = result["model_counts"]
        draws = sum(counts.values())
        model_summary = dict(
            retained_draws=int(draws), unique_models=len(counts),
            mean_size=float(sum(len(key) * value for key, value in counts.items()) / draws),
            signal_pip=result["pip"][signal_columns].tolist(),
            proposals=result["proposal"].tolist(), accepted=result["accept"].tolist(),
        )
    else:
        gamma = result[1] if config["Sample_beta"] else result
        if isinstance(gamma, list):
            gamma = gamma[0]
        model_summary = dict(retained_draws=int(gamma.shape[1]),
                             mean_size=float(gamma.sum(axis=0).mean()),
                             signal_pip=gamma[signal_columns].mean(axis=1).tolist())
    return dict(
        result_sha256=hashlib.sha256(json.dumps(structure, sort_keys=True).encode()).hexdigest(),
        array_descriptors=details, output_logical_array_bytes=int(logical_bytes),
        output_backing_array_bytes=int(sum(backing.values())), model_summary=model_summary,
    )


def memory_snapshot(psutil):
    if psutil is None:
        return {"available": False, "reason": "psutil unavailable"}
    memory = psutil.Process().memory_info()
    data = dict(available=True, rss_bytes=int(memory.rss))
    # Windows peak_wset is the OS lifetime peak working set. Linux ru_maxrss
    # uses different units/platform conventions, so do not silently substitute.
    if hasattr(memory, "peak_wset"):
        data.update(os_peak_rss_bytes=int(memory.peak_wset),
                    os_peak_source="psutil.memory_info().peak_wset (lifetime peak working set)")
    else:
        data.update(os_peak_rss_bytes=None, os_peak_source="not available on this platform")
    return data


def worker(args):
    # Set both process environment and runtime limits. Parent sets the same
    # environment before launch so libraries imported indirectly also see it.
    for name in THREAD_VARS:
        os.environ[name] = str(args.threads)
    os.environ["PYTHONHASHSEED"] = "0"
    import numpy as np
    import scipy
    from threadpoolctl import threadpool_info, threadpool_limits
    try:
        import psutil
    except ImportError:
        psutil = None

    if sha256_file(BASELINE) != BASELINE_SHA256:
        raise RuntimeError("baseline archive checksum changed")
    import_root = BASELINE if args.variant == "baseline" else ROOT
    sys.path.insert(0, str(import_root))
    module = importlib.import_module("BVS.DF_MCMC")
    sampler = module.DF_MCMC
    expected_prefix = str(import_root).lower()
    if not str(module.__file__).lower().startswith(expected_prefix):
        raise RuntimeError(f"wrong sampler imported: {module.__file__}")
    config = case_config(args.cases[0], args.quick, args.tiny)
    before_hashes = source_hashes(args.variant)
    with threadpool_limits(limits=args.threads):
        started = time.perf_counter()
        X, Y, signal_columns = make_problem(np, config)
        setup_seconds = time.perf_counter() - started
        input_hashes = {"X": array_digest(np, X), "Y": array_digest(np, Y)}
        kwargs = {key: config[key] for key in
                  ("kappa1", "kappa2", "s0", "s_initial", "M", "burn_in", "online", "RB", "Sample_beta")}
        kwargs["seed"] = config["sampler_seed"]
        environment = dict(
            python=sys.version, executable=sys.executable, numpy=np.__version__,
            scipy=scipy.__version__, platform=platform.platform(), machine=platform.machine(),
            processor=platform.processor(), logical_cpus=os.cpu_count(),
            thread_env={key: os.environ.get(key) for key in THREAD_VARS},
            active_threadpools=threadpool_info(), imported_sampler=str(module.__file__),
            imported_bvs_modules={name: str(value.__file__) for name, value in sys.modules.items()
                                  if name.startswith("BVS") and getattr(value, "__file__", None)},
        )
        if psutil is not None:
            process = psutil.Process()
            environment.update(physical_cpus=psutil.cpu_count(logical=False),
                               total_ram_bytes=int(psutil.virtual_memory().total))
            if hasattr(process, "cpu_affinity"):
                environment["cpu_affinity"] = process.cpu_affinity()
        config_text = io.StringIO()
        with contextlib.redirect_stdout(config_text):
            np.show_config()
        environment["numpy_build_configuration"] = config_text.getvalue()
        runs = []
        if args.phase == "speed":
            for _ in range(args.warmups):
                sampler(Y, X, **kwargs)
            gc.collect()
            for repeat in range(args.repeats):
                gc.collect()
                t0 = time.perf_counter()
                result = sampler(Y, X, **kwargs)
                elapsed = time.perf_counter() - t0
                summary = summarize_result(np, result, config, signal_columns)
                runs.append(dict(repeat=repeat, wall_seconds=elapsed, **summary))
                del result
        else:
            # No warmup: lifetime peak working set and Python allocation peak
            # must describe this call, not a preceding retained trace.
            gc.collect()
            memory_before = memory_snapshot(psutil)
            tracemalloc.start()
            t0 = time.perf_counter()
            result = sampler(Y, X, **kwargs)
            elapsed = time.perf_counter() - t0
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_after = memory_snapshot(psutil)
            summary = summarize_result(np, result, config, signal_columns)
            runs.append(dict(repeat=0, instrumented_wall_seconds=elapsed,
                             tracemalloc_current_bytes=current, tracemalloc_peak_bytes=peak,
                             memory_before=memory_before, memory_after=memory_after, **summary))
        after_hashes = source_hashes(args.variant)
        if before_hashes != after_hashes:
            raise RuntimeError("sampler source files changed during a worker run")
        return dict(variant=args.variant, phase=args.phase, config=config,
                    input_sha256=input_hashes, source_sha256=before_hashes,
                    source_unchanged=True, setup_seconds=setup_seconds,
                    environment=environment, runs=runs)


def aggregate(workers):
    summaries = {}
    for name in CASE_NAMES:
        entries = [entry for entry in workers if entry.get("result", {}).get("config", {}).get("name") == name]
        if not entries:
            continue
        variants = {}
        all_digests = set()
        input_digests = set()
        for variant in ("baseline", "optimized"):
            matching = [entry["result"] for entry in entries if entry["result"]["variant"] == variant]
            speed = [run for item in matching if item["phase"] == "speed" for run in item["runs"]]
            memory = [run for item in matching if item["phase"] == "memory" for run in item["runs"]]
            for item in matching:
                input_digests.add(json.dumps(item["input_sha256"], sort_keys=True))
                all_digests.update(run["result_sha256"] for run in item["runs"])
            record = {}
            if speed:
                durations = [run["wall_seconds"] for run in speed]
                record.update(speed_repeats=len(durations), median_wall_seconds=statistics.median(durations),
                              min_wall_seconds=min(durations), max_wall_seconds=max(durations),
                              raw_wall_seconds=durations, model_summary=speed[0]["model_summary"])
            if memory:
                record.update(output_backing_array_bytes=memory[0]["output_backing_array_bytes"],
                              output_logical_array_bytes=memory[0]["output_logical_array_bytes"],
                              tracemalloc_peak_bytes=memory[0]["tracemalloc_peak_bytes"],
                              os_peak_rss_bytes=memory[0]["memory_after"].get("os_peak_rss_bytes"))
            variants[variant] = record
        summary = dict(variants=variants, exact_result_hashes_match=len(all_digests) == 1,
                       input_hashes_match=len(input_digests) == 1)
        base, optimized = variants["baseline"], variants["optimized"]
        if "median_wall_seconds" in base and "median_wall_seconds" in optimized:
            summary["baseline_over_optimized_median_speed_ratio"] = base["median_wall_seconds"] / optimized["median_wall_seconds"]
        for metric in ("output_backing_array_bytes", "tracemalloc_peak_bytes", "os_peak_rss_bytes"):
            if base.get(metric) and optimized.get(metric) is not None:
                summary[f"{metric}_reduction_fraction"] = 1 - optimized[metric] / base[metric]
        summaries[name] = summary
    return summaries


def coordinator(args):
    if args.worker:
        raise ValueError("coordinator called in worker mode")
    baseline_hash = sha256_file(BASELINE)
    if baseline_hash != BASELINE_SHA256:
        raise RuntimeError("baseline archive checksum changed")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    phases = ("speed", "memory") if args.phase == "both" else (args.phase,)
    provenance = dict(
        schema_version=1, started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        actual_argv=sys.argv, cwd=str(Path.cwd()), commit=git_output("rev-parse", "HEAD"),
        dirty_state=git_output("status", "--short"),
        baseline_archive=str(BASELINE), baseline_sha256=baseline_hash,
        harness_sha256=sha256_file(__file__),
        source_sha256_before={variant: source_hashes(variant) for variant in ("baseline", "optimized")},
        contract=dict(
            decision="Measure speed and peak/retained array memory for the corrected baseline versus workspace sampler optimizations.",
            arithmetic="NumPy float64; identical generated inputs and fixed NumPy Generator seeds per case.",
            scope=[case_config(name, args.quick, args.tiny) for name in args.cases],
            speed_repeats=args.repeats, speed_warmups=args.warmups,
            memory_calls_per_variant_case=1, threads=args.threads,
            thread_limit_kind="cooperative environment variables plus threadpoolctl runtime limits",
            worker_wall_timeout_seconds=args.timeout,
            comparison="Same inputs, RNG seed, interpreter and thread limits; baseline/optimized order alternates between cases. Separate fresh worker processes for speed and memory. Speed repeats use identical deterministic chains to measure runtime variability, not Monte Carlo uncertainty.",
            speed_timer="DF_MCMC call only; includes Gram construction and sampling, excludes data generation/imports/digest computation. No tracemalloc during speed timings.",
            memory_definition="tracemalloc starts after input generation; output_backing_array_bytes deduplicates ultimate ndarray backing allocations, including burn-in slices. Windows peak RSS is lifetime peak working set and includes interpreter/import/input memory.",
            result_check="Exact result hashes and input hashes are reported. Floating-point hash mismatch requires investigation and does not by itself prove numerical inequivalence.",
            exclusions="No claim about all dimensions, machines, data, posterior mixing, or verified multimodality. No fresh accuracy theorem is inferred from timings.",
            failure_interpretation="Timeout or worker error is incomplete performance evidence, never a speed or correctness win. Source changes invalidate comparability.",
        ), workers=[], summary={}, completed=False,
    )

    def save():
        output.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    environment = os.environ.copy()
    for name in THREAD_VARS:
        environment[name] = str(args.threads)
    environment["PYTHONHASHSEED"] = "0"
    started = time.perf_counter()
    errors = False
    for case_index, name in enumerate(args.cases):
        order = ("baseline", "optimized") if case_index % 2 == 0 else ("optimized", "baseline")
        for phase in phases:
            for variant in order:
                command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                           "--variant", variant, "--phase", phase, "--cases", name,
                           "--repeats", str(args.repeats), "--warmups", str(args.warmups),
                           "--threads", str(args.threads)]
                if args.quick:
                    command.append("--quick")
                if args.tiny:
                    command.append("--tiny")
                print(f"{name}: {variant} {phase}", flush=True)
                entry = dict(command=command, case=name, variant=variant, phase=phase)
                t0 = time.perf_counter()
                try:
                    process = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                                             text=True, encoding="utf-8", timeout=args.timeout, check=False)
                    entry.update(exit_code=process.returncode, stderr=process.stderr,
                                 worker_wall_seconds=time.perf_counter() - t0)
                    if process.returncode:
                        entry["stdout"] = process.stdout
                        errors = True
                    else:
                        entry["result"] = json.loads(process.stdout)
                except subprocess.TimeoutExpired as error:
                    entry.update(timed_out=True, worker_wall_seconds=time.perf_counter() - t0,
                                 stdout=str(error.stdout or ""), stderr=str(error.stderr or ""))
                    errors = True
                except json.JSONDecodeError as error:
                    entry.update(parse_error=str(error), stdout=process.stdout)
                    errors = True
                provenance["workers"].append(entry)
                provenance["summary"] = aggregate(provenance["workers"])
                save()
    provenance.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                      total_wall_seconds=time.perf_counter() - started,
                      source_sha256_after={variant: source_hashes(variant) for variant in ("baseline", "optimized")})
    provenance["source_unchanged"] = provenance["source_sha256_before"] == provenance["source_sha256_after"]
    provenance["completed"] = not errors and provenance["source_unchanged"]
    save()
    print(json.dumps(provenance["summary"], indent=2))
    print(f"Results: {output}")
    return 0 if provenance["completed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", choices=CASE_NAMES, default=list(CASE_NAMES))
    parser.add_argument("--quick", action="store_true", help="use smaller workload dimensions")
    parser.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180., help="wall limit per subprocess in seconds")
    parser.add_argument("--phase", choices=("both", "speed", "memory"), default="both")
    parser.add_argument("--output", default=str(ROOT / "benchmarks" / "results" / "optimization_comparison.json"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=("baseline", "optimized"), default="optimized", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0 or args.threads < 1 or args.timeout <= 0:
        parser.error("repeats/threads/timeout must be positive and warmups nonnegative")
    if args.worker:
        if args.phase == "both" or len(args.cases) != 1:
            parser.error("a worker needs exactly one case and one phase")
        print(json.dumps(worker(args), allow_nan=False))
        return 0
    return coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
