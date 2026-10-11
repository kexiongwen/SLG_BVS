"""Read-only kernel comparison; no sampler implementation is modified."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import timeit

import numpy as np
import scipy
from scipy.linalg import cholesky
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _legacy_kernels():
    """Load the retired rank-1 kernels from the baseline archive.

    The incremental Cholesky updates were removed from BVS/ after this
    comparison showed full refactorization is faster at realistic model
    sizes; the archived copy keeps this benchmark runnable.
    """
    from zipfile import ZipFile
    archive = ROOT / "benchmarks" / "baselines" / "before_optimizations.zip"
    namespace = {}
    with ZipFile(archive) as saved:
        source = saved.read("BVS/cholupdate.py").decode("utf-8-sig")
    exec(compile(source, f"{archive}!/BVS/cholupdate.py", "exec"), namespace)
    return namespace["chol_add_col"], namespace["chol_del"]


chol_add_col, chol_del = _legacy_kernels()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(20261011)
    rows = []
    with threadpool_limits(1):
        for size in (5, 10, 20, 30, 50, 100, 200):
            X = rng.normal(size=(2 * (size + 1), size + 1))
            G = X.T @ X + np.eye(size + 1)
            for position in (0, size // 2, size):
                current = np.delete(np.arange(size + 1), position)
                target = np.arange(size + 1)
                L = np.linalg.cholesky(G[np.ix_(current, current)])
                operations = {
                    'add_update': lambda: chol_add_col(G[current, position], G[position, position], L, position),
                    'add_numpy_full': lambda: np.linalg.cholesky(G[np.ix_(target, target)]),
                    'add_scipy_full': lambda: cholesky(G[np.ix_(target, target)], lower=True, check_finite=False),
                }
                full_factor = np.linalg.cholesky(G)
                operations.update({
                    'delete_update': lambda: chol_del(full_factor, position),
                    'delete_numpy_full': lambda: np.linalg.cholesky(G[np.ix_(current, current)]),
                    'delete_scipy_full': lambda: cholesky(G[np.ix_(current, current)], lower=True, check_finite=False),
                })
                # Rotate method order to reduce persistent first-method bias.
                names = list(operations)
                offset = (size + position) % len(names)
                names = names[offset:] + names[:offset]
                for name in names:
                    fn = operations[name]
                    result = fn()
                    expected = G if name.startswith('add') else G[np.ix_(current, current)]
                    error = np.linalg.norm(result @ result.T - expected) / np.linalg.norm(expected)
                    assert np.isfinite(result).all() and error < 1e-12, (size, position, name, error)
                    number = 150 if size <= 50 else 50
                    batches = timeit.repeat(fn, number=number, repeat=5)
                    rows.append(dict(size_before_add=size, position=position, method=name,
                                     calls_per_batch=number, batches_seconds=batches,
                                     median_microseconds=1e6 * statistics.median(batches) / number,
                                     relative_reconstruction_error=float(error),
                                     all_diagonals_positive=bool(np.all(np.diag(result) > 0))))
        info = threadpool_info()
    output = dict(
        seed=20261011, python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        platform=platform.platform(), processor=platform.processor(), threadpools=info,
        conventions='float64 SPD Gram matrices; existing lower factors; full methods include advanced-index submatrix copying; update methods include cross-product gathering. Factors and indices are precomputed outside timings.',
        limitations='Kernel microbenchmark only, not end-to-end sampler speed. One well-conditioned matrix per size, three positions, five timing batches. Does not establish near-singular stability or a universal crossover threshold.',
        rows=rows,
    )
    Path(args.output).write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    for row in rows:
        if row['position'] == row['size_before_add'] // 2:
            print(row['size_before_add'], row['method'], round(row['median_microseconds'], 3))


if __name__ == '__main__':
    main()
