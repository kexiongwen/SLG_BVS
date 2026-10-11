"""Optimization contracts; run with python tests/test_optimization_contracts.py.

The baseline archive records the corrected sampler immediately before the
optimizations. It is loaded under an isolated package name, so its proposal
and Cholesky functions cannot accidentally resolve to the current BVS package.
"""

import sys
from functools import lru_cache
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from BVS.DF_MCMC import DF_MCMC


@lru_cache(maxsize=1)
def baseline_driver():
    archive = ROOT / "benchmarks" / "baselines" / "before_optimizations.zip"
    alias = "_bvs_before_optimizations"
    package = ModuleType(alias)
    package.__package__ = alias
    package.__path__ = []
    package.__spec__ = ModuleSpec(alias, loader=None, is_package=True)
    sys.modules[alias] = package
    try:
        with ZipFile(archive) as saved:
            # Load dependencies in order; all relative imports remain inside
            # the alias and no extraction or repository writes are required.
            for name in ("cholupdate", "proposal", "DF_MCMC"):
                qualified_name = f"{alias}.{name}"
                filename = f"{archive}!/BVS/{name}.py"
                module = ModuleType(qualified_name)
                module.__package__ = alias
                module.__file__ = filename
                module.__spec__ = ModuleSpec(qualified_name, loader=None)
                sys.modules[qualified_name] = module
                setattr(package, name, module)
                source = saved.read(f"BVS/{name}.py").decode("utf-8-sig")
                exec(compile(source, filename, "exec"), module.__dict__)
    except Exception:
        for name in tuple(sys.modules):
            if name == alias or name.startswith(alias + "."):
                del sys.modules[name]
        raise

    driver = package.DF_MCMC.DF_MCMC
    assert driver is not DF_MCMC
    assert driver.__globals__["propose"].__module__ == alias + ".proposal"
    assert package.proposal.chol_add_col.__module__ == alias + ".cholupdate"
    return driver


def problems():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(30, 6))
    Y = 1.2 * X[:, 1] - 0.6 * X[:, 4] + rng.normal(size=30)
    yield "regular", Y, X, dict(s0=3, s_initial=2, seed=7)
    yield "screened_empty_start", Y, X, dict(
        s0=3, s_initial=0, gamma_init=[], screen=4, seed=19,
    )

    rng = np.random.default_rng(123)
    X = rng.normal(size=(30, 3))
    Y = 3 * X[:, 0] + 0.1 * rng.normal(size=30)
    yield "unavailable_swap", Y, X, dict(
        s0=1, s_initial=1, gamma_init=[0], screen=1, seed=7,
    )

    z = rng.normal(size=30)
    X = np.column_stack([z, z, z])
    Y = 3 * z + 0.1 * rng.normal(size=30)
    yield "unavailable_add", Y, X, dict(
        s0=2, s_initial=1, gamma_init=[0], seed=19,
    )
    yield "null_only", Y, X, dict(s0=0, s_initial=0, seed=7)


def assert_equivalent(actual, expected, context):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), context
        assert actual.keys() == expected.keys(), context
        for key in expected:
            if key == "model_counts":
                assert actual[key] == expected[key], context
            elif key in ("pip", "proposal", "accept", "n_iter"):
                np.testing.assert_array_equal(actual[key], expected[key],
                                              err_msg=f"{context}: {key}")
            else:
                assert_equivalent(actual[key], expected[key], f"{context}: {key}")
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected), context
        assert len(actual) == len(expected), context
        for i, (value, reference) in enumerate(zip(actual, expected)):
            assert_equivalent(value, reference, f"{context}: output {i}")
    else:
        expected_array = np.asarray(expected)
        if expected_array.dtype == np.bool_:
            np.testing.assert_array_equal(actual, expected, err_msg=context)
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12,
                                       err_msg=context)


def test_fixed_seed_outputs_match_corrected_baseline():
    before = baseline_driver()
    for name, Y, X, fixture in problems():
        for burn in (0, 9):
            for sample_beta in (False, True):
                for rb in (False, True):
                    for online in (False, True):
                        kw = dict(kappa1=1, kappa2=1, M=35, burn_in=burn,
                                  Sample_beta=sample_beta, RB=rb, online=online,
                                  **fixture)
                        context = (f"{name}, burn={burn}, Sample_beta={sample_beta}, "
                                   f"RB={rb}, online={online}")
                        expected = before(Y, X, **kw)
                        actual = DF_MCMC(Y, X, **kw)
                        assert_equivalent(actual, expected, context)
                        if online:
                            assert actual["n_iter"] == kw["M"], context
                            assert sum(actual["model_counts"].values()) == kw["M"], context
                            if name == "unavailable_swap":
                                assert actual["proposal"][2] > 0, context
                                assert actual["accept"][2] == 0, context
                            if name == "unavailable_add":
                                assert actual["proposal"][0] > 0, context
                                assert actual["accept"][0] == 0, context
                        else:
                            arrays = actual if isinstance(actual, list) else [actual]
                            for array in arrays:
                                assert array.shape[-1] == kw["M"], context
                                assert np.isfinite(array).all(), context


def test_retained_traces_own_only_kept_sample_storage():
    before = baseline_driver()
    _, Y, X, fixture = next(problems())
    kept, burn = 3, 121
    p = X.shape[1]
    for sample_beta in (False, True):
        for rb in (False, True):
            kw = dict(kappa1=1, kappa2=1, M=kept, burn_in=burn,
                      Sample_beta=sample_beta, RB=rb, **fixture)
            expected = before(Y, X, **kw)
            result = DF_MCMC(Y, X, **kw)
            context = f"storage: Sample_beta={sample_beta}, RB={rb}"
            assert_equivalent(result, expected, context)
            arrays = result if isinstance(result, list) else [result]
            for array in arrays:
                assert array.shape[-1] == kept, context
                assert array.flags.owndata, f"{context}: returned trace does not own storage"
                assert array.base is None, f"{context}: retained a hidden backing buffer"
            expected_bytes = kept * (
                p + (8 * p + 8 if sample_beta else 0) + (8 * p if rb else 0)
            )
            assert sum(array.nbytes for array in arrays) == expected_bytes, context


if __name__ == "__main__":
    test_fixed_seed_outputs_match_corrected_baseline()
    test_retained_traces_own_only_kept_sample_storage()
    print("all optimization contract tests passed")
