"""Exercise the actual simulation-study helpers without running its large grid.

Run from the repository root: python tests/test_notebook_regressions.py
"""
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from BVS.DF_MCMC import DF_MCMC
from BVS.proposal import make_add, make_delete, weight


def notebook_helpers():
    """Load only function definitions; skip imports and experiment-loop cells."""
    path = ROOT / "test2.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = dict(
        np=np, time=time, cholesky=cholesky, solve_triangular=solve_triangular,
        DF_MCMC=DF_MCMC, make_add=make_add, make_delete=make_delete,
        weight=weight, KAPPA1=0.5, KAPPA2=1.0,
    )
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source)
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if definitions:
            module = ast.Module(body=definitions, type_ignores=[])
            exec(compile(module, f"{path}:cell-{i}", "exec"), namespace)
    return namespace


def capture_sampler(namespace):
    """Observe the real sampler inputs and traces without changing its behavior."""
    captured = {}

    def run(*args, **kwargs):
        captured["gamma_init"] = np.array(kwargs["gamma_init"], copy=True)
        captured["result"] = DF_MCMC(*args, **kwargs)
        return captured["result"]

    namespace["DF_MCMC"] = run
    return captured


def test_stepwise_initial_model_and_small_experiment():
    # Run the notebook's stepwise -> sampler -> metrics path on a small
    # correlated design whose signal columns are deliberately not {0, 1}.
    helpers = notebook_helpers()
    captured = capture_sampler(helpers)
    rng = np.random.default_rng(832)
    n, p = 80, 6
    covariance = 0.4 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.standard_normal((n, p)) @ cholesky(covariance).T
    beta_star = np.array([0., 0., 3., 0., -2., 0.])
    Y = (X @ beta_star + 0.2 * rng.standard_normal(n))[:, None]
    XTX, XTY = X.T @ X, X.T @ Y
    YTY = float((Y.T @ Y).item())
    g, kappa = p - 1, 1.5
    gamma0 = helpers["forward_backward"](XTX, XTY, YTY, g, kappa, n, p)
    assert gamma0.dtype == bool
    assert np.array_equal(np.flatnonzero(gamma0), [2, 4])

    result = helpers["lit_run"](X, Y, beta_star, gamma0, M=40, seed=12)
    beta_trace, gamma_trace, sigma2_trace = captured["result"]
    assert np.array_equal(captured["gamma_init"], [2, 4])
    assert np.array_equal(gamma_trace[:, 0], gamma0)
    assert np.array_equal(result["sizes"], gamma_trace.sum(axis=0))
    for name in ("T1", "T2"):
        assert result[name].shape == (40,)
        assert np.all(np.isfinite(result[name]))
        assert np.all(result[name] >= 0)
    assert np.all(sigma2_trace > 0)
    assert 0 <= result["acc"] <= 1
    assert result["time"] > 0
    np.testing.assert_allclose(result["T2"], ((X @ beta_trace) ** 2).sum(axis=0))
    mode_count = helpers["local_modes"](
        result["models"], XTX, XTY, YTY, g, kappa, n, p,
    )
    assert 0 <= mode_count <= len(result["models"])


def test_t1_uses_exact_conditional_mean_for_correlated_model():
    helpers = notebook_helpers()
    captured = capture_sampler(helpers)
    X = np.array([[1., 0.], [1., 1.], [0., 1.], [1., 2.]])
    Y = np.array([1., 2., 0., 1.])[:, None]
    beta_star = np.array([1., 0.])
    result = helpers["lit_run"](
        X, Y, beta_star, np.ones(2, dtype=bool), M=8, seed=30,
    )
    gamma_trace = captured["result"][1]
    assert np.all(gamma_trace[:, 0])
    expected = []
    for gamma in gamma_trace.T:
        mean = np.zeros(X.shape[1])
        Xg = X[:, gamma]
        if gamma.any():
            mean[gamma] = 0.5 * np.linalg.solve(Xg.T @ Xg, (Xg.T @ Y).ravel())
        expected.append(np.sum((mean - beta_star) ** 2))
    np.testing.assert_allclose(result["T1"], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result["T1"][0], 1 / 9)


def test_null_model_metrics_are_defined():
    helpers = notebook_helpers()
    captured = capture_sampler(helpers)
    rng = np.random.default_rng(92)
    X = rng.standard_normal((20, 4))
    Y = rng.standard_normal((20, 1))
    beta_star = np.array([1., 0., -2., 0.])
    result = helpers["lit_run"](
        X, Y, beta_star, np.zeros(4, dtype=bool), M=12, seed=70,
    )
    gamma_trace = captured["result"][1]
    assert not gamma_trace[:, 0].any()
    null_steps = ~gamma_trace.any(axis=0)
    assert () in result["models"]
    np.testing.assert_allclose(result["T1"][null_steps], beta_star @ beta_star)
    np.testing.assert_allclose(result["T2"][null_steps], 0)
    assert np.all(np.isfinite(result["T1"]))
    assert np.all(np.isfinite(result["T2"]))


if __name__ == "__main__":
    test_stepwise_initial_model_and_small_experiment()
    test_t1_uses_exact_conditional_mean_for_correlated_model()
    test_null_model_metrics_are_defined()
    print("all notebook regression tests passed")
