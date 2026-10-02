# regression tests for two fixed bugs; run from the repository root with
#   python3 -m pytest tests/ -v          (or: python3 tests/test_regressions.py)
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BVS.DF_MCMC import DF_MCMC


def make_problem(scale=1.0, seed=5):
    # small problem whose full posterior can be enumerated: signals in
    # columns 0 and 2, eight noise covariates
    rng = np.random.default_rng(seed)
    N, P = 80, 10
    X = rng.standard_normal((N, P)) * scale
    Y = X[:, [0, 2]] @ np.array([[3.0], [2.0]]) + rng.standard_normal((N, 1)) * scale
    return Y, X


def exact_pip(Y, X, kappa1=1, kappa2=1, s0=6):
    # posterior inclusion probabilities from exact enumeration of all models
    # with |gamma| <= s0 under the g-prior posterior (paper eq. (4))
    Y = np.asarray(Y).ravel()
    N, P = X.shape
    g = P ** (2 * kappa1) - 1
    kappa = kappa1 + kappa2

    models, logw = [], []
    for k in range(s0 + 1):
        for comb in combinations(range(P), k):
            Xg = X[:, list(comb)]
            rss = Y @ Y - Y @ (Xg @ np.linalg.solve(Xg.T @ Xg, Xg.T @ Y))
            logw.append(-kappa * np.log(P) * k - (N / 2) * np.log(Y @ Y / g + rss))
            models.append(comb)

    w = np.exp(np.array(logw) - max(logw))
    w /= w.sum()

    pip = np.zeros(P)
    for comb, wt in zip(models, w):
        pip[list(comb)] += wt
    return pip


def test_rb_trace_alignment_and_tail():
    # regression (DF_MCMC RB trace mode): rb_sample was written at column
    # i - burn_in while being sliced at [burn_in:], so the last burn_in
    # columns of the returned trace were never written and the trace was
    # shifted by burn_in iterations relative to the gamma trace
    Y, X = make_problem()
    M, burn = 3000, 500

    gamma_s, rb_s = DF_MCMC(Y, X, 1, 1, 6, 2, M=M, burn_in=burn, seed=11, RB=True)
    stats = DF_MCMC(Y, X, 1, 1, 6, 2, M=M, burn_in=burn, seed=11, RB=True, online=True)

    assert rb_s.shape == (X.shape[1], M)
    assert not np.all(rb_s[:, -burn:] == 0), "RB trace tail was never written"

    # same seed -> identical chain, so the trace mean must reproduce the
    # online RB estimate (which computes the same average correctly)
    assert np.allclose(rb_s.mean(axis=1), stats["beta_rb"], rtol=1e-10, atol=1e-12)

    # sanity: strong signals recovered
    assert rb_s.mean(axis=1)[0] > 1.5


def test_column_scale_invariance():
    # regression (cholupdate): the absolute zeroing R[|R| < 1e-5] = 0 broke
    # the scale invariance of the gamma posterior (the g-prior makes the
    # posterior of gamma exactly invariant to column rescaling). At small
    # column scales the Cholesky factor entries fall below the threshold,
    # the factor is corrupted, and the sampler either crashes with a
    # LinAlgError (scale 1e-7, diagonal zeroed) or returns biased PIPs
    for scale in (1.0, 1e-3, 1e-7):
        Y, X = make_problem(scale=scale)
        pip = DF_MCMC(Y, X, 1, 1, 6, 2, M=6000, burn_in=1000, seed=2).mean(axis=1)
        exact = exact_pip(Y, X)

        assert np.max(np.abs(pip - exact)) < 0.05, (
            f"scale={scale}: sampler PIP {np.round(pip[:4], 3)} vs "
            f"exact {np.round(exact[:4], 3)}"
        )


def test_rng_determinism_and_isolation():
    # RNG migration to numpy Generator: the same seed yields bit-identical
    # chains, independent of the legacy global np.random state (which
    # earlier versions seeded and consumed)
    Y, X = make_problem()
    kw = dict(M=800, burn_in=200, online=True, Sample_beta=True)

    np.random.seed(123)
    np.random.rand(1000)
    a = DF_MCMC(Y, X, 1, 1, 6, 2, seed=7, **kw)
    b = DF_MCMC(Y, X, 1, 1, 6, 2, seed=7, **kw)

    np.random.seed(98765)
    c = DF_MCMC(Y, X, 1, 1, 6, 2, seed=7, **kw)

    for key in ("pip", "beta_mean", "sigma2_mean", "proposal", "accept"):
        assert np.array_equal(a[key], b[key]), key
        assert np.array_equal(a[key], c[key]), key
    assert a["model_counts"] == b["model_counts"] == c["model_counts"]


if __name__ == "__main__":
    test_rb_trace_alignment_and_tail()
    test_column_scale_invariance()
    test_rng_determinism_and_isolation()
    print("all regression tests passed")
