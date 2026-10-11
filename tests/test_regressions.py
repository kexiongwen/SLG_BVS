# regression tests for sampler correctness; run from the repository root with
#   python3 -m pytest tests/ -v          (or: python3 tests/test_regressions.py)
import sys
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BVS.DF_MCMC import DF_MCMC, sample_beta


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


def test_beta_conditional_covariance():
    # A finite set of centered noise vectors has covariance exactly I, so
    # this checks the conditional mean/covariance without Monte Carlo error.
    gram = np.array([[4.0, 2.0], [2.0, 2.0]])
    L = np.linalg.cholesky(gram)
    xty = np.array([[1.0], [-1.0]])
    LZ = np.linalg.solve(L, xty)
    gamma = np.array([True, False, True])
    g, S, sigma2 = 9.0, 2.0, 1.7
    shrinkage = g / (1 + g)
    noise = np.sqrt(2) * np.array([[1.0, 0.0], [-1.0, 0.0],
                                  [0.0, 1.0], [0.0, -1.0]])

    class BasisRng:
        def __init__(self):
            self.vectors = iter(noise)

        def standard_normal(self, shape):
            return next(self.vectors).reshape(shape)

    beta_s = np.zeros((gamma.size, len(noise)))
    sigma2_s = np.zeros(len(noise))
    rng = BasisRng()
    # Hold sigma2 fixed to isolate beta | gamma, sigma2, y.
    with patch("BVS.DF_MCMC.invgamma.rvs",
               return_value=sigma2 / (0.5 * S * shrinkage)):
        for i in range(len(noise)):
            sample_beta(L, LZ, S, gamma, i, g, 80, 0, False,
                        None, None, beta_s, sigma2_s, 0.0, rng)

    expected_mean = shrinkage * np.linalg.solve(gram, xty)
    centered = beta_s[gamma] - expected_mean
    covariance = centered @ centered.T / len(noise)
    expected_covariance = shrinkage * sigma2 * np.linalg.inv(gram)
    assert np.allclose(sigma2_s, sigma2)
    assert np.all(beta_s[~gamma] == 0)
    assert np.allclose(beta_s[gamma].mean(axis=1), expected_mean.ravel())
    assert np.allclose(covariance, expected_covariance, rtol=1e-12, atol=1e-12)


def test_no_candidate_self_loops_record_all_posteriors():
    # Screening leaves one variable. At capacity a swap has no candidate,
    # but its self-loop must still contribute beta, sigma2 and RB samples.
    rng = np.random.default_rng(123)
    X = rng.normal(size=(30, 3))
    Y = 3 * X[:, 0] + 0.1 * rng.normal(size=30)
    kw = dict(kappa1=1, kappa2=1, s0=1, s_initial=1, M=100,
              burn_in=10, seed=7, screen=1, gamma_init=[0],
              Sample_beta=True, RB=True)
    beta_s, gamma_s, sigma2_s, rb_s = DF_MCMC(Y, X, **kw)
    stats = DF_MCMC(Y, X, online=True, **kw)

    # Independent two-model conditional mean from the marginal posterior.
    p = X.shape[1]
    g = p**2 - 1
    shrinkage = g / (1 + g)
    yty = Y @ Y
    xx = X[:, 0] @ X[:, 0]
    xy = X[:, 0] @ Y
    residual_null = (1 + 1 / g) * yty
    residual_selected = residual_null - xy**2 / xx
    delete_ratio = p**2 * (residual_selected / residual_null)**(len(Y) / 2)
    expected_rb = shrinkage * xy / xx / (1 + delete_ratio)

    assert stats["proposal"][2] > 0, "test must exercise unavailable swaps"
    assert stats["accept"][2] == 0
    assert gamma_s.all(), "the strong signal should stay in the model"
    assert np.all(beta_s != 0), "a retained beta sample was never written"
    assert np.all(sigma2_s != 1), "a retained sigma2 sample was never written"
    assert np.allclose(rb_s, expected_rb, rtol=1e-12, atol=1e-12)
    assert np.allclose(stats["beta_mean"][0], beta_s.mean())
    assert np.allclose(stats["beta_sd"][0], beta_s.std())
    assert np.allclose(stats["sigma2_mean"], sigma2_s.mean())
    assert np.allclose(stats["beta_rb"][0], expected_rb)
    assert sum(stats["model_counts"].values()) == kw["M"]


if __name__ == "__main__":
    test_rb_trace_alignment_and_tail()
    test_column_scale_invariance()
    test_rng_determinism_and_isolation()
    test_beta_conditional_covariance()
    test_no_candidate_self_loops_record_all_posteriors()
    print("all regression tests passed")
