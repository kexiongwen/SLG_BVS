"""RB regressions; run with python tests/test_rb_regressions.py."""

import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BVS.DF_MCMC import DF_MCMC
from BVS.proposal import _add_stage, _del_stage, rb_beta


def evaluate_rb(X, Y, gamma, s0, reuse=False):
    n, p = X.shape
    g = p**2 - 1
    xtx, xty = X.T @ X, X.T @ Y
    yty = np.sum(Y**2)
    pos = np.flatnonzero(gamma)
    L = np.linalg.cholesky(xtx[np.ix_(pos, pos)])
    LZ = np.linalg.solve(L, xty[pos])
    S = (1 + 1 / g) * yty - np.sum(LZ**2)
    cached = {}
    if reuse:
        del_w, _ = _del_stage(L, gamma, S, g, 2, n, xty, p, LZ=LZ)
        _, _, u, d, ok = _add_stage(L, gamma, S, g, 2, n, xty, xtx, p, LZ=LZ)
        cached = dict(LZ=LZ, del_w=del_w, add_uv=(u, d, ok))
    return rb_beta(L, gamma, S, g, 2, n, xty, yty, xtx, s0, **cached)


def model_posterior(X, Y, gamma):
    # Independent least-squares posterior, without Cholesky stage helpers or
    # their posterior-ratio identities.
    n, p = X.shape
    g = p**2 - 1
    selected = X[:, gamma]
    coefficients = np.linalg.lstsq(selected, Y, rcond=None)[0]
    residual = Y - selected @ coefficients
    S = np.sum(Y**2) / g + np.sum(residual**2)
    log_weight = -2 * np.sum(gamma) * np.log(p) - n / 2 * np.log(S)
    mean = np.zeros(p)
    mean[gamma] = g / (1 + g) * coefficients.ravel()
    return log_weight, mean


def exact_conditional_means(X, Y, gamma, s0):
    result = np.zeros(len(gamma))
    for j in range(len(gamma)):
        smaller = gamma.copy()
        smaller[j] = False
        larger = smaller.copy()
        larger[j] = True
        if larger.sum() > s0:
            continue
        log_without, _ = model_posterior(X, Y, smaller)
        log_with, mean_with = model_posterior(X, Y, larger)
        log_weights = np.array([log_without, log_with])
        weights = np.exp(log_weights - log_weights.max())
        result[j] = weights[1] / weights.sum() * mean_with[j]
    return result


def strong_signal_problem():
    n = 3000
    Q = np.linalg.qr(np.random.default_rng(9).normal(size=(n, 3)))[0]
    X = np.sqrt(n) * Q
    return X, X[:, [0]] + X[:, [1]]


def test_strong_signal_rb_is_finite_with_and_without_reuse():
    X, Y = strong_signal_problem()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for indices in ([], [0], [1], [0, 1]):
            gamma = np.zeros(3, dtype=bool)
            gamma[indices] = True
            expected = exact_conditional_means(X, Y, gamma, s0=2)
            for reuse in (False, True):
                actual = evaluate_rb(X, Y, gamma, s0=2, reuse=reuse)
                assert np.all(np.isfinite(actual))
                assert np.allclose(actual, expected, rtol=1e-10, atol=1e-12)
        # With an absent strong predictor, inclusion tends to one and its
        # conditional coefficient tends to the g-prior mean, 8/9 here.
        absent = evaluate_rb(X, Y, np.array([True, False, False]), s0=2)
        assert np.isclose(absent[1], 8 / 9)


def test_rb_matches_independent_finite_conditional_means():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(30, 5))
    X[:, 1] += .7 * X[:, 0]
    Y = .4 * X[:, [0]] - .3 * X[:, [2]] + rng.normal(size=(30, 1))
    for size in range(4):
        for indices in combinations(range(5), size):
            gamma = np.zeros(5, dtype=bool)
            gamma[list(indices)] = True
            expected = exact_conditional_means(X, Y, gamma, s0=3)
            for reuse in (False, True):
                actual = evaluate_rb(X, Y, gamma, s0=3, reuse=reuse)
                assert np.allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_rb_preserves_zero_for_collinear_candidate():
    rng = np.random.default_rng(22)
    X = rng.normal(size=(40, 3))
    X[:, 1] = X[:, 0]
    Y = X[:, [0]] + rng.normal(size=(40, 1))
    gamma = np.array([True, False, False])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for reuse in (False, True):
            actual = evaluate_rb(X, Y, gamma, s0=2, reuse=reuse)
            assert np.all(np.isfinite(actual))
            assert actual[1] == 0


def test_strong_signal_rb_trace_and_online_are_finite_and_agree():
    # This positive-burn-in run formerly accumulated NaN as soon as an
    # underfitted model gave an absent true predictor an overflowing BF.
    X, Y = strong_signal_problem()
    kwargs = dict(M=20, burn_in=1, gamma_init=[], seed=2, RB=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        gamma_trace, rb_trace = DF_MCMC(Y, X, 1, 1, 2, 0, **kwargs)
        stats = DF_MCMC(Y, X, 1, 1, 2, 0, online=True, **kwargs)
    assert np.any(gamma_trace.sum(axis=0) < 2)
    assert np.all(np.isfinite(rb_trace))
    assert np.all(np.isfinite(stats["beta_rb"]))
    assert np.allclose(rb_trace.mean(axis=1), stats["beta_rb"], atol=1e-12)
    for gamma, actual in zip(gamma_trace.T, rb_trace.T):
        expected = exact_conditional_means(X, Y, gamma, s0=2)
        assert np.allclose(actual, expected, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    test_strong_signal_rb_is_finite_with_and_without_reuse()
    test_rb_matches_independent_finite_conditional_means()
    test_rb_preserves_zero_for_collinear_candidate()
    test_strong_signal_rb_trace_and_online_are_finite_and_agree()
    print("all RB regression tests passed")
