"""Proposal regressions; run with python tests/test_proposal_regressions.py."""

import sys
import warnings
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BVS import proposal
from BVS.DF_MCMC import DF_MCMC


class FixedChoices:
    """Select a particular supported move without relying on random seeds."""

    def __init__(self, choices):
        self.choices = iter(choices)

    def choice(self, size, p):
        selected = next(self.choices)
        assert 0 <= selected < size and p[selected] > 0
        assert np.isclose(np.sum(p), 1)
        return selected


def call_swap(X, Y, gamma, k, j, s0):
    n, p = X.shape
    g = p**2 - 1
    xtx, xty = X.T @ X, X.T @ Y
    yty = (Y.T @ Y).item()
    pos = np.flatnonzero(gamma)
    L = np.linalg.cholesky(xtx[np.ix_(pos, pos)])
    LZ = np.linalg.solve(L, xty[pos])
    S = (1 + 1 / g) * yty - np.sum(LZ**2)
    ha, hd, _ = proposal.move_prob(len(pos), s0, .4, .4, .2)
    return proposal.propose(
        L, gamma, S, 2, len(pos), s0, ha, hd, xtx, xty, yty,
        g, 2, n, p, p, .4, .4, .2, LZ,
        FixedChoices([int(np.searchsorted(pos, k)), j])
    )


def strong_signal_problem(second_coefficient=1.0):
    n = 4000
    x0 = np.tile([1., -1.], n // 2)
    x1 = np.tile([1., 1., -1., -1.], n // 4)
    noise = np.tile([1., -1., -1., 1.], n // 4)
    X = np.column_stack([x0, x1])
    Y = (x0 + second_coefficient * x1 + .1 * noise)[:, None]
    return X, Y


def test_equal_strong_signal_swaps_are_accepted():
    # The intermediate add/delete factors are inf and 0, although the two
    # full-rank singleton models have exactly the same posterior probability.
    X, Y = strong_signal_problem()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for k, j in [(0, 1), (1, 0)]:
            gamma = np.array([False, False])
            gamma[k] = True
            result = call_swap(X, Y, gamma, k, j, s0=1)
            assert np.isfinite(result[4])
            assert np.isclose(result[4], 1, rtol=1e-10)
            assert np.array_equal(result[0], ~gamma)

        stats = DF_MCMC(
            Y, X, 1, 1, 1, 1, M=200, burn_in=10, online=True,
            gamma_init=[0], seed=1
        )
    assert stats["proposal"][2] > 0
    assert stats["accept"][2] == stats["proposal"][2]
    assert np.all(stats["pip"] > .3) and np.all(stats["pip"] < .7)


def log_posterior(X, Y, gamma):
    # Independent least-squares calculation, without proposal stage helpers.
    n, p = X.shape
    selected = X[:, gamma]
    fitted = selected @ np.linalg.lstsq(selected, Y, rcond=None)[0]
    residual = Y - fitted
    S = np.sum(Y**2) / (p**2 - 1) + np.sum(residual**2)
    return -2 * np.sum(gamma) * np.log(p) - n / 2 * np.log(S)


def swap_probability(X, Y, gamma, k, j):
    p = len(gamma)
    deletion = np.zeros(p)
    lp = log_posterior(X, Y, gamma)
    for removed in np.flatnonzero(gamma):
        smaller = gamma.copy()
        smaller[removed] = False
        bf = np.exp(log_posterior(X, Y, smaller) - lp)
        deletion[removed] = np.clip(bf, 1 / p, 1)

    middle = gamma.copy()
    middle[k] = False
    lp_middle = log_posterior(X, Y, middle)
    addition = np.zeros(p)
    for added in np.flatnonzero(~middle):
        if added == k:
            continue
        larger = middle.copy()
        larger[added] = True
        bf = np.exp(log_posterior(X, Y, larger) - lp_middle)
        addition[added] = np.clip(bf, 1 / p, p)
    return deletion[k] / deletion.sum() * addition[j] / addition.sum()


def test_swap_ratios_match_independent_detailed_balance():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(24, 5))
    X[:, 1] += .8 * X[:, 0]
    Y = .6 * X[:, [0]] + rng.normal(size=(24, 1))
    nonunit = 0
    for size in (1, 2, 3):
        for indices in combinations(range(X.shape[1]), size):
            gamma = np.zeros(X.shape[1], dtype=bool)
            gamma[list(indices)] = True
            for k in np.flatnonzero(gamma):
                for j in np.flatnonzero(~gamma):
                    other = gamma.copy()
                    other[k], other[j] = False, True
                    q_forward = swap_probability(X, Y, gamma, k, j)
                    q_reverse = swap_probability(X, Y, other, j, k)
                    expected = np.exp(
                        log_posterior(X, Y, other) - log_posterior(X, Y, gamma)
                    ) * q_reverse / q_forward
                    actual = call_swap(X, Y, gamma, k, j, s0=3)[4]
                    assert np.isclose(actual, expected, rtol=1e-10, atol=1e-14)
                    nonunit += not np.isclose(expected, 1)
    assert nonunit > 0


def test_extreme_swap_ratios_have_valid_endpoints():
    # An extreme posterior ratio legitimately rounds to infinity or zero;
    # both give the correct MH decision and must not produce NaN or warnings.
    X, Y = strong_signal_problem(second_coefficient=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        forward = call_swap(X, Y, np.array([True, False]), 0, 1, s0=1)[4]
        reverse = call_swap(X, Y, np.array([False, True]), 1, 0, s0=1)[4]
    assert forward == np.inf
    assert reverse == 0


def test_swap_rejects_when_failed_add_has_no_reverse_delete():
    # make_add falls back to its input model if both Cholesky attempts fail.
    # That candidate lacks j, so reverse deletion has zero probability.
    def failed_add(L, S, gamma, j, XTX, XTY, YTY, g):
        LZ = np.linalg.solve(L, XTY[gamma])
        return gamma.copy(), L, LZ, S

    X, Y = strong_signal_problem()
    with patch.object(proposal, "make_add", side_effect=failed_add):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = call_swap(X, Y, np.array([True, False]), 0, 1, s0=1)
    assert result[4] == 0


if __name__ == "__main__":
    test_equal_strong_signal_swaps_are_accepted()
    test_swap_ratios_match_independent_detailed_balance()
    test_extreme_swap_ratios_have_valid_endpoints()
    test_swap_rejects_when_failed_add_has_no_reverse_delete()
    print("all proposal regression tests passed")
