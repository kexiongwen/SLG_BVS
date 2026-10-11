"""Sampler boundary regressions; run with python tests/test_sampler_boundaries.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BVS.DF_MCMC import DF_MCMC


def make_problem():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(30, 5))
    Y = 2 * X[:, 1] - X[:, 3] + rng.normal(size=30)
    return Y, X


def assert_value_error(call, message):
    try:
        call()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError("expected ValueError containing " + message)


def test_zero_burn_in_records_initial_state_in_every_mode():
    Y, X = make_problem()
    initial = np.array([False, True, False, True, False])
    for sample_beta in (False, True):
        for rb in (False, True):
            kw = dict(M=1, burn_in=0, seed=7, gamma_init=[1, 3],
                      Sample_beta=sample_beta, RB=rb)
            trace = DF_MCMC(Y, X, 1, 1, 3, 0, **kw)
            stats = DF_MCMC(Y, X, 1, 1, 3, 0, online=True, **kw)
            gamma_s = trace[1] if sample_beta else (trace[0] if rb else trace)
            assert np.array_equal(gamma_s[:, 0], initial)
            assert np.array_equal(stats["pip"], initial)
            assert stats["model_counts"] == {(1, 3): 1}
            assert stats["n_iter"] == 1
            assert not stats["proposal"].any()
            assert not stats["accept"].any()
            if sample_beta:
                beta_s, _, sigma2_s = trace[:3]
                assert np.all(beta_s[initial, 0] != 0)
                assert np.all(beta_s[~initial, 0] == 0)
                assert sigma2_s[0] > 0 and sigma2_s[0] != 1
                assert np.array_equal(stats["beta_mean"], beta_s[:, 0])
                assert np.all(stats["beta_sd"] == 0)
                assert stats["sigma2_mean"] == sigma2_s[0]
            if rb:
                assert np.array_equal(stats["beta_rb"], trace[-1][:, 0])
                assert np.any(stats["beta_rb"] != 0)


def test_zero_burn_in_online_summaries_match_complete_traces():
    Y, X = make_problem()
    for initial in ([], [1, 3]):
        for sample_beta in (False, True):
            for rb in (False, True):
                kw = dict(M=25, burn_in=0, seed=9, gamma_init=initial,
                          Sample_beta=sample_beta, RB=rb)
                trace = DF_MCMC(Y, X, 1, 1, 3, 0, **kw)
                stats = DF_MCMC(Y, X, 1, 1, 3, 0, online=True, **kw)
                gamma_s = trace[1] if sample_beta else (trace[0] if rb else trace)
                assert gamma_s.shape == (X.shape[1], kw["M"])
                assert np.array_equal(stats["pip"], gamma_s.mean(axis=1))
                expected_counts = {}
                for gamma in gamma_s.T:
                    key = tuple(np.flatnonzero(gamma))
                    expected_counts[key] = expected_counts.get(key, 0) + 1
                assert stats["model_counts"] == expected_counts
                assert sum(expected_counts.values()) == kw["M"]
                if sample_beta:
                    beta_s, _, sigma2_s = trace[:3]
                    assert np.all(sigma2_s > 0)
                    assert np.allclose(stats["beta_mean"], beta_s.mean(axis=1))
                    assert np.allclose(stats["beta_sd"], beta_s.std(axis=1))
                    assert np.isclose(stats["sigma2_mean"], sigma2_s.mean())
                if rb:
                    assert np.allclose(stats["beta_rb"], trace[-1].mean(axis=1))


def test_initial_model_must_fit_model_space():
    Y, X = make_problem()
    kw = dict(M=2, burn_in=1, seed=5)
    assert_value_error(lambda: DF_MCMC(Y, X, 1, 1, 2, 3, **kw), "s_initial")
    assert_value_error(
        lambda: DF_MCMC(Y, X, 1, 1, 2, 0, gamma_init=[0, 1, 2], **kw),
        "gamma_init",
    )
    assert_value_error(lambda: DF_MCMC(Y, X, 1, 1, 6, 6, **kw), "s_initial")
    assert_value_error(
        lambda: DF_MCMC(Y, X, 1, 1, 3, 3, screen=2, **kw), "s_initial"
    )
    # Explicit indices override the random initial size; duplicates describe
    # the same subset and should not count against the model-size bound.
    trace = DF_MCMC(Y, X, 1, 1, 2, 99, gamma_init=[1, 1, 3],
                    M=1, burn_in=0, seed=5)
    assert np.array_equal(np.flatnonzero(trace[:, 0]), [1, 3])


def test_initial_model_indices_are_valid():
    Y, X = make_problem()
    for initial in ([-1], [5], [1.5], [True, False], [[1, 2]]):
        assert_value_error(
            lambda: DF_MCMC(Y, X, 1, 1, 3, 0, gamma_init=initial,
                            M=1, burn_in=0), "gamma_init"
        )


def test_iteration_and_model_size_arguments_are_valid():
    Y, X = make_problem()
    for name, value in (("M", 0), ("M", 1.5), ("burn_in", -1),
                        ("burn_in", 0.5), ("s0", -1), ("s0", 1.5),
                        ("s_initial", -1), ("s_initial", 1.5)):
        kw = dict(kappa1=1, kappa2=1, s0=2, s_initial=0, M=2, burn_in=0)
        kw[name] = value
        assert_value_error(lambda: DF_MCMC(Y, X, **kw), name)


def test_prior_and_screen_arguments_are_valid():
    Y, X = make_problem()
    kw = dict(kappa1=1, kappa2=1, s0=2, s_initial=0, M=2, burn_in=0)
    for value in (0, -1, np.nan):
        assert_value_error(
            lambda: DF_MCMC(Y, X, **{**kw, "kappa1": value}), "kappa1")
    for value in (0, -3, 1.5, True):
        assert_value_error(
            lambda: DF_MCMC(Y, X, **{**kw, "screen": value}), "screen")


def test_zero_capacity_keeps_only_the_null_model():
    Y, X = make_problem()
    kw = dict(M=5, burn_in=0, seed=3, Sample_beta=True, RB=True)
    beta_s, gamma_s, sigma2_s, rb_s = DF_MCMC(Y, X, 1, 1, 0, 0, **kw)
    stats = DF_MCMC(Y, X, 1, 1, 0, 0, online=True, **kw)
    assert not gamma_s.any() and not beta_s.any() and not rb_s.any()
    assert np.all(sigma2_s > 0) and np.all(sigma2_s != 1)
    assert stats["model_counts"] == {(): kw["M"]}
    assert not stats["proposal"].any()
    assert np.isclose(stats["sigma2_mean"], sigma2_s.mean())


if __name__ == "__main__":
    test_zero_burn_in_records_initial_state_in_every_mode()
    test_zero_burn_in_online_summaries_match_complete_traces()
    test_initial_model_must_fit_model_space()
    test_initial_model_indices_are_valid()
    test_iteration_and_model_size_arguments_are_valid()
    test_prior_and_screen_arguments_are_valid()
    test_zero_capacity_keeps_only_the_null_model()
    print("all sampler boundary regression tests passed")
