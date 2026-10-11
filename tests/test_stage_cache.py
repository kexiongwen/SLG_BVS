"""Bounded stage-cache tests; run with python tests/test_stage_cache.py."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BVS import proposal


class Choices:
    def __init__(self, *indices):
        self.indices = iter(indices)
        self.probabilities = []

    def choice(self, size, p):
        assert len(p) == size
        assert np.isclose(p.sum(), 1)
        self.probabilities.append(p.copy())
        result = next(self.indices)
        assert p[result] > 0
        return result


def problem(zero=False):
    rng = np.random.default_rng(26)
    X = rng.normal(size=(36, 9))
    Y = .4 * X[:, [1]] + rng.normal(size=(36, 1))
    if zero:
        X[:] = 0
    return dict(X=X, Y=Y, XTX=X.T @ X, XTY=X.T @ Y, YTY=np.sum(Y**2),
                N=X.shape[0], P=X.shape[1], P0=X.shape[1], g=X.shape[1]**2-1,
                kappa=2, s0=4, PA=.4, PD=.4, PS=.2)


def state(data, indices):
    gamma = np.zeros(data["P"], dtype=bool)
    gamma[list(indices)] = True
    pos = np.flatnonzero(gamma)
    L = np.linalg.cholesky(data["XTX"][np.ix_(pos, pos)])
    LZ = np.linalg.solve(L, data["XTY"][pos])
    S = (1 + 1 / data["g"]) * data["YTY"] - np.sum(LZ**2)
    return dict(L=L, gamma=gamma, S=S, LZ=LZ)


def propose(data, current, ads, choices, cache):
    s = current["gamma"].sum()
    ha, hd, _ = proposal.move_prob(s, data["s0"], data["PA"], data["PD"], data["PS"])
    arguments = {key: value for key, value in data.items() if key not in ("X", "Y")}
    return proposal.propose(**arguments, **current, ADS=ads, s=s, h_a=ha, h_d=hd,
                            rng=choices, cache=cache)


def rb(data, current, cache):
    keys = ("g", "kappa", "N", "XTY", "YTY", "XTX", "s0")
    arguments = {key: data[key] for key in keys}
    return proposal.rb_beta(**arguments, **current, p0=data["P0"], cache=cache)


def log_posterior(data, gamma):
    selected = data["X"][:, gamma]
    residual = data["Y"] - selected @ np.linalg.lstsq(selected, data["Y"], rcond=None)[0]
    S = data["YTY"] / data["g"] + np.sum(residual**2)
    return -data["kappa"] * gamma.sum() * np.log(data["P0"]) - data["N"] / 2 * np.log(S)


def independent_probabilities(data, gamma, addition):
    result = np.zeros(data["P"])
    candidates = np.flatnonzero(~gamma if addition else gamma)
    for j in candidates:
        other = gamma.copy()
        other[j] = addition
        raw = np.exp(log_posterior(data, other) - log_posterior(data, gamma))
        result[j] = np.clip(raw, 1 / data["P0"], data["P0"] if addition else 1)
    return result / result.sum()


def test_rejected_state_reuses_stages_and_rb_without_growing_cache():
    data, cache = problem(), {}
    current = state(data, [1, 6])
    with patch.object(proposal, "_add_stage", wraps=proposal._add_stage) as add:
        with patch.object(proposal, "_del_compact", wraps=proposal._del_compact) as delete:
            first = propose(data, current, 0, Choices(3), cache)
            second = propose(data, current, 0, Choices(3), cache)
            assert add.call_count == 1
            assert delete.call_count == 2  # each proposed model needs a reverse stage
            assert first[4] == second[4]
            assert first[5]["pre"]["cache"] is cache
            assert first[5]["post"]["cache"] is not cache
            expected = rb(data, current, cache)
            stage_counts = (add.call_count, delete.call_count)
            keys = set(cache)
            for _ in range(3):
                result = rb(data, current, cache)
                assert np.array_equal(result, expected)
                result[:] = -999  # callers must not corrupt the stored RB result
            assert stage_counts == (add.call_count, delete.call_count)
            assert set(cache) == keys
            assert np.array_equal(rb(data, current, cache), expected)
    assert all(not isinstance(value, dict) for value in cache.values())


def test_accepted_state_reuses_proposed_delete_stage():
    data, cache = problem(), {}
    current = state(data, [1, 6])
    with patch.object(proposal, "_del_compact", wraps=proposal._del_compact) as delete:
        result = propose(data, current, 0, Choices(3), cache)
        accepted = dict(gamma=result[0], L=result[1], LZ=result[2], S=result[3])
        next_cache = result[5]["post"]["cache"]
        assert delete.call_count == 1
        propose(data, accepted, 1, Choices(1), next_cache)
        assert delete.call_count == 1
        assert np.array_equal(next_cache["pos"], [1, 3, 6])


def test_cache_invalidates_for_replacement_state_and_context():
    data, cache = problem(), {}
    current = state(data, [1, 6])
    rb(data, current, cache)
    # A different model, even with the same cardinality, must replace all data.
    replacement = state(data, [2, 5])
    assert np.allclose(rb(data, replacement, cache), rb(data, replacement, None))
    assert np.array_equal(cache["pos"], [2, 5])
    old_stage = cache["delete"]
    copied = {**replacement, "gamma": replacement["gamma"].copy(), "L": replacement["L"].copy()}
    rb(data, copied, cache)
    assert cache["delete"] is not old_stage
    # Fixed-data identity and scalar posterior/capacity changes also invalidate.
    for updates in ({"XTY": data["XTY"].copy()}, {"kappa": 3}, {"s0": 2}, {"P0": 11}):
        changed = {**data, **updates}
        old_stage = cache["delete"]
        assert np.allclose(rb(changed, replacement, cache), rb(changed, replacement, None))
        assert cache["delete"] is not old_stage
        assert len(cache) <= 10


def test_compact_delete_distribution_and_add_delete_mh_ratios():
    data = problem()
    current = state(data, [1, 6])
    gamma = current["gamma"]
    pos = np.flatnonzero(gamma)
    expected_delete = independent_probabilities(data, gamma, addition=False)
    raw_dense, trunc_dense = proposal._del_stage(
        current["L"], gamma, current["S"], data["g"], data["kappa"],
        data["N"], data["XTY"], data["P0"], current["LZ"]
    )
    assert raw_dense.shape == trunc_dense.shape == (data["P"],)
    assert np.all(trunc_dense[~gamma] == 0)
    for local, k in enumerate(pos):
        choices, cache = Choices(local), {}
        result = propose(data, current, 1, choices, cache)
        assert choices.probabilities[0].shape == (len(pos),)
        assert np.allclose(choices.probabilities[0], expected_delete[pos])
        assert all(values.shape == (len(pos),) for values in cache["delete"])
        other = result[0]
        reverse = independent_probabilities(data, other, addition=True)[k]
        expected = np.exp(log_posterior(data, other) - log_posterior(data, gamma)) * reverse / expected_delete[k]
        assert np.isclose(result[4], expected, rtol=1e-10)
    expected_add = independent_probabilities(data, gamma, addition=True)
    for j in np.flatnonzero(~gamma):
        result = propose(data, current, 0, Choices(j), {})
        reverse = independent_probabilities(data, result[0], addition=False)[j]
        expected = np.exp(log_posterior(data, result[0]) - log_posterior(data, gamma)) * reverse / expected_add[j]
        assert np.isclose(result[4], expected, rtol=1e-10)
    public = proposal.weight(
        current["L"], gamma, current["S"], data["g"], data["kappa"], data["N"],
        data["XTY"], data["YTY"], data["XTX"], ADS=1, p0=data["P0"], LZ=current["LZ"]
    )
    assert np.allclose(public[0], raw_dense)
    assert np.allclose(public[1], expected_delete)


def test_null_model_with_no_add_candidates_is_cached():
    data, cache = problem(zero=True), {}
    current = state(data, [])
    with patch.object(proposal, "_add_stage", wraps=proposal._add_stage) as add:
        assert propose(data, current, 0, Choices(), cache) is None
        assert propose(data, current, 0, Choices(), cache) is None
        assert np.array_equal(rb(data, current, cache), np.zeros(data["P"]))
        assert add.call_count == 1


def test_scalar_reverse_probability_handles_cancellation_and_zero():
    for weights in (np.array([1e30, 1., 2.]), np.array([0., 2., 0.]), np.zeros(3),
                    np.array([.2, 1., 3.])):
        for excluded in range(len(weights)):
            expected = proposal._exclude_normalize(weights, excluded)
            for candidate in range(len(weights)):
                actual = proposal._excluded_probability(weights, weights.sum(), excluded, candidate)
                assert np.isclose(actual, expected[candidate], rtol=1e-12, atol=0)


if __name__ == "__main__":
    test_rejected_state_reuses_stages_and_rb_without_growing_cache()
    test_accepted_state_reuses_proposed_delete_stage()
    test_cache_invalidates_for_replacement_state_and_context()
    test_compact_delete_distribution_and_add_delete_mh_ratios()
    test_null_model_with_no_add_candidates_is_cached()
    test_scalar_reverse_probability_handles_cancellation_and_zero()
    print("all stage-cache tests passed")
