# Run with pytest or directly: python tests/test_ess_regressions.py
import sys
from pathlib import Path

import arviz as az
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BVS.DF_MCMC import DF_MCMC
from BVS.ESS import ess_from_df_mcmc


def direct_ess(row, method):
    # Independent ArviZ calculation, retaining the public constant-row policy.
    row = np.asarray(row, dtype=float)
    if np.all(row == row[0]):
        return np.nan
    return float(az.ess(row[None, :], method=method))


def test_all_sampler_trace_modes():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(40, 5))
    Y = 0.7 * X[:, 0] + rng.normal(size=40)

    for sample_beta in (False, True):
        for rb in (False, True):
            result = DF_MCMC(
                Y, X, 1, 1, 3, 1, M=300, burn_in=40, seed=7,
                Sample_beta=sample_beta, RB=rb,
            )
            traces = {}
            if sample_beta:
                traces.update(beta=result[0], gamma=result[1], sigma2=result[2])
                if rb:
                    traces["beta_rb"] = result[3]
            elif rb:
                traces.update(gamma=result[0], beta_rb=result[1])
            else:
                traces["gamma"] = result

            # Check the sampler's list output and the equivalent public tuple
            # form, with defaults and distinct nondefault method selections.
            inputs = (result, tuple(result)) if isinstance(result, list) else (result,)
            for method_beta, method_gamma in (("bulk", "mean"), ("mean", "bulk")):
                expected = {}
                for key, trace in traces.items():
                    method = method_gamma if key == "gamma" else method_beta
                    expected[key] = (
                        direct_ess(trace, method) if key == "sigma2" else
                        np.array([direct_ess(row, method) for row in trace])
                    )
                for value in inputs:
                    actual = ess_from_df_mcmc(value, method_beta, method_gamma)
                    assert actual.keys() == expected.keys()
                    for key in expected:
                        np.testing.assert_allclose(
                            actual[key], expected[key], equal_nan=True,
                            err_msg=f"Sample_beta={sample_beta}, RB={rb}, key={key}",
                        )


def test_constant_rb_rows_and_rejected_outputs():
    gamma = np.zeros((2, 12), dtype=bool)
    rb = np.ones((2, 12))
    summary = ess_from_df_mcmc([gamma, rb])
    assert np.isnan(summary["gamma"]).all()
    assert np.isnan(summary["beta_rb"]).all()

    for size in (0, 1, 5):
        for malformed in ([gamma] * size, (gamma,) * size):
            try:
                ess_from_df_mcmc(malformed)
            except ValueError as exc:
                assert "2, 3, or 4 arrays" in str(exc)
            else:
                raise AssertionError(f"accepted a malformed result of length {size}")

    X = np.eye(4)
    stats = DF_MCMC(np.arange(1.0, 5.0), X, 1, 1, 2, 1,
                    M=8, burn_in=2, seed=7, online=True, RB=True)
    try:
        ess_from_df_mcmc(stats)
    except ValueError as exc:
        assert "online=False" in str(exc)
    else:
        raise AssertionError("accepted online statistics without a stored trace")


if __name__ == "__main__":
    test_all_sampler_trace_modes()
    test_constant_rb_rows_and_rejected_outputs()
    print("all ESS regression tests passed")
