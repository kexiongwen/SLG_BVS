#effective sample size for DF_MCMC output, computed with ArviZ (bulk ESS by
#default: rank-normalized chains combined with Geyer's initial monotone
#sequence estimator, Vehtari et al. 2021). Chains are passed as a single
#chain (M,) or several chains (n_chains, M). Constant chains return NaN --
#their ESS is undefined (ArviZ itself would report the draw count)
import numpy as np
import arviz as az


def ess(chains, method="bulk", relative=False):

    #ESS of one scalar quantity; NaN for a constant chain

    x = np.asarray(chains, dtype=float)

    if x.ndim == 1:
        x = x[None, :]

    if x.ndim != 2:
        raise ValueError("chains must have shape (M,) or (n_chains, M)")

    if np.all(x == x[:, :1]):
        return np.nan

    return float(az.ess(x, method=method, relative=relative))


def ess_gamma(gamma_sample, method="mean"):

    #per-covariate ESS of the inclusion indicators from a (P, M) trace; the
    #default method "mean" targets the posterior inclusion probability, and
    #constant rows (covariates always in or always out) give NaN

    g = np.asarray(gamma_sample)
    out = np.full(g.shape[0], np.nan)

    for j in range(g.shape[0]):
        row = g[j].astype(float)

        if np.all(row == row[0]):
            continue

        out[j] = ess(row, method=method)

    return out


def ess_beta(beta_sample, method="bulk"):

    #per-coefficient ESS from a (P, M) beta trace; the row of a covariate
    #that is never included stays exactly zero and gives NaN

    b = np.asarray(beta_sample)
    out = np.full(b.shape[0], np.nan)

    for j in range(b.shape[0]):
        row = b[j]

        if np.all(row == row[0]):
            continue

        out[j] = ess(row, method=method)

    return out


def ess_sigma2(sigma2_sample, method="bulk"):

    #ESS of the sigma2 chain

    return ess(sigma2_sample, method=method)


def ess_from_df_mcmc(result, method_beta="bulk", method_gamma="mean"):

    #ESS summary from a DF_MCMC return value: the online stats dict stores
    #no trace and raises. Trace outputs contain gamma alone, [gamma, RB],
    #[beta, gamma, sigma2], or [beta, gamma, sigma2, RB]. RB traces use the
    #coefficient method and are reported separately under "beta_rb".
    #Note that under screening the trace is in screened coordinates

    if isinstance(result, dict):
        raise ValueError("ESS needs the stored trace: run DF_MCMC with online=False")

    out = {}
    rb_sample = None

    if isinstance(result, (list, tuple)):
        if len(result) == 2:
            gamma_sample, rb_sample = result
        elif len(result) in (3, 4):
            beta_sample, gamma_sample, sigma2_sample = result[:3]
            out["beta"] = ess_beta(beta_sample, method=method_beta)
            out["sigma2"] = ess_sigma2(sigma2_sample, method=method_beta)
            if len(result) == 4:
                rb_sample = result[3]
        else:
            raise ValueError(
                "DF_MCMC trace lists must contain 2, 3, or 4 arrays: "
                "[gamma, beta_rb], [beta, gamma, sigma2], or "
                "[beta, gamma, sigma2, beta_rb]"
            )
    else:
        gamma_sample = result

    out["gamma"] = ess_gamma(gamma_sample, method=method_gamma)
    if rb_sample is not None:
        out["beta_rb"] = ess_beta(rb_sample, method=method_beta)

    return out
