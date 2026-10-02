import numpy as np
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular
from scipy.stats import invgamma

try:  # importable both as package (BVS.DF_MCMC) and as flat module
    from .proposal import move_prob, propose, rb_beta
except ImportError:
    from proposal import move_prob, propose, rb_beta


def sample_beta(
    L,
    LZ,
    S,
    gamma,
    i,
    g,
    N,
    w,
    online,
    beta_sum,
    beta_sq,
    beta_sample,
    sigma2_sample,
    sigma2_sum,
    rng,
):

    # posterior draws given the current model:
    # sigma2 ~ InvGamma((w+N)/2, (w+S*g/(1+g))/2), then
    # beta_gamma ~ N(q^2 (X_gamma'X_gamma)^(-1) X_gamma'y,
    #                  q^2 sigma2 (X_gamma'X_gamma)^(-1)),  q^2 = g/(1+g)
    # accumulated into the online summaries or stored in the trace;
    # returns the running sigma2_sum

    q = (g / (1 + g)) ** 0.5

    sigma2_draw = invgamma.rvs((w + N) / 2, random_state=rng) * 0.5 * (
        w + S * (g / (1 + g))
    )

    beta_draw = q * (
        q * solve_triangular(L.T, LZ, check_finite=False)
        + solve_triangular(
            L,
            rng.standard_normal((gamma.sum(), 1)),
            lower=True,
            check_finite=False,
        )
        * sigma2_draw**0.5
    )

    if online:
        beta_sum[gamma] += beta_draw[:, 0]
        beta_sq[gamma] += beta_draw[:, 0] ** 2
        sigma2_sum += sigma2_draw
    else:
        sigma2_sample[i] = sigma2_draw
        beta_sample[gamma, i : i + 1] = beta_draw

    return sigma2_sum


def DF_MCMC(
    Y,
    X,
    kappa1,
    kappa2,
    s0,
    s_initial,
    M=10000,
    burn_in=10000,
    Sample_beta=False,
    online=False,
    screen=None,
    seed=None,
    PA=0.4,
    PD=0.4,
    PS=0.2,
    w=0,
    RB=False,
    gamma_init=None,
):

    # online=True: return online sufficient statistics (pip, model counts,
    # beta/sigma2 summaries, acceptance counts) instead of storing the full
    # (P x total_iters) trace; screen=m: restrict the search to the top m
    # columns ranked by marginal |x_j'y|/||x_j|| (paper eq. (11));
    # PA,PD,PS: add/delete/swap proposal probabilities; w: prior degrees of
    # freedom for sigma2 in the Sample_beta draws; RB=True: track the
    # Rao-Blackwellized estimate of E[beta|y] (paper sec. 5.2), returned as
    # stats['beta_rb'] (online) or appended after the trace outputs; it
    # reuses the weight stages computed during the proposal, so it costs at
    # most one extra stage computation per iteration; gamma_init: iterable of
    # starting columns (working, i.e. possibly screened, coordinates)
    # overriding the random draw of s_initial columns

    if abs(PA + PD + PS - 1) > 1e-8:
        raise ValueError("PA, PD and PS must sum to 1")

    # all randomness flows from this per-call Generator: results for a given
    # seed are exactly reproducible and no global numpy.random state is read
    # or written
    rng = np.random.default_rng(seed)

    Y = np.asarray(Y).reshape(-1, 1)
    N, P0 = np.shape(X)
    g = P0 ** (2 * kappa1) - 1
    kappa = kappa1 + kappa2

    keep = None
    if screen is not None and screen < P0:
        # prior constants g and p^kappa stay tied to the original p
        XTY_full = (X.T @ Y).ravel()
        xnorm = np.sqrt(np.einsum("ij,ij->j", X, X))
        keep = np.argsort(-np.abs(XTY_full) / xnorm)[:screen]
        X = X[:, keep]

    XTX = X.T @ X
    XTY = X.T @ Y
    YTY = (Y.T @ Y).item()
    N, P = np.shape(X)

    # initial model; a user-supplied gamma_init is validated directly,
    # otherwise random draws are retried if (numerically) singular --
    # LAPACK accepts near-zero pivots, so also require the smallest pivot
    # diag(L)^2 to clear a tolerance relative to the Gram scale -- falling
    # back to the null model (whose empty factor is handled throughout)
    if gamma_init is not None:
        gamma = np.zeros(P, dtype=bool)
        gamma[np.asarray(list(gamma_init), dtype=int)] = 1
        pos = np.flatnonzero(gamma)

        if pos.size == 0:
            L = np.zeros((0, 0))
        else:
            try:
                L = cholesky(XTX[np.ix_(pos, pos)])
            except np.linalg.LinAlgError:
                raise ValueError("gamma_init gives a singular model")
            if (np.diag(L) ** 2).min() <= 1e-10 * np.diag(XTX)[pos].max():
                raise ValueError("gamma_init gives a singular model")
    else:
        for _ in range(100):
            gamma = np.zeros(P, dtype=bool)
            gamma[rng.choice(P, size=s_initial, replace=False)] = 1

            pos = np.flatnonzero(gamma)

            if pos.size == 0:
                L = np.zeros((0, 0))
                break

            try:
                L = cholesky(XTX[np.ix_(pos, pos)])
            except np.linalg.LinAlgError:
                continue

            if (np.diag(L) ** 2).min() > 1e-10 * np.diag(XTX)[pos].max():
                break
        else:
            gamma = np.zeros(P, dtype=bool)
            L = np.zeros((0, 0))

    LZ = solve_triangular(L, XTY[gamma], lower=True, check_finite=False)
    S = (1 + 1 / g) * YTY - (LZ**2).sum()

    if online:
        pip_count = np.zeros(P)
        model_counts = {}
        prop = np.zeros(3)
        acc = np.zeros(3)

        if Sample_beta:
            beta_sum = np.zeros(P)
            beta_sq = np.zeros(P)
            sigma2_sum = 0.0
            beta_sample = None
            sigma2_sample = None

        if RB:
            rb_sum = np.zeros(P)

    else:
        gamma_sample = np.zeros((P, M + burn_in), dtype=bool)
        gamma_sample[:, 0] = gamma

        if Sample_beta:
            beta_sample = np.zeros((P, M + burn_in))
            sigma2_sample = np.ones(M + burn_in)
            beta_sum = None
            beta_sq = None
            sigma2_sum = 0.0

        if RB:
            rb_sample = np.zeros((P, M + burn_in))
            rb_sample[:, 0] = rb_beta(
                L, gamma, S, g, kappa, N, XTY, YTY, XTX, s0, p0=P0
            )

    # MCMC loop

    for i in range(1, M + burn_in):
        s = gamma.sum()

        h_a, h_d, h_s = move_prob(s, s0, PA, PD, PS)

        ADS = int(rng.choice(3, p=[h_a, h_d, h_s]))

        if online:
            prop[ADS] += 1

        proposal = propose(
            L,
            gamma,
            S,
            ADS,
            s,
            s0,
            h_a,
            h_d,
            XTX,
            XTY,
            YTY,
            g,
            kappa,
            N,
            P,
            P0,
            PA,
            PD,
            PS,
            LZ,
            rng,
        )

        if proposal is None:
            # the requested move type has no candidate: record the unchanged
            # state and skip the move

            if online:
                if i >= burn_in:
                    pip_count += gamma
                    mkey = tuple(np.flatnonzero(gamma))
                    model_counts[mkey] = model_counts.get(mkey, 0) + 1
            else:
                gamma_sample[:, i] = gamma

            continue

        gamma_proposal, L_proposal, LZ_proposal, S_proposal, ratio, reuse = proposal

        if ratio > 1 or rng.random() < ratio:
            gamma = gamma_proposal
            L = L_proposal
            LZ = LZ_proposal
            S = S_proposal
            rb_reuse = reuse["post"]

            if online:
                acc[ADS] += 1
        else:
            rb_reuse = reuse["pre"]

        if online:
            if i >= burn_in:
                pip_count += gamma
                mkey = tuple(np.flatnonzero(gamma))
                model_counts[mkey] = model_counts.get(mkey, 0) + 1

        else:
            gamma_sample[:, i] = gamma

        if Sample_beta and i >= burn_in:
            sigma2_sum = sample_beta(
                L,
                LZ,
                S,
                gamma,
                i,
                g,
                N,
                w,
                online,
                beta_sum,
                beta_sq,
                beta_sample,
                sigma2_sample,
                sigma2_sum,
                rng,
            )

        if RB and i >= burn_in:
            rb_draw = rb_beta(
                L, gamma, S, g, kappa, N, XTY, YTY, XTX, s0, p0=P0, LZ=LZ, **rb_reuse
            )

            if online:
                rb_sum += rb_draw
            else:
                rb_sample[:, i] = rb_draw

    if online:
        stats = {"n_iter": M, "proposal": prop, "accept": acc}

        pip = pip_count / M
        counts = model_counts

        if keep is not None:
            pip_full = np.zeros(P0)
            pip_full[keep] = pip
            pip = pip_full
            counts = {tuple(sorted(keep[list(m)])): c for m, c in counts.items()}

        stats["pip"] = pip
        stats["model_counts"] = counts

        if Sample_beta:
            beta_mean = beta_sum / M
            beta_sd = np.sqrt(np.maximum(beta_sq / M - beta_mean**2, 0))

            if keep is not None:
                beta_mean_full = np.zeros(P0)
                beta_sd_full = np.zeros(P0)
                beta_mean_full[keep] = beta_mean
                beta_sd_full[keep] = beta_sd
                beta_mean = beta_mean_full
                beta_sd = beta_sd_full

            stats["beta_mean"] = beta_mean
            stats["beta_sd"] = beta_sd
            stats["sigma2_mean"] = sigma2_sum / M

        if RB:
            rb = rb_sum / M

            if keep is not None:
                rb_full = np.zeros(P0)
                rb_full[keep] = rb
                rb = rb_full

            stats["beta_rb"] = rb

        return stats

    if Sample_beta:
        result = [
            beta_sample[:, burn_in:],
            gamma_sample[:, burn_in:],
            sigma2_sample[burn_in:],
        ]

        if RB:
            result.append(rb_sample[:, burn_in:])
    else:
        result = gamma_sample[:, burn_in:]

        if RB:
            result = [gamma_sample[:, burn_in:], rb_sample[:, burn_in:]]

    return result
