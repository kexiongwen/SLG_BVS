#proposal layer of the LIT-MH sampler: the informed thresholded proposal
#distribution (weight, paper eq. (9)-(13)), move-type probabilities (move_prob,
#eq. (12)), proposal state construction with rank-1 Cholesky updates
#(chol_full, make_delete, make_add), and the per-move proposal with its MH
#acceptance ratio (propose)
import numpy as np
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular

try:  # importable both as package (BVS.proposal) and as flat module
    from .cholupdate import chol_add_col, chol_del
except ImportError:
    from cholupdate import chol_add_col, chol_del


def move_prob(s, s0, PA, PD, PS):

    if s == 0:
        h = (1, 0, 0)
    elif s < s0:
        h = (PA, PD, PS)
    else:
        # at capacity, delete and swap each with probability 1/2 (paper eq. (12))
        h = (0, 0.5, 0.5)

    return h


def weight(L, gamma, S, g, kappa, N, XTY, YTY, XTX, ADS, ind=None, p0=None):

    P, _ = np.shape(XTY)
    XTY = XTY.ravel()

    if p0 is None:
        p0 = P

    weight = np.zeros(P)
    trunc_weight = np.zeros(P)

    if ADS == 1:
        # delete: removing the covariate at model position m loses fitted SS
        # h_m^2/C_mm, where C=(X_gamma'X_gamma)^(-1) and h=C X_gamma'y
        if np.any(gamma):
            z = solve_triangular(L, XTY[gamma], lower=True)
            h = solve_triangular(L.T, z)
            c = (solve_triangular(L, np.identity(np.shape(L)[0]), lower=True) ** 2).sum(
                axis=0
            )

            with np.errstate(divide="ignore", invalid="ignore"):
                fit = np.where(c > 0, (h**2) / c, 0.0)

            weight[gamma] = (p0**kappa) * (S / (S + fit)) ** (N / 2)
            trunc_weight[gamma] = np.maximum(np.minimum(weight[gamma], 1), 1 / p0)

    else:
        # add (ADS==0), or the add stage of a swap (ADS==2, ind excluded):
        # one batched triangular solve yields the residual projections u_j, d_j
        # for every candidate at once, replacing a Cholesky update per candidate
        dg = np.diag(XTX)

        if np.shape(L)[0] > 0:
            z = solve_triangular(L, XTY[gamma], lower=True)
            W = solve_triangular(L, XTX[gamma, :], lower=True)
            u = XTY - W.T @ z
            d = dg - (W**2).sum(axis=0)

        else:
            u = XTY.copy()
            d = dg.copy()

        # u_j^2/(d_j*S)<1 exactly by Cauchy-Schwarz; the clip only guards
        # floating point error. Candidates with d_j<=tol are (numerically)
        # collinear with the current model and get weight 0, so a singular
        # model is never proposed
        ok = d > 1e-12 * dg

        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(ok, (u**2) / (d * S), 0.0)
        r = np.minimum(r, 1 - 1e-12)

        weight = np.where(ok, (1 - r) ** (-N / 2) / (p0**kappa), 0.0)
        weight[gamma] = 0

        if ADS == 2 and ind is not None:
            weight[ind] = 0

        trunc_weight = np.where(ok, np.maximum(np.minimum(weight, p0), 1 / p0), 0.0)
        trunc_weight[gamma] = 0

        if ADS == 2 and ind is not None:
            trunc_weight[ind] = 0

    total = trunc_weight.sum()
    weight_normalize = trunc_weight / total if total > 0 else trunc_weight

    result = [weight, weight_normalize]
    return result


def rb_beta(L, gamma, S, g, kappa, N, XTY, YTY, XTX, s0, p0=None):

    # Rao-Blackwellized conditional posterior mean E[beta_j | gamma_{-j}, y]
    # for every j (paper sec. 5.2): the two models differing only in j are
    # averaged with their posterior weights. All ingredients are by-products
    # of the informed proposal computation, so each call costs about one
    # extra proposal. At capacity |gamma| = s0 the model gamma union {j}
    # lies outside M(s0), so out-of-model coordinates are 0

    P = gamma.shape[0]
    XTY_r = np.asarray(XTY).ravel()

    if p0 is None:
        p0 = P

    q2 = g / (1 + g)
    out = np.zeros(P)

    # in-model coordinates: E[beta_j | gamma_{-j}, y] = mu_j / (1 + B(gamma, gamma\{j}))
    pos = np.flatnonzero(gamma)

    if pos.size:
        w_del, _ = weight(L, gamma, S, g, kappa, N, XTY, YTY, XTX, 1, p0=p0)
        LZ = solve_triangular(L, XTY_r[pos], lower=True)
        mu = q2 * solve_triangular(L.T, LZ)
        out[pos] = mu / (1.0 + w_del[pos])

    # out-of-model coordinates: E[beta_j | gamma_{-j}, y] = B*mu / (1 + B),
    # with mu_j = q2*u_j/d_j from the same residual projections as the
    # add-stage weights and B the raw add weight
    sel = ~gamma

    if gamma.sum() < s0:
        dg = np.diag(XTX)

        if pos.size:
            z = solve_triangular(L, XTY_r[pos], lower=True)
            W = solve_triangular(L, XTX[pos, :], lower=True)
            u = XTY_r - W.T @ z
            d = dg - (W**2).sum(axis=0)
        else:
            u = XTY_r.copy()
            d = dg.copy()

        ok = sel & (d > 1e-12 * dg)

        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(ok, (u**2) / (d * S), 0.0)
            mu_out = np.where(ok, q2 * u / d, 0.0)
        r = np.minimum(r, 1 - 1e-12)
        B = np.where(ok, (1 - r) ** (-N / 2) / (p0**kappa), 0.0)

        out[sel] = np.where(ok[sel], B[sel] * mu_out[sel] / (1.0 + B[sel]), 0.0)

    return out


def chol_full(XTX, gamma):

    # full Cholesky of a model's Gram submatrix; fallback when a rank-1
    # update fails

    pos = np.flatnonzero(gamma)

    if pos.size == 0:
        return np.zeros((0, 0))

    return cholesky(XTX[np.ix_(pos, pos)])


def make_delete(L, S, gamma, k, XTX, XTY, YTY, g):

    # proposal state with covariate k removed; if the rank-1 downdate and
    # the full Cholesky both fail, collapses to the current state

    gamma1 = gamma.copy()
    index = gamma[0 : k + 1].sum() - 1
    gamma1[k] = 0

    try:
        L1 = chol_del(L, index)
    except (ValueError, np.linalg.LinAlgError):
        try:
            L1 = chol_full(XTX, gamma1)
        except np.linalg.LinAlgError:
            LZc = solve_triangular(L, XTY[gamma], lower=True)
            return gamma.copy(), L, LZc, S

    LZ1 = solve_triangular(L1, XTY[gamma1], lower=True)
    S1 = (1 + 1 / g) * YTY - (LZ1**2).sum()

    return gamma1, L1, LZ1, S1


def make_add(L, S, gamma, j, XTX, XTY, YTY, g):

    # proposal state with covariate j added; same collapse rule as make_delete

    gamma1 = gamma.copy()
    gamma1[j] = 1
    index = gamma1[0 : j + 1].sum() - 1

    try:
        L1 = chol_add_col(XTX[np.flatnonzero(gamma), j], XTX[j, j], L, index)
    except (ValueError, np.linalg.LinAlgError):
        try:
            L1 = chol_full(XTX, gamma1)
        except np.linalg.LinAlgError:
            LZc = solve_triangular(L, XTY[gamma], lower=True)
            return gamma.copy(), L, LZc, S

    LZ1 = solve_triangular(L1, XTY[gamma1], lower=True)
    S1 = (1 + 1 / g) * YTY - (LZ1**2).sum()

    return gamma1, L1, LZ1, S1


def propose(
    L, gamma, S, ADS, s, s0, h_a, h_d, XTX, XTY, YTY, g, kappa, N, P, P0, PA, PD, PS
):

    # one informed LIT-MH proposal of the requested move type: draws the
    # candidate(s), builds the proposal state, and returns it together with
    # the MH acceptance ratio; returns None if the move has no candidate

    if ADS == 0:
        w_add, q_add = weight(L, gamma, S, g, kappa, N, XTY, YTY, XTX, 0, p0=P0)

        if q_add.sum() == 0:
            return None

        j = np.random.choice(P, 1, p=q_add).item()
        gamma1, L1, LZ1, S1 = make_add(L, S, gamma, j, XTX, XTY, YTY, g)

        w_del, q_del = weight(L1, gamma1, S1, g, kappa, N, XTY, YTY, XTX, 1, p0=P0)
        h_d_rev = move_prob(s + 1, s0, PA, PD, PS)[1]

        ratio = (h_d_rev * q_del[j]) / (h_a * q_add[j]) * w_add[j]

        return gamma1, L1, LZ1, S1, ratio

    elif ADS == 1:
        w_del, q_del = weight(L, gamma, S, g, kappa, N, XTY, YTY, XTX, 1, p0=P0)
        k = np.random.choice(P, 1, p=q_del).item()
        gamma1, L1, LZ1, S1 = make_delete(L, S, gamma, k, XTX, XTY, YTY, g)

        w_add, q_add = weight(L1, gamma1, S1, g, kappa, N, XTY, YTY, XTX, 0, p0=P0)
        h_a_rev = move_prob(s - 1, s0, PA, PD, PS)[0]

        ratio = (h_a_rev * q_add[k]) / (h_d * q_del[k]) * w_del[k]

        return gamma1, L1, LZ1, S1, ratio

    else:
        w_del, q_del = weight(L, gamma, S, g, kappa, N, XTY, YTY, XTX, 1, p0=P0)
        k = np.random.choice(P, 1, p=q_del).item()
        gamma0, L0, LZ0, S0 = make_delete(L, S, gamma, k, XTX, XTY, YTY, g)

        w_add, q_add = weight(L0, gamma0, S0, g, kappa, N, XTY, YTY, XTX, 2, k, p0=P0)

        if q_add.sum() == 0:
            return None

        j = np.random.choice(P, 1, p=q_add).item()
        gamma1, L1, LZ1, S1 = make_add(L0, S0, gamma0, j, XTX, XTY, YTY, g)

        _, q_del1 = weight(L1, gamma1, S1, g, kappa, N, XTY, YTY, XTX, 1, p0=P0)
        _, q_add1 = weight(L0, gamma0, S0, g, kappa, N, XTY, YTY, XTX, 2, j, p0=P0)

        # move-type probabilities cancel between forward and reverse swap
        ratio = (w_add[j] * w_del[k]) * (
            (q_add1[k] * q_del1[j]) / (q_add[j] * q_del[k])
        )

        return gamma1, L1, LZ1, S1, ratio
