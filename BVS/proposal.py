#proposal layer of the LIT-MH sampler: the informed thresholded proposal
#distribution (weight, paper eq. (9)-(13)), move-type probabilities (move_prob,
#eq. (12)), proposal state construction with rank-1 Cholesky updates
#(chol_full, make_delete, make_add), and the per-move proposal with its MH
#acceptance ratio (propose). The weight computation is split into per-move-type
#stages (_del_stage, _add_stage) whose results are pure functions of the model
#state: propose() reuses one add-stage computation for both swap exclusions,
#and the Rao-Blackwellized estimator (rb_beta) reuses the stages already
#computed for the current state in the same iteration
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


def _normalize(trunc):

    # normalized proposal distribution from truncated weights

    total = trunc.sum()

    return trunc / total if total > 0 else trunc


def _exclude_normalize(trunc, ind):

    # normalized proposal distribution with candidate ind excluded: the two
    # exclusions of a swap share one _add_stage computation, so only the
    # renormalization (copy, zero, sum) is redone

    t = trunc.copy()
    t[ind] = 0.0

    return _normalize(t)


def _del_stage(L, gamma, S, g, kappa, N, XTY, p0, LZ=None):

    # delete-stage weights: raw B(gamma, gamma\{k}) for k in gamma (0
    # elsewhere) and the truncated copy clipped to [1/p0, 1]; LZ is the
    # caller-maintained L^{-1} X_gamma'y, reused instead of re-solved

    XTY = np.asarray(XTY).ravel()
    P = XTY.shape[0]

    weight = np.zeros(P)
    trunc_weight = np.zeros(P)

    if np.any(gamma):
        if LZ is None:
            z = solve_triangular(L, XTY[gamma], lower=True, check_finite=False)
        else:
            z = np.asarray(LZ).ravel()
        h = solve_triangular(L.T, z, check_finite=False)
        c = (
            solve_triangular(
                L, np.identity(np.shape(L)[0]), lower=True, check_finite=False
            )
            ** 2
        ).sum(axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            fit = np.where(c > 0, (h**2) / c, 0.0)

        weight[gamma] = (p0**kappa) * (S / (S + fit)) ** (N / 2)
        trunc_weight[gamma] = np.maximum(np.minimum(weight[gamma], 1), 1 / p0)

    return weight, trunc_weight


def _add_stage(L, gamma, S, g, kappa, N, XTY, XTX, p0, LZ=None):

    # add-stage weights for every candidate j not in gamma: raw
    # B(gamma, gamma U {j}) clipped to [1/p0, p0], plus the residual
    # projections (u, d, ok) it was built from (reused by rb_beta); one
    # batched triangular solve yields u_j, d_j for all candidates at once

    XTY = np.asarray(XTY).ravel()
    dg = np.diag(XTX)

    if np.shape(L)[0] > 0:
        if LZ is None:
            z = solve_triangular(L, XTY[gamma], lower=True, check_finite=False)
        else:
            z = np.asarray(LZ).ravel()
        W = solve_triangular(L, XTX[gamma, :], lower=True, check_finite=False)
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

    trunc_weight = np.where(ok, np.maximum(np.minimum(weight, p0), 1 / p0), 0.0)
    trunc_weight[gamma] = 0

    return weight, trunc_weight, u, d, ok


def weight(L, gamma, S, g, kappa, N, XTY, YTY, XTX, ADS, ind=None, p0=None, LZ=None):

    # proposal weights and the normalized proposal distribution for move type
    # ADS (0 add, 1 delete, 2 swap-add stage excluding candidate ind); LZ is
    # the optional caller-maintained L^{-1} X_gamma'y

    P, _ = np.shape(XTY)

    if p0 is None:
        p0 = P

    if ADS == 1:
        w, trunc = _del_stage(L, gamma, S, g, kappa, N, XTY, p0, LZ=LZ)
    else:
        w, trunc, _, _, _ = _add_stage(
            L, gamma, S, g, kappa, N, XTY, XTX, p0, LZ=LZ
        )

        if ADS == 2 and ind is not None:
            w[ind] = 0
            trunc[ind] = 0

    return [w, _normalize(trunc)]


def rb_beta(
    L, gamma, S, g, kappa, N, XTY, YTY, XTX, s0, p0=None, LZ=None, del_w=None, add_uv=None
):

    # Rao-Blackwellized conditional posterior mean E[beta_j | gamma_{-j}, y]
    # for every j (paper sec. 5.2): the two models differing only in j are
    # averaged with their posterior weights. del_w, add_uv=(u,d,ok) and LZ
    # optionally pass the delete-stage weights, add-stage projections and
    # L^{-1} X_gamma'y already computed for this state during the proposal,
    # so the estimator costs at most one additional stage computation. At
    # capacity |gamma| = s0 the model gamma union {j} lies outside M(s0),
    # so out-of-model coordinates are 0

    P = gamma.shape[0]
    XTY_r = np.asarray(XTY).ravel()

    if p0 is None:
        p0 = P

    q2 = g / (1 + g)
    out = np.zeros(P)

    # in-model coordinates: E[beta_j | gamma_{-j}, y] = mu_j / (1 + B(gamma, gamma\{j}))
    pos = np.flatnonzero(gamma)

    if pos.size:
        if del_w is None:
            del_w, _ = _del_stage(L, gamma, S, g, kappa, N, XTY_r, p0)
        if LZ is None:
            LZ = solve_triangular(L, XTY_r[pos], lower=True, check_finite=False)
        mu = q2 * solve_triangular(L.T, np.asarray(LZ).ravel(), check_finite=False)
        out[pos] = mu / (1.0 + del_w[pos])

    # out-of-model coordinates: E[beta_j | gamma_{-j}, y] = B*mu / (1 + B),
    # with mu_j = q2*u_j/d_j from the same residual projections as the
    # add-stage weights and B the raw add weight
    sel = ~gamma

    if gamma.sum() < s0:
        dg = np.diag(XTX)

        if add_uv is not None:
            u, d, ok = add_uv
            ok = ok & sel
        else:
            if pos.size:
                z = solve_triangular(
                    L, XTY_r[pos], lower=True, check_finite=False
                )
                W = solve_triangular(L, XTX[pos, :], lower=True, check_finite=False)
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
            LZc = solve_triangular(
                L, XTY[gamma], lower=True, check_finite=False
            )
            return gamma.copy(), L, LZc, S

    LZ1 = solve_triangular(L1, XTY[gamma1], lower=True, check_finite=False)
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
            LZc = solve_triangular(
                L, XTY[gamma], lower=True, check_finite=False
            )
            return gamma.copy(), L, LZc, S

    LZ1 = solve_triangular(L1, XTY[gamma1], lower=True, check_finite=False)
    S1 = (1 + 1 / g) * YTY - (LZ1**2).sum()

    return gamma1, L1, LZ1, S1


def propose(
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
    LZ=None,
    rng=None,
):

    # one informed LIT-MH proposal of the requested move type: draws the
    # candidate(s), builds the proposal state, and returns it together with
    # the MH acceptance ratio; returns None if the move has no candidate.
    # The sixth element exposes the stage results already computed for the
    # pre-move state and the proposal state, for reuse by rb_beta

    if rng is None:
        rng = np.random.default_rng()

    if ADS == 0:
        w_add, trunc_add, u, d, ok = _add_stage(
            L, gamma, S, g, kappa, N, XTY, XTX, P0, LZ=LZ
        )
        q_add = _normalize(trunc_add)

        if q_add.sum() == 0:
            return None

        j = int(rng.choice(P, p=q_add))
        gamma1, L1, LZ1, S1 = make_add(L, S, gamma, j, XTX, XTY, YTY, g)

        w_del, trunc_del = _del_stage(L1, gamma1, S1, g, kappa, N, XTY, P0, LZ=LZ1)
        q_del = _normalize(trunc_del)
        h_d_rev = move_prob(s + 1, s0, PA, PD, PS)[1]

        ratio = (h_d_rev * q_del[j]) / (h_a * q_add[j]) * w_add[j]

        return (
            gamma1,
            L1,
            LZ1,
            S1,
            ratio,
            {"pre": {"add_uv": (u, d, ok)}, "post": {"del_w": w_del}},
        )

    elif ADS == 1:
        w_del, trunc_del = _del_stage(L, gamma, S, g, kappa, N, XTY, P0, LZ=LZ)
        q_del = _normalize(trunc_del)
        k = int(rng.choice(P, p=q_del))
        gamma1, L1, LZ1, S1 = make_delete(L, S, gamma, k, XTX, XTY, YTY, g)

        w_add, trunc_add, u, d, ok = _add_stage(
            L1, gamma1, S1, g, kappa, N, XTY, XTX, P0, LZ=LZ1
        )
        q_add = _normalize(trunc_add)
        h_a_rev = move_prob(s - 1, s0, PA, PD, PS)[0]

        ratio = (h_a_rev * q_add[k]) / (h_d * q_del[k]) * w_del[k]

        return (
            gamma1,
            L1,
            LZ1,
            S1,
            ratio,
            {"pre": {"del_w": w_del}, "post": {"add_uv": (u, d, ok)}},
        )

    else:
        w_del, trunc_del = _del_stage(L, gamma, S, g, kappa, N, XTY, P0, LZ=LZ)
        q_del = _normalize(trunc_del)
        k = int(rng.choice(P, p=q_del))
        gamma0, L0, LZ0, S0 = make_delete(L, S, gamma, k, XTX, XTY, YTY, g)

        # the two exclusions of a swap (k forward, j reverse) share the same
        # add-stage computation on gamma0
        w_add, trunc_add, _, _, _ = _add_stage(
            L0, gamma0, S0, g, kappa, N, XTY, XTX, P0, LZ=LZ0
        )
        q_add = _exclude_normalize(trunc_add, k)

        if q_add.sum() == 0:
            return None

        j = int(rng.choice(P, p=q_add))
        gamma1, L1, LZ1, S1 = make_add(L0, S0, gamma0, j, XTX, XTY, YTY, g)

        w_del1, trunc_del1 = _del_stage(L1, gamma1, S1, g, kappa, N, XTY, P0, LZ=LZ1)
        q_del1 = _normalize(trunc_del1)
        q_add1 = _exclude_normalize(trunc_add, j)

        # move-type probabilities cancel between forward and reverse swap
        ratio = (w_add[j] * w_del[k]) * (
            (q_add1[k] * q_del1[j]) / (q_add[j] * q_del[k])
        )

        return (
            gamma1,
            L1,
            LZ1,
            S1,
            ratio,
            {"pre": {"del_w": w_del}, "post": {"del_w": w_del1}},
        )
