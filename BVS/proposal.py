#proposal layer of the LIT-MH sampler: the informed thresholded proposal
#distribution (weight, paper eq. (9)-(13)), move-type probabilities (move_prob,
#eq. (12)), proposal state construction by full Cholesky refactorization
#(chol_full, make_delete, make_add), and the per-move proposal with its MH
#acceptance ratio (propose). The weight computation is split into per-move-type
#stages (_del_stage, _add_stage) whose results are pure functions of the model
#state: a bounded cache retains the current model's stages across rejections.
#Swap exclusions share one add stage, and rb_beta reuses these stages and
#caches its deterministic conditional mean (never posterior random draws).
import numpy as np
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular
from scipy.special import expit


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


def _excluded_probability(trunc, total, excluded, candidate):

    # Only this scalar is needed for the reverse half of a swap. Re-sum the
    # remaining entries if subtraction would cancel a dominant weight.
    if candidate == excluded or trunc[candidate] == 0:
        return 0.0
    remaining = total - trunc[excluded]
    if remaining <= 1e-8 * total:
        remaining = trunc[:excluded].sum() + trunc[excluded + 1 :].sum()
    return trunc[candidate] / remaining if remaining > 0 else 0.0


def _bind_cache(cache, L, gamma, S, g, kappa, N, XTY, YTY, XTX, p0, s0):

    # Cache inputs are immutable for the lifetime of a model in DF_MCMC.
    # Object identity avoids hashing/comparing the full Gram matrix on every
    # iteration. External callers modifying these arrays in place must clear
    # their cache first. Retaining these references also prevents id reuse.
    if cache is None:
        cache = {}
    arrays = (L, gamma, XTY, XTX)
    parameters = (S, g, kappa, N, YTY, p0, s0)
    previous = cache.get("arrays")
    if (
        previous is None
        or any(old is not new for old, new in zip(previous, arrays))
        or cache.get("parameters") != parameters
    ):
        cache.clear()
        cache.update(arrays=arrays, parameters=parameters, pos=np.flatnonzero(gamma))
    return cache


def _cached_lz(cache, L, XTY, LZ):

    if "LZ" not in cache:
        cache["LZ"] = (
            np.asarray(LZ).ravel()
            if LZ is not None
            else solve_triangular(
                L, np.asarray(XTY).ravel()[cache["pos"]], lower=True, check_finite=False
            )
        )
    return cache["LZ"]


def _del_compact(L, pos, S, kappa, N, XTY, p0, LZ=None, coefficients=None):

    # The hot path stores only the s actual deletion candidates. The public
    # dense wrapper below expands these values for backward compatibility.
    if pos.size == 0:
        empty = np.empty(0)
        return empty, empty.copy(), empty.copy()
    z = (
        solve_triangular(L, np.asarray(XTY).ravel()[pos], lower=True, check_finite=False)
        if LZ is None else np.asarray(LZ).ravel()
    )
    h = (
        solve_triangular(L.T, z, check_finite=False)
        if coefficients is None else coefficients
    )
    c = (solve_triangular(L, np.identity(pos.size), lower=True, check_finite=False) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fit = np.where(c > 0, (h**2) / c, 0.0)
    raw = (p0**kappa) * (S / (S + fit)) ** (N / 2)
    return raw, np.maximum(np.minimum(raw, 1), 1 / p0), h


def _cached_delete(cache, L, S, kappa, N, XTY, p0, LZ):

    if "delete" not in cache:
        z = _cached_lz(cache, L, XTY, LZ)
        raw, trunc, h = _del_compact(
            L, cache["pos"], S, kappa, N, XTY, p0, z, cache.get("coefficients")
        )
        cache["coefficients"] = h
        cache["delete"] = (raw, trunc, _normalize(trunc))
    return cache["delete"]


def _delete_probability(cache, variable):

    pos = cache["pos"]
    index = np.searchsorted(pos, variable)
    return cache["delete"][2][index] if index < pos.size and pos[index] == variable else 0.0


def _del_stage(L, gamma, S, g, kappa, N, XTY, p0, LZ=None):

    # delete-stage weights: raw B(gamma, gamma\{k}) for k in gamma (0
    # elsewhere) and the truncated copy clipped to [1/p0, 1]; LZ is the
    # caller-maintained L^{-1} X_gamma'y, reused instead of re-solved

    P = np.asarray(XTY).size
    pos = np.flatnonzero(gamma)
    raw, trunc, _ = _del_compact(L, pos, S, kappa, N, XTY, p0, LZ)
    weight = np.zeros(P)
    trunc_weight = np.zeros(P)
    weight[pos], trunc_weight[pos] = raw, trunc
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

    weight, trunc_weight = _add_weights(gamma, S, kappa, N, p0, u, d, ok)
    return weight, trunc_weight, u, d, ok


def _add_weights(gamma, S, kappa, N, p0, u, d, ok):

    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(ok, (u**2) / (d * S), 0.0)
    r = np.minimum(r, 1 - 1e-12)

    # Very large raw Bayes factors may overflow to infinity; clipping below
    # still gives finite proposal weights, and swap ratios use log residuals.
    with np.errstate(over="ignore"):
        weight = np.where(ok, (1 - r) ** (-N / 2) / (p0**kappa), 0.0)
    weight[gamma] = 0

    trunc_weight = np.where(ok, np.maximum(np.minimum(weight, p0), 1 / p0), 0.0)
    trunc_weight[gamma] = 0

    return weight, trunc_weight


def _cached_add(cache, L, gamma, S, g, kappa, N, XTY, XTX, p0, LZ):

    if "add" not in cache:
        if "add_uv" in cache:
            u, d, ok = cache["add_uv"]
            raw, trunc = _add_weights(gamma, S, kappa, N, p0, u, d, ok)
        else:
            z = _cached_lz(cache, L, XTY, LZ)
            raw, trunc, u, d, ok = _add_stage(
                L, gamma, S, g, kappa, N, XTY, XTX, p0, LZ=z
            )
            cache["add_uv"] = (u, d, ok)
        total = trunc.sum()
        probability = trunc / total if total > 0 else trunc.copy()
        cache["add"] = (raw, trunc, probability, total)
    return cache["add"]


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
    L, gamma, S, g, kappa, N, XTY, YTY, XTX, s0, p0=None, LZ=None, del_w=None, add_uv=None,
    cache=None,
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

    if p0 is None:
        p0 = P

    cache = _bind_cache(cache, L, gamma, S, g, kappa, N, XTY, YTY, XTX, p0, s0)
    if "rb" in cache:
        # Preserve the public result's mutability without exposing the stored
        # value to accidental edits by callers.
        return cache["rb"].copy()

    q2 = g / (1 + g)
    out = np.zeros(P)

    # in-model coordinates: E[beta_j | gamma_{-j}, y] = mu_j / (1 + B(gamma, gamma\{j}))
    pos = cache["pos"]

    if pos.size:
        if del_w is None:
            raw_delete, _, _ = _cached_delete(cache, L, S, kappa, N, XTY, p0, LZ)
        else:
            raw_delete = np.asarray(del_w)[pos]
        if "coefficients" not in cache:
            z = _cached_lz(cache, L, XTY, LZ)
            cache["coefficients"] = solve_triangular(L.T, z, check_finite=False)
        out[pos] = q2 * cache["coefficients"] / (1.0 + raw_delete)

    # Out-of-model coordinates average mu_j = q2*u_j/d_j with the conditional
    # inclusion probability expit(log B), avoiding inf/inf for large B.
    # The residual projections are shared with the add-stage weights.
    sel = ~gamma

    if pos.size < s0:
        if add_uv is not None and "add_uv" not in cache:
            cache["add_uv"] = add_uv
        if "add_uv" not in cache:
            _cached_add(cache, L, gamma, S, g, kappa, N, XTY, XTX, p0, LZ)
        u, d, ok = cache["add_uv"]
        ok = ok & sel

        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(ok, (u**2) / (d * S), 0.0)
            mu_out = np.where(ok, q2 * u / d, 0.0)
        r = np.minimum(r, 1 - 1e-12)
        log_B = -(N / 2) * np.log1p(-r) - kappa * np.log(p0)
        out[ok] = expit(log_B[ok]) * mu_out[ok]

    cache["rb"] = out
    return out.copy()


def chol_full(XTX, gamma):

    # full Cholesky of a model's Gram submatrix; at realistic model sizes one
    # LAPACK call is faster and more stable than Python-level rank-1 updates

    pos = np.flatnonzero(gamma)

    if pos.size == 0:
        return np.zeros((0, 0))

    return cholesky(XTX[np.ix_(pos, pos)])


def make_delete(L, S, gamma, k, XTX, XTY, YTY, g):

    # proposal state with covariate k removed. Deleting the factor's last
    # column needs no factorization (leading principal submatrix); any other
    # deletion refactorizes. If refactorization fails, collapses to the
    # current state

    gamma1 = gamma.copy()
    index = gamma[0 : k + 1].sum() - 1
    gamma1[k] = 0

    if index == L.shape[0] - 1:
        L1 = L[:index, :index].copy()
    else:
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
    cache=None,
):

    # one informed LIT-MH proposal of the requested move type: draws the
    # candidate(s), builds the proposal state, and returns it together with
    # the MH acceptance ratio; returns None if the move has no candidate.
    # The sixth element exposes the stage results already computed for the
    # pre-move state and the proposal state, for reuse by rb_beta

    if rng is None:
        rng = np.random.default_rng()
    cache = _bind_cache(cache, L, gamma, S, g, kappa, N, XTY, YTY, XTX, P0, s0)

    if ADS == 0:
        w_add, _, q_add, total = _cached_add(
            cache, L, gamma, S, g, kappa, N, XTY, XTX, P0, LZ
        )

        if total == 0:
            return None

        j = int(rng.choice(P, p=q_add))
        gamma1, L1, LZ1, S1 = make_add(L, S, gamma, j, XTX, XTY, YTY, g)

        proposal_cache = _bind_cache({}, L1, gamma1, S1, g, kappa, N, XTY, YTY, XTX, P0, s0)
        _cached_delete(proposal_cache, L1, S1, kappa, N, XTY, P0, LZ1)
        q_reverse = _delete_probability(proposal_cache, j)
        h_d_rev = move_prob(s + 1, s0, PA, PD, PS)[1]

        ratio = (h_d_rev * q_reverse) / (h_a * q_add[j]) * w_add[j]

        return (
            gamma1,
            L1,
            LZ1,
            S1,
            ratio,
            {"pre": {"cache": cache}, "post": {"cache": proposal_cache}},
        )

    elif ADS == 1:
        w_del, _, q_del = _cached_delete(cache, L, S, kappa, N, XTY, P0, LZ)
        selected = int(rng.choice(cache["pos"].size, p=q_del))
        k = int(cache["pos"][selected])
        gamma1, L1, LZ1, S1 = make_delete(L, S, gamma, k, XTX, XTY, YTY, g)

        proposal_cache = _bind_cache({}, L1, gamma1, S1, g, kappa, N, XTY, YTY, XTX, P0, s0)
        _, _, q_add, _ = _cached_add(
            proposal_cache, L1, gamma1, S1, g, kappa, N, XTY, XTX, P0, LZ1
        )
        h_a_rev = move_prob(s - 1, s0, PA, PD, PS)[0]

        ratio = (h_a_rev * q_add[k]) / (h_d * q_del[selected]) * w_del[selected]

        return (
            gamma1,
            L1,
            LZ1,
            S1,
            ratio,
            {"pre": {"cache": cache}, "post": {"cache": proposal_cache}},
        )

    else:
        _, _, q_del = _cached_delete(cache, L, S, kappa, N, XTY, P0, LZ)
        selected = int(rng.choice(cache["pos"].size, p=q_del))
        k = int(cache["pos"][selected])
        gamma0, L0, LZ0, S0 = make_delete(L, S, gamma, k, XTX, XTY, YTY, g)

        # the two exclusions of a swap (k forward, j reverse) share the same
        # add-stage computation on gamma0
        _, trunc_add, _, _, _ = _add_stage(
            L0, gamma0, S0, g, kappa, N, XTY, XTX, P0, LZ=LZ0
        )
        total_add = trunc_add.sum()
        q_add = _exclude_normalize(trunc_add, k)

        if q_add.sum() == 0:
            return None

        j = int(rng.choice(P, p=q_add))
        gamma1, L1, LZ1, S1 = make_add(L0, S0, gamma0, j, XTX, XTY, YTY, g)

        proposal_cache = _bind_cache({}, L1, gamma1, S1, g, kappa, N, XTY, YTY, XTX, P0, s0)
        _cached_delete(proposal_cache, L1, S1, kappa, N, XTY, P0, LZ1)
        q_del_reverse = _delete_probability(proposal_cache, j)
        q_add_reverse = _excluded_probability(trunc_add, total_add, j, k)

        # Equal model sizes cancel the sparsity prior and move probabilities.
        # Compute the posterior ratio directly from the two endpoint residuals:
        # multiplying the intermediate add/delete factors can give inf * 0
        # even when the swap ratio is finite. A missing reverse move rejects.
        if q_add_reverse == 0 or q_del_reverse == 0:
            ratio = 0.0
        else:
            log_ratio = (
                (N / 2) * (np.log(S) - np.log(S1))
                + np.log(q_add_reverse)
                + np.log(q_del_reverse)
                - np.log(q_add[j])
                - np.log(q_del[selected])
            )
            with np.errstate(over="ignore", under="ignore"):
                ratio = np.exp(log_ratio)

        return (
            gamma1,
            L1,
            LZ1,
            S1,
            ratio,
            {"pre": {"cache": cache}, "post": {"cache": proposal_cache}},
        )
