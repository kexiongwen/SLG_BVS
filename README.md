# SLG_BVS

A Python implementation of the **LIT-MH** sampler for high-dimensional Bayesian variable selection, from Zhou, Yang, Vats, Roberts & Rosenthal (2022), *Dimension-free mixing for high-dimensional Bayesian variable selection*, JRSS-B.

The model is a sparse linear regression $y = X_\gamma\beta_\gamma + e$ with a g-prior and a sparsity prior on the model (paper eq. 3):

$$\beta_\gamma \mid \gamma \sim \mathrm{MN}(0,\; g\phi^{-1}(X_\gamma'X_\gamma)^{-1}),\qquad \pi(\gamma) \propto p^{-\kappa_2|\gamma|},\qquad 1+g = p^{2\kappa_1}$$

The sampler performs add/delete/swap Metropolis–Hastings moves on the model space $\mathcal{M}(s_0)=\{\gamma:|\gamma|\le s_0\}$ with a **locally informed and thresholded** proposal distribution: neighbouring posterior ratios $B(\gamma,\gamma')$ are clipped to $[p^{\ell_\star},p^{L_\star}]$ and used as proposal weights (eqs. 9–13). The two-sided truncation is the paper's central design choice — it prevents informed proposals from getting stuck through exceedingly small acceptance probabilities (the counterexample in §1.2).

## Module structure

| File | Contents |
|---|---|
| [`BVS/DF_MCMC.py`](BVS/DF_MCMC.py) | Sampler driver `DF_MCMC()` and conditional-posterior sampling `sample_beta()` |
| [`BVS/proposal.py`](BVS/proposal.py) | Proposal layer: `weight` (informed thresholded proposal distribution), `move_prob` (move-type probabilities), `chol_full`/`make_delete`/`make_add` (proposal state construction), `propose` (proposal + MH ratio) |
| [`BVS/cholupdate.py`](BVS/cholupdate.py) | Rank-1 Cholesky primitives: `update`/`downdate`/`chol_del`/`chol_add`/`chol_add_col`, giving $O(s^2)$ cost per move |
| [`BVS/ESS.py`](BVS/ESS.py) | Effective sample size (ArviZ): `ess`, `ess_gamma`, `ess_beta`, `ess_sigma2`, `ess_from_df_mcmc` |
| `tests/test_regressions.py` | Regression tests: RB trace-mode alignment against the online estimate, and column-scale invariance against exactly enumerated posteriors (run `python3 -m pytest tests/` or `python3 tests/test_regressions.py`) |
| `test1.ipynb` | Simulation study of paper §5.1: LIT-MH-1 finding high-posterior models, Table-1 style metrics ($\hat\gamma_{\max}$, $H_{\max}$, $t_{\max}$) |
| `test2.ipynb` | Simulation study of paper §5.3: multi-modal block-correlated design swept over $\sigma_\beta$ with forward–backward initialization and Table-2 style metrics (local modes, acceptance, ESS of $T_1$/$T_2$ per second) |

Dependencies: `numpy` (tested on 2.x) and `scipy`; `ESS.py` additionally requires `arviz >= 1.3`. No installation needed — the samplers live in the `BVS/` package; run from the repository root (as the notebooks do) and import via `BVS.*`.

## Quick start

```python
import numpy as np
from BVS.DF_MCMC import DF_MCMC

# simulate data: N=200, P=500, AR(1)-correlated design, 8 signals
rng = np.random.default_rng(1)
N, P, rho = 200, 500, 0.5
X = np.empty((N, P))
X[:, 0] = rng.standard_normal(N)
for j in range(1, P):
    X[:, j] = rho * X[:, j-1] + np.sqrt(1-rho**2) * rng.standard_normal(N)

true_idx = [0, 1, 4, 9, 12, 18, 25, 30]
beta = np.zeros((P, 1))
beta[true_idx] = [[3], [1.5], [2], [1.5], [1.5], [1], [-1], [2]]
Y = X @ beta + rng.standard_normal((N, 1))

# online mode: no (P x M) trace stored, sufficient statistics returned
stats = DF_MCMC(Y, X, kappa1=1, kappa2=1, s0=15, s_initial=8,
                M=20000, burn_in=2000, seed=3, online=True)

pip = stats['pip']                                  # posterior inclusion probabilities
top = sorted(stats['model_counts'].items(), key=lambda kv: -kv[1])[0]
print('MAP model:', list(top[0]),
      'posterior prob %.3f' % (top[1] / stats['n_iter']))
print('PIP at signals:', pip[true_idx])
```

In this example the MAP model recovers all 8 true signals (posterior probability ≈ 0.99), signal PIPs ≈ 1, and the largest noise PIP is below 0.001.

For posterior samples of β/σ² and ESS diagnostics, use trace mode:

```python
from BVS.ESS import ess_from_df_mcmc

beta_s, gamma_s, sigma2_s = DF_MCMC(Y, X, 1, 1, 15, 8, M=20000, burn_in=2000,
                                    seed=3, Sample_beta=True)
print('posterior mean of signal coefficients:', beta_s.mean(axis=1)[true_idx])  # ≈ truth
print('posterior mean of sigma2:', sigma2_s.mean())                             # ≈ 1

summ = ess_from_df_mcmc((beta_s, gamma_s, sigma2_s))
print('ESS of sigma2:', summ['sigma2'])
print('ESS of inclusion indicators:', summ['gamma'])   # constant rows are NaN
```

For very large P (tens of thousands and beyond), screen first by marginal regression (paper eq. 11, ranking columns by $|x_j'y|/\lVert x_j\rVert$):

```python
stats = DF_MCMC(Y, X, 1, 1, 15, 8, M=5000, burn_in=1000, seed=9,
                online=True, screen=500, Sample_beta=True)
# XTX is formed only for the screened columns; the returned pip / beta_mean /
# model counts are mapped back to the original P coordinates
```

Measured on P=20000, N=300 with 10 signals: `screen=500` plus online statistics runs 5000 iterations in about 2 seconds and the MAP model equals the truth.

## API reference

### `DF_MCMC(Y, X, kappa1, kappa2, s0, s_initial, ...)`

| Parameter | Default | Description |
|---|---|---|
| `Y, X` | — | Response (shape (N,) or (N,1)) and (N,P) design matrix |
| `kappa1` | — | $1+g=p^{2\kappa_1}$ |
| `kappa2` | — | Sparsity prior exponent; $\kappa=\kappa_1+\kappa_2$ |
| `s0` | — | Maximum model size |
| `s_initial` | — | Size of the random initial model (redrawn automatically if numerically collinear; falls back to the null model if all draws fail) |
| `M`, `burn_in` | 10000 | Kept samples and burn-in iterations |
| `Sample_beta` | False | Draw the conditional posteriors $\sigma^2\sim\mathrm{InvGamma}$ and $\beta\mid\sigma^2$ (g-prior normal) |
| `online` | False | Online-statistics mode, no trace stored (see below) |
| `screen` | None | Number of columns kept by marginal screening |
| `seed` | None | Random seed, consumed by a per-call `numpy.random.default_rng` (no global RNG state is read or written) |
| `PA, PD, PS` | 0.4/0.4/0.2 | Add/delete/swap proposal probabilities (must sum to 1; at capacity $s_0$ the sampler switches to delete/swap with 1/2 each, and the null model forces adds) |
| `w` | 0 | Prior degrees of freedom for $\sigma^2$ |
| `RB` | False | Track the Rao-Blackwellized estimate of $\mathbb{E}[\beta\mid y]$ (paper §5.2); reuses the weight stages computed during the proposal, so it costs at most one extra stage computation per iteration |
| `gamma_init` | None | Iterable of starting columns (working, i.e. possibly screened, coordinates), overriding the random draw of `s_initial` — used e.g. for the forward–backward stepwise initialization of paper §5.3 |

**Return values** (three modes):

| Mode | Returns |
|---|---|
| Default (trace, no β) | `gamma_sample`, a (P, M) boolean array |
| `Sample_beta=True` | `[beta_sample, gamma_sample, sigma2_sample]` with shapes (P, M), (P, M), (M,) |
| `online=True` | dict: `n_iter`, `pip`, `model_counts` (model → count, includes the MAP model), `proposal`/`accept` (per-move-type proposal and acceptance counts); with `Sample_beta=True` also `beta_mean`, `beta_sd`, `sigma2_mean` |

With `RB=True`, the Rao-Blackwellized estimate $\hat\beta_{\mathrm{RB}}$ (paper §5.2) is added: as `stats['beta_rb']` in online mode, or appended as a final (P, M) array to the trace outputs. Each $\hat\beta_{\mathrm{RB},j} = \mathbb{E}[\beta_j \mid \gamma_{-j}, y]$ averages the two models differing only in coordinate $j$ with their posterior weights, and every iteration contributes a smoothed value for **all** $j$ — so the averaged estimate converges much faster than counting inclusions (verified: it recovers the exactly-enumerated $\mathbb{E}[\beta\mid y]$ to 2e-4 on enumerable problems, and reaches the optimal MSE within ~25 iterations in the paper's §5.1 SNR=3 setting, where plain conditional means need the chain to settle and RW-MH needs ~20,000 iterations). At capacity $\lvert\gamma\rvert=s_0$ the out-of-model entries are 0, since $\gamma\cup\{j\}$ lies outside $\mathcal{M}(s_0)$.

### Other modules

- `weight(L, gamma, S, ..., ADS, ind=None, p0=None, LZ=None)`: proposal weights and the normalized proposal distribution; `ADS` 0/1/2 selects the add/delete/swap-add stage, `ind` is the variable excluded by a swap, `p0` is the original p used in the prior (differs from the working column count under screening), and `LZ` optionally passes the caller-maintained $L^{-1}X_\gamma'y$ to skip re-solving.
- `propose(L, gamma, S, ADS, ..., LZ=None)`: one proposal, returning `(gamma', L', LZ', S', ratio, reuse)` where `reuse` exposes the weight stages already computed for the pre-move and proposal states (consumed by the Rao-Blackwellized estimator); returns `None` when the move has no candidate.
- `make_add` / `make_delete`: single-covariate state construction (rank-1 update with fallback).
- `ess(chains)`: accepts (M,) or (n_chains, M); `ess_from_df_mcmc(result)` digests DF_MCMC's trace outputs directly (the online dict raises — no trace is stored).

## Mapping to the paper

| Code | Paper |
|---|---|
| `weight()` | Eq. (4)(5) neighbourhood posterior ratios; eqs. (9)–(10)(13) thresholded informed proposal (adds clipped to $[1/p,\,p]$, deletes to $[1/p,\,1]$) |
| `move_prob()` | The $h_\star(s)$ of eq. (12) (forced add at the null, $h_d=h_s=1/2$ at capacity) |
| `propose()` | The MH acceptance ratio of eq. (14) (including the h factors; they cancel for swaps) |
| `screen=` | Marginal-regression screening of eq. (11) (Fan & Lv 2008); excluded candidates get weight 0 rather than the floor $p^{\ell_a}$, and the chain on the restricted space remains exactly invariant |
| `RB=` / `rb_beta()` | The Rao-Blackwellized estimator of $\mathbb{E}[\beta\mid y]$ from §5.2, built from the same quantities the proposal weights are computed from |
| `cholupdate.py` | The $O(s^2)$ Cholesky updates of §2.3 (George & McCulloch 1997; Smith & Kohn 1996), $O(p\,s^2)$ per iteration |

## Numerical robustness

- All weight computations are vectorized: the add/swap stages obtain every candidate's residual projections from one batched triangular solve, and the delete stage uses the diagonal of $(X_\gamma'X_\gamma)^{-1}$ — roughly two orders of magnitude faster than per-candidate updates.
- Stage results are pure functions of the model state and are reused wherever possible: a swap evaluates its add-stage neighbourhood once (the forward and reverse exclusions only renormalize the same weights), and the Rao-Blackwellized estimator reuses the stage computed for the current state during the proposal. All reuse is bit-exact, verified by same-seed bit-level regression across sampler modes; scipy triangular solves run with `check_finite=False` on internally computed arrays.
- All randomness flows from a per-call `numpy.random.default_rng(seed)` instance, passed down to the move-type and candidate draws and (via scipy's `random_state`) to the conditional σ²/β draws — a given seed is exactly reproducible regardless of other `numpy.random` usage in the process.
- Numerically collinear candidates (residual projection $d_j\le 10^{-12}\,\mathrm{diag}$) get weight zero and never enter a singular model; truncation is applied after the zeroing.
- Failed rank-1 updates fall back to a full Cholesky; if that also fails the move is recorded as a stay. A collinear initial model is redrawn automatically.
- When $s_0$ exceeds the rank of the design, proposal stages with no candidates skip as self-loops.

## Verified properties

- **Exact invariance**: on small problems where the full posterior can be enumerated, the empirical distribution matches the closed-form posterior with total-variation distance ≤ 0.005 across regimes (spread over model sizes, noise-only data concentrated at the null, mass at capacity, swap-only transitions); custom PA/PD/PS mixes pass as well.
- **Proposal ratios**: the MH ratios returned by `propose()` for all three move types agree with independent closed-form computation to ~1e-14 relative error.
- **Rao-Blackwellized estimator**: `rb_beta` matches the exact two-model formula to 1.8e-15 (including states at capacity $s_0$), and the sampler-averaged `beta_rb` recovers the exactly-enumerated $\mathbb{E}[\beta\mid y]$ to 2e-4.
- **Degenerate inputs**: exactly duplicated columns, impossible initial sizes, and $s_0$ beyond the design rank all run without crashing.
- **Consistency**: online statistics exactly match trace statistics; every refactor has passed same-seed bit-level regression (the migration from the legacy global RNG to `numpy.random.default_rng` deliberately changed the random stream, so seeds do not reproduce pre-migration draws).

## Caveats

- **Screening drops weak signals**: marginal screening requires the signal to be large relative to $\sqrt{\log p/n}$; with small samples weak signals may not make the screened set (an inherent limitation of sure independence screening, not a bug). A `screen` size on the order of $n$ or $20 s_0$ is a reasonable default.
- Under **screening with trace mode**, the returned `gamma_sample` is in screened coordinates (shape (screen, M)); the online mode's `pip` and other summaries are already mapped back to the original coordinates.
- `kappa1=1, kappa2=1` (i.e. $g=p^2$, $\kappa=2$) is the common default; the theory requires $\kappa=O(s_0)$ and $s_0\log p=O(n)$ (Remark 2 of the paper).
- Trace mode uses $O(P\cdot M)$ memory; use `online=True` for large P.
- In `ESS.py`, constant chains (covariates that never switch) return NaN — their ESS is undefined.

## References

- Zhou, Q., Yang, J., Vats, D., Roberts, G. O., & Rosenthal, J. S. (2022). Dimension-free mixing for high-dimensional Bayesian variable selection. *Journal of the Royal Statistical Society Series B*, 84(3), 742–768. doi:10.1111/rssb.12546
- Fan, J., & Lv, J. (2008). Sure independence screening for ultrahigh dimensional feature space. *JRSS-B*.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: an improved R̂ for assessing convergence of MCMC. *Bayesian Analysis*.

## License

Released under the [MIT License](LICENSE).
