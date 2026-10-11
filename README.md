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
| [`BVS/ESS.py`](BVS/ESS.py) | Effective sample size (ArviZ): `ess`, `ess_gamma`, `ess_beta`, `ess_sigma2`, `ess_from_df_mcmc` |
| `tests/test_*.py` | Regression tests for posterior moments, MH and RB numerical stability, sampler boundaries, ESS, and notebook helpers (run `python -m pytest tests/`, or run each test script directly) |
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
| `s0` | — | Non-negative integer maximum model size; zero restricts the chain to the null model |
| `s_initial` | — | Integer size of the random initial model, between zero and `min(s0, P)` in working coordinates (redrawn if numerically collinear; falls back to the null model if all draws fail) |
| `M`, `burn_in` | 10000 | Positive integer kept-sample count and non-negative integer burn-in count; with `burn_in=0`, the initial state is the first complete retained sample |
| `Sample_beta` | False | Draw the conditional posteriors $\sigma^2\sim\mathrm{InvGamma}$ and $\beta\mid\sigma^2$ (g-prior normal) |
| `online` | False | Online-statistics mode, no trace stored (see below) |
| `screen` | None | Number of columns kept by marginal screening |
| `seed` | None | Random seed, consumed by a per-call `numpy.random.default_rng` (no global RNG state is read or written) |
| `PA, PD, PS` | 0.4/0.4/0.2 | Add/delete/swap proposal probabilities (must sum to 1; at capacity $s_0$ the sampler switches to delete/swap with 1/2 each, and the null model forces adds) |
| `w` | 0 | Prior degrees of freedom for $\sigma^2$ |
| `RB` | False | Track the Rao-Blackwellized estimate of $\mathbb{E}[\beta\mid y]$ (paper §5.2); caches the current model's deterministic estimate across rejected moves |
| `gamma_init` | None | Iterable of integer starting-column indices in working (possibly screened) coordinates, with at most `s0` distinct columns; overrides `s_initial`. Convert a Boolean mask with `np.flatnonzero(mask)` |

**Return values** (three modes):

| Mode | Returns |
|---|---|
| Default (trace, no β) | `gamma_sample`, a (P, M) boolean array |
| `Sample_beta=True` | `[beta_sample, gamma_sample, sigma2_sample]` with shapes (P, M), (P, M), (M,) |
| `online=True` | dict: `n_iter`, `pip`, `model_counts` (model → count, includes the MAP model), `proposal`/`accept` (per-move-type proposal and acceptance counts); with `Sample_beta=True` also `beta_mean`, `beta_sd`, `sigma2_mean` |

With `RB=True`, the Rao-Blackwellized estimate $\hat\beta_{\mathrm{RB}}$ (paper §5.2) is added: as `stats['beta_rb']` in online mode, or appended as a final (P, M) array to the trace outputs. Each $\hat\beta_{\mathrm{RB},j} = \mathbb{E}[\beta_j \mid \gamma_{-j}, y]$ averages the two models differing only in coordinate $j$ with their posterior weights, and every iteration contributes a smoothed value for **all** $j$ — so the averaged estimate converges much faster than counting inclusions (verified: it recovers the exactly-enumerated $\mathbb{E}[\beta\mid y]$ to 2e-4 on enumerable problems, and reaches the optimal MSE within ~25 iterations in the paper's §5.1 SNR=3 setting, where plain conditional means need the chain to settle and RW-MH needs ~20,000 iterations). At capacity $\lvert\gamma\rvert=s_0$ the out-of-model entries are 0, since $\gamma\cup\{j\}$ lies outside $\mathcal{M}(s_0)$.

Both output modes retain exactly `M` samples. With `burn_in=0`, this includes the supplied or randomly initialized model and its β/σ²/RB values; with `M=1, burn_in=0`, no model transition occurs. Proposal and acceptance counters include attempted transitions during burn-in but do not count the initial state. Drawing β/σ² at the initial state changes the random stream relative to versions that left that sample uninitialized.

Trace arrays allocate only the `M` retained columns and own their storage; burn-in does not leave a hidden backing buffer attached to the result. With both `Sample_beta=True` and `RB=True`, these arrays occupy `(17 * P + 8) * M` bytes (using the working column count after screening). This excludes the dense Gram matrix and temporary proposal arrays.

### Other modules

- `weight(L, gamma, S, ..., ADS, ind=None, p0=None, LZ=None)`: proposal weights and the normalized proposal distribution; `ADS` 0/1/2 selects the add/delete/swap-add stage, `ind` is the variable excluded by a swap, `p0` is the original p used in the prior (differs from the working column count under screening), and `LZ` optionally passes the caller-maintained $L^{-1}X_\gamma'y$ to skip re-solving.
- `propose(L, gamma, S, ADS, ..., LZ=None, rng=None, cache=None)`: one proposal, returning `(gamma', L', LZ', S', ratio, reuse)`; returns `None` when the move has no candidate. The driver keeps `reuse['pre']['cache']` on rejection and replaces it with `reuse['post']['cache']` on acceptance. These dictionaries can also be passed to `rb_beta` via `**reuse['pre']` or `**reuse['post']`. A supplied cache is bound to immutable model/data arrays and scalar parameters; external callers that modify those arrays in place must clear the cache before reuse.
- `make_add` / `make_delete`: single-covariate state construction by full refactorization of the target model's Gram matrix; deleting the factor's last column needs no factorization.
- `ess(chains)`: accepts (M,) or (n_chains, M); `ess_from_df_mcmc(result)` accepts all trace outputs, including both `RB=True` structures. It reports `gamma` and, when present, `beta`, `sigma2`, and `beta_rb`; RB uses `method_beta`. The online dict raises because no trace is stored.

## Mapping to the paper

| Code | Paper |
|---|---|
| `weight()` | Eq. (4)(5) neighbourhood posterior ratios; eqs. (9)–(10)(13) thresholded informed proposal (adds clipped to $[1/p,\,p]$, deletes to $[1/p,\,1]$) |
| `move_prob()` | The $h_\star(s)$ of eq. (12) (forced add at the null, $h_d=h_s=1/2$ at capacity) |
| `propose()` | The MH acceptance ratio of eq. (14) (including the h factors; they cancel for swaps) |
| `screen=` | Marginal-regression screening of eq. (11) (Fan & Lv 2008); excluded candidates get weight 0 rather than the floor $p^{\ell_a}$, and the chain on the restricted space remains exactly invariant |
| `RB=` / `rb_beta()` | The Rao-Blackwellized estimator of $\mathbb{E}[\beta\mid y]$ from §5.2, built from the same quantities the proposal weights are computed from |

## Numerical robustness

- All weight computations are vectorized: the add/swap stages obtain every candidate's residual projections from one batched triangular solve, and the delete stage uses the diagonal of $(X_\gamma'X_\gamma)^{-1}$ — roughly two orders of magnitude faster than per-candidate updates.
- A bounded cache lazily retains the current model's add/delete stages and deterministic RB estimate across rejected moves; accepted moves replace that cache. Every retained iteration still contributes to summaries, and conditional β/σ² draws are freshly sampled each time. No cache of all visited models is accumulated.
- The proposal hot path represents deletion weights with only the active model's `s` entries. A swap shares its intermediate add stage and computes the reverse add probability as a scalar. The public `weight()` API continues to return dense vectors. Compact normalization can change floating-point rounding; regression tests check exact model trajectories and numerical agreement of posterior outputs against the saved pre-optimization baseline on bounded fixtures.
- All randomness flows from a per-call `numpy.random.default_rng(seed)` instance, passed down to the move-type and candidate draws and (via scipy's `random_state`) to the conditional σ²/β draws — a given seed is exactly reproducible regardless of other `numpy.random` usage in the process.
- Numerically collinear candidates (residual projection $d_j\le 10^{-12}\,\mathrm{diag}$) get weight zero and never enter a singular model; truncation is applied after the zeroing.
- Proposal states refactorize the target model's Gram submatrix with one LAPACK Cholesky call — at realistic model sizes this is faster and more stable than Python-level rank-1 updates (see [cholupdate_optimization_notes.md](cholupdate_optimization_notes.md); the retired kernels remain in the baseline archive). Deleting the factor's last column only truncates the factor. If refactorization fails, the move is recorded as a stay. A collinear initial model is redrawn automatically.
- When $s_0$ exceeds the rank of the design, proposal stages with no candidates become self-loops that still contribute complete posterior samples.
- Swap acceptance ratios combine endpoint residuals and proposal probabilities in the log domain; RB inclusion probabilities use a stable logistic transform of the log posterior ratio, avoiding `inf * 0` and `inf / inf` under strong signals.

## Verified properties

- **Exact invariance**: on small problems where the full posterior can be enumerated, the empirical distribution matches the closed-form posterior with total-variation distance ≤ 0.005 across regimes (spread over model sizes, noise-only data concentrated at the null, mass at capacity, swap-only transitions); custom PA/PD/PS mixes pass as well.
- **Proposal ratios**: the MH ratios returned by `propose()` for all three move types agree with independent closed-form computation to ~1e-14 relative error.
- **Rao-Blackwellized estimator**: `rb_beta` matches the exact two-model formula to 1.8e-15 (including states at capacity $s_0$), and the sampler-averaged `beta_rb` recovers the exactly-enumerated $\mathbb{E}[\beta\mid y]$ to 2e-4.
- **Degenerate inputs**: exactly duplicated columns and $s_0$ beyond the design rank are handled by candidate filtering and self-loops. Initial sizes outside the model space or available columns, and invalid explicit indices, raise `ValueError` before sampling.
- **Consistency**: online statistics match trace summaries in regression tests. Optimization contract tests compare fixed-seed model trajectories and numerical posterior outputs with the corrected pre-optimization baseline, including zero burn-in, unavailable moves, screening and the null-only model space. These checks do not guarantee identical trajectories for all seeds near floating-point acceptance boundaries. The migration from the legacy global RNG to `numpy.random.default_rng` deliberately changed the random stream.

## Optimization benchmark

The first three optimizations retain only kept trace columns, cache deterministic current-model stages across rejections, and use compact deletion candidates plus a scalar reverse swap probability. The baseline archive in `benchmarks/baselines/before_optimizations.zip` preserves the corrected working code immediately before these changes, including the earlier correctness fixes; it is not the older Git HEAD.

Measured on this Windows machine on 2026-10-11 with Python 3.14.7, NumPy 2.5.3 and SciPy 1.18.1, using one BLAS thread. Timings are medians of five calls after one warmup; memory comes from separate fresh processes. All calls include Gram construction. Data generation, imports, output hashing and memory instrumentation are excluded from timing. Online cases use `N=250, P=800, M=1500, burn_in=500`; the trace case uses `N=250, P=2000, M=2000, burn_in=2000` with both beta and RB output. Exact configurations and raw repeats are recorded in the results.

| Workload | Before (s) | After (s) | Speedup | Peak tracked allocations, before → after (MiB) |
|---|---:|---:|---:|---:|
| Strong signals, online RB | 0.875 | 0.663 | 1.32× | 5.465 → 5.510 |
| Weak signals, plain online | 0.934 | 0.789 | 1.18× | 6.105 → 6.107 |
| Correlated competing predictors, online RB | 0.887 | 0.661 | 1.34× | 5.472 → 5.511 |
| Stored beta/gamma/sigma2/RB traces | 2.465 | 2.182 | 1.13× | 160.921 → 96.149 |

The trace outputs' actual backing allocations fell from **129.73 to 64.87 MiB (50%)**. Their visible shapes already contained only `M` columns before the change, but the slices kept larger buffers alive. Peak tracked allocations fell **40.3%**, while Windows peak working set fell from **188.36 to 176.32 MiB (6.4%)**. These are different metrics: `tracemalloc` measures tracked allocations after input generation, whereas the OS working set includes interpreter/import/input memory and depends on which pages are resident. Online peak tracked allocations increased by less than 1% due to the cache; its benefit here is speed.

All four workloads produced identical result hashes across variants and repeats, and all 32 regression tests passed, including 80 baseline/current mode combinations. These are bounded checks on this machine and these seeds, not a universal speed or bitwise-equivalence guarantee. Variant order alternates across cases rather than within each case's repeats; small differences should not be overinterpreted.

- [Raw measurements and reproducibility metadata](benchmarks/results/optimizations-2026-10-11/comparison.json)
- [Validated run manifest with input/output hashes](benchmarks/results/optimizations-2026-10-11/manifest.json)
- [Regression test output](benchmarks/results/validation.txt)

To repeat the benchmark from the repository root (requires `threadpoolctl`; `psutil` supplies optional OS memory metrics):

```shell
python benchmarks/compare_optimizations.py --output benchmarks/results/optimization_comparison.json
```

The harness verifies the archived baseline's checksum and loads each variant in an isolated process. Its JSON includes source hashes, seeds, library and BLAS configuration, individual timings, allocation totals and measurement limitations. The recorded run additionally uses `benchmarks/optimization_contract.json` and the computation-audit bounded runner for provenance.

## Caveats

- **Screening drops weak signals**: marginal screening requires the signal to be large relative to $\sqrt{\log p/n}$; with small samples weak signals may not make the screened set (an inherent limitation of sure independence screening, not a bug). A `screen` size on the order of $n$ or $20 s_0$ is a reasonable default.
- Under **screening with trace mode**, the returned `gamma_sample` is in screened coordinates (shape (screen, M)); the online mode's `pip` and other summaries are already mapped back to the original coordinates.
- `kappa1=1, kappa2=1` (i.e. $g=p^2$, $\kappa=2$) is the common default; the theory requires $\kappa=O(s_0)$ and $s_0\log p=O(n)$ (Remark 2 of the paper).
- Trace mode uses $O(P\cdot M)$ memory independent of the burn-in length. Online mode avoids these traces but still stores visited-model counts. Both modes retain the dense $O(P^2)$ Gram matrix and $O(P+s^2)$ current-state cache; temporary add-stage solves use $O(Ps)$ space.
- In `ESS.py`, constant chains (covariates that never switch) return NaN — their ESS is undefined.

## References

- Zhou, Q., Yang, J., Vats, D., Roberts, G. O., & Rosenthal, J. S. (2022). Dimension-free mixing for high-dimensional Bayesian variable selection. *Journal of the Royal Statistical Society Series B*, 84(3), 742–768. doi:10.1111/rssb.12546
- Fan, J., & Lv, J. (2008). Sure independence screening for ultrahigh dimensional feature space. *JRSS-B*.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: an improved R̂ for assessing convergence of MCMC. *Bayesian Analysis*.

## License

Released under the [MIT License](LICENSE).
