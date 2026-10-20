# RDM reconstruction and uncertainty-aware orbital optimization

This study tests conditional advantages of sequential OO against equally
corrected and safeguarded analytic Newton OO. The hypotheses, endpoints,
measurement budget, and fresh test seeds are preserved in the study workspace's
`protocol.json`. The original 549-trajectory study is development evidence.

## Research basis and practical choices

Fermionic marginal constraints and statistical reconstruction are established
methods. The contribution being tested here is their usefulness for this OO
implementation under a shared-count measurement model, including comparison
with a Newton method given the same correction and acceptance rule. No general
SOO advantage follows from the construction alone.

- [Rubin, Babbush and McClean (2018)](https://arxiv.org/abs/1801.03524)
  develop fermionic marginal constraints for hybrid quantum algorithms,
  including positive-semidefinite and approximately representable RDM
  reconstruction. This directly motivates joint contraction constraints and
  D/Q/G positivity. Our spatial half-scaled tensors require the repository's
  own factors and spin maps; copying spin-orbital formulas unchanged would
  impose incorrect constraints.
- [Schwemmer et al. (2015)](https://arxiv.org/abs/1310.8465) demonstrate
  systematic errors from physical maximum-likelihood and least-squares state
  reconstruction. Physical feasibility therefore cannot establish unbiased
  energy estimates or valid confidence intervals. We score bias, RMSE,
  completion and actual OO updates separately and validate proposals against
  raw held-out counts.
- [Nakanishi, Fujii and Todo (2020)](https://arxiv.org/abs/1903.12166)
  study sequential exact minimization of trigonometric parameter profiles.
  [Wierichs et al. (2022)](https://arxiv.org/abs/2107.12390) connect finite
  Fourier spectra to generalized parameter-shift and function reconstruction.
  These support analytic coordinate models, but their generic circuit
  observation setting does not make our nine same-RDM evaluations independent.
  The existing Aberth minimizer remains appropriate for a nominal degree-four
  orbital-energy profile; an uncertainty penalty generally is not degree four.
- [Pedrielli et al. (2026)](https://arxiv.org/abs/2502.02625v2) use a
  Fourier-compatible Gaussian-process model and uncertainty to estimate VQE
  gradients and allocate observations. This is relevant to Bayesian updates,
  but the implementation here uses sufficient-statistic Beta draws and joint
  replay rather than reproducing that GP algorithm. A GP extension would need
  the full observation covariance and treatment of reused data across states.
- [Huggins et al. (2021)](https://arxiv.org/abs/1907.13117) exploit integral
  factorization and basis-rotation measurements, including postselection.
  Such measurement designs can improve resources, but require a different
  acquisition model. Independent Pauli counts alone cannot retrospectively
  supply their joint measurement records. They are outside this study's fixed
  measurement budget and independent-word model.

The following cost statements describe our implementation, not asymptotic
performance claims about those papers. Let `L` be measured Pauli words, `q`
affine constraints, `K` spin orbitals, `p` orbital rotations and `B` synthetic
draws.

| Approach | Expected benefit | Limits | Additional classical work |
|---|---|---|---|
| Joint affine correction | Remove particle-number/contraction inconsistencies cheaply | Estimated weights can bias the estimator; no PSD or full representability guarantee | Sparse constraint products and a `q`-dimensional pseudoinverse per batch; dense solve scales as `q^3` |
| Spatial PSD or DQG projection | Enforce stronger marginal consistency | DQG remains a relaxation; projection may worsen an individual observable | Conic solve; metric matrices have order `K^2`, and dense eigenvalue checks scale as `K^6` |
| Binomial constrained MLE | Respect the actual independent-count likelihood and physical relaxation | Boundary bias, incomplete identification and numerical convergence; no universal energy-MSE guarantee | Nonlinear convex likelihood plus the same conic constraints |
| Fixed-sector density-matrix MLE | Exact mixed-state representability within the chosen number sector | Rapid growth of sector dimension; no pure-state or total-spin guarantee | A PSD matrix with `d^2` entries, where `d=binom(n,Nalpha) binom(n,Nbeta)` |
| Joint coefficient covariance | Detect uncertainty in relative energies without discarding cancellation | Estimated covariance alone is not a finite-sample confidence guarantee | `C V C^T` and the Fourier transform, with only nine profile coefficients |
| Shared held-out acceptance | A finite-sample bound on harmful complete OO proposals | Conservative rejection; fewer training shots; does not protect later noisy UCC updates | One split and one classical full-proposal energy-difference bound |
| Robust or Bayesian/bootstrap proposals | Reduce sensitivity to unstable minima and expose multimodal updates | Training heuristic or model-dependent diagnostic, not a coverage guarantee | Approximately `B` complete correction/proposal replays for simulation-based diagnostics |

All of these postprocessing methods use zero additional quantum measurements.
The sector dimension is four for the H2 reference and 225 for four-electron
LiH with six spatial orbitals. Solver wall time and failure rates are measured
rather than inferred from matrix size alone. Newton additionally forms a
`p`-by-`p` Hessian; avoiding that Hessian does not imply that SOO is faster for
this small problem.

Our practical order is to use joint affine correction as the inexpensive
online candidate, assess DQG/MLE offline on identical counts, and retain raw
held-out validation for either proposer. Robust and posterior diagnostics
remain development experiments. Improving RDM RMSE is insufficient to promote
a method: its subsequent *physical* OO energy change, cost and failure rate
must improve. Estimated corrected energies need not be variational bounds on
the exact prepared-state energy.

For correlated UCCD states, the spin-orbital 1-RDM is not generally idempotent.
McWeeny idempotency purification, or the corresponding doubly-occupied
projector condition in the spin-summed convention, would impose a
Slater-determinant condition that the prepared state need not satisfy.
Eigenvalue clipping with fixed trace is a
useful projection building block, but independently purifying two marginals
can violate their mutual contraction relations; joint constraints are used
instead.

## Measurement statistics

`QuantumEngine::get_pauli_samples` returns the original per-word binomial
counts and means. `RdmMeasurementPlan` exposes the sparse linear reconstruction
map and energy weights. Identity contributions are exact. The old RDM API and
sampling sequence are retained; H2 statevector and finite-shot SOO/Newton
integration checks reproduced the old evaluator exactly.

With independent word estimates `mu`, RDMs have the form `r = M mu`, including
the identity column. For nine frozen-RDM trial energies, `E = E_nuc + C mu`.
Consequently `Cov(E) = C V C^T`, and real Fourier coefficients have covariance
`T C V C^T T^T`. `optimizer::uncertainty::FourierProfile` propagates this
covariance and computes the variance of an energy *difference*. Constant-offset
uncertainty cancels. More classical angle evaluations do not add measurements.

Explicitly, the same raw count vector determines all nine values, so
`E_i = E_nuc + c_i^T mu` and
`Var(E_i-E_j) = (c_i-c_j)^T V (c_i-c_j)`. Treating the values as independent
would replace this with the sum of marginal variances and discard the cross
term. That can either inflate or underestimate uncertainty. Additional angles
are additional linear views of the same data, not additional experimental
replicates. Conditional on the frozen RDM, the nine values interpolate its
trigonometric energy exactly even when that RDM is noisy.

With a deterministic linear correction `P`, the corrected covariance is
`M P V P^T M^T`; in this implementation the affine weights are estimated from
counts. Treating the fitted map as fixed is therefore only a local diagnostic.
The parametric bootstrap reruns that estimation and the full adaptive sweep
on each single resampled count vector. Independently perturbing nine energies
or independently bootstrapping successive coordinates would lose the shared
correlations and selection effects. A posterior-mean proposal can instead
optimize the mean reconstructed RDM because frozen-RDM energy is linear in
the RDM; averaging selected rotation angles does not have that property.

Finite-sample covariance diagnostics use Jeffreys posterior variances, avoiding
zero estimated uncertainty after all-equal outcomes. These are diagnostics,
not frequentist coverage guarantees. The nominal Aberth solver is retained.
The optional robust-score minimizer uses a grid and local refinement and does
not claim a certified global minimum. Independent Beta draws are diagnostic
Pauli posteriors, not automatically physically representable RDMs.

## Joint RDM constraints

The repository stores a real spin-summed 1-RDM and a half-scaled spin-summed
2-RDM. The implemented affine conditions are

- `trace(gamma1) = N`;
- `sum[p,q] gamma2[p,q,p,q] = N(N-1)/2`;
- `sum[q] gamma2[p,q,r,q] = (N-1) gamma1[p,r]/2`.

The covariance-weighted joint correction is performed in Pauli space. Estimated
weights make its finite-sample bias/variance an empirical question. It is not a
full physical reconstruction, and corrected Pauli estimates can exceed unit
bounds. Raw observations remain available for validation and analysis.

`scripts/reconstruct_rdms.py` additionally prototypes joint spatial positivity,
DQG weighted reconstruction, DQG-constrained binomial maximum likelihood, and
an exact fixed-(Nalpha,Nbeta) density-matrix likelihood reference. It verifies
primal constraint residuals, Pauli bounds, and metric eigenvalues. DQG imposes
necessary conditions only. Solvers' numerical statuses and residuals are
recorded rather than treated as proof of exact representability.

Real, Sz-conserving spin-orbital plans are compiled from the existing measured
Pauli catalog. Missing words are errors, never requests for extra shots. Total
spin purity is not assumed. McWeeny idempotency and energy regularization are
not used for the correlated prepared states. Reconstruction is frozen during
each OO sweep.

## Shared full-sweep acceptance rule

`orbital_statistics` is optional in method JSON. Its defaults preserve nominal
optimization. `mitigation` accepts `none` or `affine`; `acceptance` accepts
`nominal`, `split_nominal`, or `heldout`.

A complete small configuration is
[`examples/config_h2_uncertainty_oo.json`](../examples/config_h2_uncertainty_oo.json).
It uses three fixed macros, 10,000 shots/word, affine correction, held-out
acceptance and saved OO counts. To use the same rule with Newton, change
`method.orbital_optimizer.solver_config` to `{"type":"newton_raphson"}`.
After building the research evaluator in the scientific Python environment,
run it with a fresh output path:

```bash
mkdir -p local_runs
target/release/examples/evaluate_shot_noise \
  --config examples/config_h2_uncertainty_oo.json \
  --output local_runs/h2_uncertainty_oo_smoke.json
```

The research evaluator writes all recorded OO sufficient statistics to its
JSON output. The ordinary application logs the acceptance decision and saves
run conditions in the existing DB schema; it does not persist those count
arrays. The negative outer energy tolerance in this smoke configuration only
disables early stopping and is not a noise-aware convergence rule.

For split policies, hypergeometric subsampling partitions each word's existing
counts into 80% training and 20% validation outcomes. It neither draws extra
quantum measurements nor advances the measurement RNG. A complete sequential
sweep or Newton step is proposed using training data only. The heldout policy
tests exactly one complete proposal against its unchanged baseline.

Let `a` be the difference of the proposal and baseline Pauli energy weights,
and let `S` be validation shots per nonidentity word. Conditional on training,
Hoeffding's inequality gives the one-sided upper bound

`a . mu_validation + sqrt(2 log(1/alpha) sum_nonidentity(a_l^2/S))`.

An update is accepted only when this is strictly below
`-min_improvement_ha`. The same validation data estimate both energies, so
the difference weights include their correlations. The default trajectory
error budget is 0.05, divided over the fixed macro budget. A union bound
controls the probability of any harmful accepted OO update under the ideal
independent-shot model; it does not control noisy UCC updates or guarantee
lower final-state error. No validation-driven backtracking is performed.

The probability bound concerns the joint event "accepted and harmful", not
the conditional probability "harmful given acceptance". At low power, rare
accepted steps can mostly be harmful without violating that bound. It is not
a Bayesian probability that a particular accepted step is beneficial.

`split_nominal` is a diagnostic control for reserving validation shots without
using the acceptance decision. No-update controls prevent universal rejection
from being described as optimizer success.

Conditional on training, each validation contribution is
`a_l X_ls / S_l`, with range length `2 |a_l| / S_l`. Summing the squared
ranges gives `4 sum_l a_l^2/S_l`. Hoeffding's one-sided inequality therefore
places the true energy change below the implemented upper bound except with
probability `alpha`. The identity has no range contribution. This argument
allows arbitrary adaptive coordinate choices within the *training* sweep.
Fresh validation batches at later macros give the stated union bound even
though the states depend on earlier decisions. The numerical implementation
uses floating-point arithmetic; these are statistical guarantees under the
specified ideal independent-shot model, not interval-arithmetic certificates.

The bound validates raw counts and their corresponding means. Passing corrected
means as if they were independent held-out observations is rejected.

## Evaluation and artifacts

`examples/evaluate_shot_noise.rs` records count metadata, proposed rotations,
the common confidence decision, corrected/training energies, exact proposed
and applied energy changes, orthogonality, and separate correction/proposal/
validation timing. Exact diagnostics never determine optimization decisions.
`--exact-ucc` is an explicitly recorded controlled experiment: UCC uses exact
energies while OO remains finite-shot. Actual sampled-batch counters determine
its modeled quantum cost.

`examples/frozen_oo.rs` replays saved noiseless states and exports their
Hamiltonians, exact sampling probabilities, sparse maps, and spin-metric plans.
Those probabilities are used only to simulate independent measurement batches
and evaluate diagnostics. All twelve fixed-state variants share each batch.

The staged study is prepared using:

```bash
.paper-env/venv/bin/python scripts/run_uncertainty_study.py prepare \
  --work local_runs/uncertainty_oo_fresh \
  --previous local_runs/lih_shot_noise_20seeds_20261020
```

It preserves sources, binaries, Python versions, configs and seed schedules.
The fixed-state phase has 21 snapshots and 63 shot/snapshot cells, each with
1,000 independent batches. The pilot has 320 runs covering five development
seeds, 1.6/3.0 A, 1,000/100,000 shots, eight variants, and exact/noisy UCC.
The planned confirmation has 7,200 fresh-seed trajectories and noiseless
controls. Large phases use one CPU per trajectory with all numerical-library
thread counts fixed to one.

Build the research examples with the repository's scientific Python
environment before preparing a fresh workspace:

```bash
export PYO3_PYTHON="$PWD/.paper-env/venv/bin/python"
export LIBRARY_PATH="$PWD/.paper-env/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
srun -p defq -N 1 -n 1 -c 1 --mem=5G --time=00:20:00 \
  cargo build --release --locked -j 1 \
  --example evaluate_shot_noise --example frozen_oo --example bootstrap_oo \
  --example audit_oo_decisions

# After the prepare command above, keep a lightweight controller on the
# submission host. It does not reserve compute CPUs while waiting.
.paper-env/venv/bin/python -u scripts/drive_uncertainty_study.py \
  --work local_runs/uncertainty_oo_fresh
```

The controller creates verification fixtures, submits missing development
phases, validates the complete pilot, prospectively freezes confirmation
settings, and submits nine 800-task arrays and six controls. Each array uses
local indices 0--799 and a preserved global offset, respecting Slurm's array
index limit. Observation failures do not trigger experiment restarts.

The final audit replays all recorded finite-shot classical OO decisions from
saved Pauli counts, reconstructing the same training/validation partitions and
accepted rotations. It also independently recomputes every saved physical
energy, including the noiseless controls, from
the stored amplitudes and cumulative orbital rotation. These statevector
diagnostics do not feed optimization and add no quantum measurements. The
strict independent audit cannot certify completion while any phase is active.

`scripts/render_uncertainty_results.py` exports endpoint, primary-comparison,
run and noiseless-control CSV files and supplementary figures from the frozen
confirmation summary. It copies the existing sign-test results and bootstrap
intervals without recomputation, verifies the frozen source hash, and records
the input-summary hash. The original frozen report and summary are preserved.

The stronger reconstruction benchmark uses 27 representative snapshot/shot
cells and 25 matched batches each. The separate H2 sector-density reference
uses 25 batches and four estimators. Numerical failures remain in the results.
The `frozen_oo reconstructed` subcommand additionally evaluates SOO and Newton
updates from the saved fitted RDMs, paired with the raw and affine nominal
updates on those same batches. Reconstruction failures remain in denominators;
energy-change statistics are conditional on completed fits. These full-data
postprocessing comparisons do not claim held-out confidence guarantees.

Physical-estimator error comparisons retain the raw and affine baselines from
the exact same successfully fitted batches. Fit failures stay in the completion
denominators; comparing a successful-fit subset with an all-batch baseline can
otherwise confound estimator quality with selection. Mean fit times and
one-time model setup costs are recorded separately.

`bootstrap_oo` compares nominal, robust-score, and posterior-mean proposals for
both methods. Parametric bootstrap repeats the complete estimated affine
correction and adaptive sweep. Projected Beta draws instead use a fixed fitted
affine map; that diagnostic distribution is not an exact physically constrained
posterior. A Newton-ray risk search uses the same beta as the SOO profile risk
search. All original proposals use the same independent held-out test.
The study has eighteen original batches with 256 synthetic draws per
scheme/proposal, and zero additional quantum measurements. Synthetic draws
must not be counted as independent experimental trials.

The research evaluator also supports `--affine-ucc`, mutually exclusive with
`--exact-ucc`. A separate forty-run development extension compares affine
correction inside UCC energy evaluations against the matching raw-UCC pilot
baselines. It retains the same measurement budget. Estimated correction
weights can introduce parameter-dependent bias, so the mean corrected UCC
objective is not assumed to retain an exact degree-two Fourier representation.
These cases are excluded from the prespecified primary confirmation family.

Use `scripts/render_uncertainty_results.py --phase ucc_extension --work <study>`
to render the same-seed raw/affine UCC comparison separately from the primary
OO policies. It retains completion counts, uses identical joint-completion
subsets for paired endpoint means, and exports individual paired outcomes.

`scripts/verify_reconstruction_accuracy.py` provides selected LiH numerical
checks of the MLE prototype at default and tighter conic tolerances. Its
population-likelihood controls use fractional expected counts with a known
optimum, explicitly as numerical tests rather than measurements. Actual saved
counts are fitted with fresh models, without truth-based warm starts. These
checks help distinguish numerical error from finite-sample estimator behavior;
they do not establish exact optimality for every benchmark fit.

Primary and auxiliary evaluators have separately preserved binary/source
snapshots. Reported timings refer to those frozen versions, and all primary
variants use the same evaluator binary.

Primary comparisons concern completion-first paired final exact-state errors
for safeguarded, affine-corrected SOO versus matching Newton in nine cells.
Two-sided sign tests ask about win probability, not equality of mean error.
Holm adjustment covers the nine primary comparisons. Mean differences,
accuracy rates, tails, failures, and computational cost are also required.
Bootstrap analysis resamples entire trial-seed blocks rather than individual
macro iterations. Nonsignificance does not establish equivalence.

`scripts/analyze_safeguard_tradeoff.py --work <study>` additionally summarizes
the completed fixed-state safety/utility tradeoff by shot count. It compares
splitnominal and heldout at identical training budgets, retains per-cell CSV
rows, and distinguishes unconditional harmful acceptance from harm conditional
on acceptance. Its pooled fixture statistics are descriptive and do not
recompute primary confirmation comparisons.

## Completion audit

- Measurement API, splitting, covariance, affine correction and the shared
  validation wrapper are implemented. Forty-three library tests pass, including
  rejection of invalid held-out observations. H2 and completed LiH baseline
  replays match the previous evaluator exactly. Independent H2 Fock-operator
  checks reproduce the D/Q/G maps for particle numbers zero through four.
- All 63,000 fixed-state batches and all 320 pilot trajectories have completed
  and passed analysis. Six noiseless controls have completed. The protocol is
  prospectively frozen, and the 7,200 fresh-seed trajectories remain active.
- All eighteen whole-sweep diagnostic tasks have completed and passed analysis.
  The small H2 reference completed all 25 batches for every estimator. Larger
  reconstruction benchmarks and the UCC correction extension remain active.
- Full confirmation results, final scientific reports, scheduler accounting,
  artifact inspection and the requirement-by-requirement completion audit
  remain required. Submission or partial results do not complete the objective.
- Eight Python pipeline tests pass, including matched completed-fit baselines,
  frozen-statistic presentation,
  expired scheduler IDs and rejection of changed cached reconstruction inputs.
  Supplemental plot rendering was checked with a labeled synthetic fixture;
  that fixture is not scientific evidence.
