# LiH 1.6 Å shot-noise comparison

This experiment compares OO(seq), raw analytic Newton OO, and stabilized
analytic Newton OO at LiH/STO-3G, 1.6 Å, four electrons and six spatial orbitals
without frozen core. "Mizukami-type" refers to the analytic orbital
gradient/Hessian formulation implemented here; it does not imply execution of
the original authors' program.

## Protocol

- Start every trajectory from RHF orbitals and zero UCCD amplitudes.
- Perform ten fixed macro iterations. Each contains one orbital update/sweep
  and one sequential five-point sweep of all 76 UCCD amplitudes.
- Compare 1,000, 10,000 and 100,000 shots per independent nonidentity Pauli word,
  with five trial seeds and one noiseless control for each optimizer: 48 runs.
- Use the same full-RDM measurement model in OO and UCC. A Pauli estimate is
  shared by all RDM elements containing that word; different words are sampled
  independently. No hardware noise or measurement grouping is included.
- Keep stabilized Newton settings fixed: Hessian floor `1e-4`, step-norm cap
  `0.2`, gradient tolerance `1e-6`, at most twenty backtracks. Raw Newton bypasses
  all three stabilization measures.

Trial base seeds are `20261020, 21261020, 22261020, 23261020, 24261020`.
They must be separated because the engine uses `base_seed + evaluation_count`.
Consecutive base seeds would reuse measurement RNG seeds in different trials.
The preparation command rejects overlapping seed windows. Matched trial seeds
are intentionally reused across optimizers and shot counts.

## Measurements and scoring

`examples/evaluate_shot_noise.rs` calls the existing orbital and UCC optimizers
and preserves their sampling evaluation order. It additionally computes exact
statevector diagnostics without calling the sampling path or advancing its seed
counter. A small H2 regression check confirmed that these diagnostics preserve
the existing noisy optimization trajectory.

Record, after every macro iteration:

- The fresh finite-shot energy estimate and the exact energy of the optimized
  state under the current rotated Hamiltonian.
- UCC amplitudes and cumulative orbital rotation, with an orthogonality check.
- OO energy before/after the update for both the noisy frozen RDM and the exact
  state; Newton shift, step norm, line-search scale and update status.
- RDM evaluation count, modeled shots, and elapsed time.

The main score is the **final exact state energy minus FCI**, after the fixed
budget, not the minimum noisy energy or the best exact iterate. Exact diagnostic
values do not select steps or determine termination. A completed run therefore
does not by itself demonstrate convergence.

Every full run uses `1 + 10*(1 + (1 + 5*76) + 1) = 3831` measured/optimization
RDM evaluations. With 2,778 measured words, the modeled total is
`3831 * 2778 * shots_per_word`. Extra exact diagnostic evaluations are excluded
from that modeled measurement budget. Noiseless controls have no finite-shot
hardware cost assigned to them.

Report the mean, sample standard deviation, median and maximum final error,
completion/failure counts, and final fractions within 1.6 mHa and 0.1 mHa of FCI.
Failures are included in the denominators of accuracy rates; completed-only
energy statistics are explicitly labeled. Five seeds support a preliminary
comparison, rather than a precise failure-probability estimate.

## Running on Slurm

From the repository root, using the dedicated scientific Python environment:

```bash
export PYO3_PYTHON="$PWD/.paper-env/venv/bin/python"
export LIBRARY_PATH="$PWD/.paper-env/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
srun -p defq -N 1 -n 1 -c 4 --mem=8G --time=00:20:00 \
    cargo build --release --locked --example evaluate_shot_noise -j 4

# Choose a fresh workspace. Preparation preserves source and binary snapshots.
WORK="$PWD/local_runs/lih_shot_noise_fresh"
srun -p defq -N 1 -n 1 -c 2 --mem=4G --time=00:10:00 \
    .paper-env/venv/bin/python scripts/evaluate_lih_shot_noise.py \
    prepare --work "$WORK"

sbatch --array=0-47%16 \
    --output="$WORK/logs/slurm-%A_%a.out" \
    --error="$WORK/logs/slurm-%A_%a.err" \
    scripts/evaluate_lih_shot_noise.sbatch "$WORK"

# After all array tasks finish:
.paper-env/venv/bin/python scripts/evaluate_lih_shot_noise.py \
    summarize --work "$WORK"
```

Each task uses two CPUs and 2 GB; the array concurrency may be adjusted to
available resources. The worker runs the preserved binary and script snapshot.
The manifest records package versions, source/binary hashes, reference energies,
seeds and input paths. Individual execution records include Slurm IDs, hostname,
exit status and affinity. Exact FCI is recomputed with PySCF in the same basis.

The summarizer emits `summary.json`, `report.md`, and PNG/PDF figures for exact
energy trajectories, final errors and Newton diagnostics. Its source and package
versions are also preserved. An optional `--allow-partial` creates a progress
preview; final conclusions require all tasks to finish.

## 2026-10-20 evaluation

The independent-seed workspace is
`local_runs/lih_shot_noise_20261020_independent`. Its generated `report.md` and
figures contain the results. An earlier incomplete workspace,
`local_runs/lih_shot_noise_20261020`, was canceled because its consecutive base
seeds overlapped; those pilot trajectories are excluded from the final analysis.

All 48 runs completed ten iterations, with no solver/process failures. The
noiseless controls all reached an FCI error of approximately `0.010571` mHa.
The final exact-state FCI errors below are means ± sample standard deviations
over five trial seeds, in mHa:

| Shots / Pauli word | OO(seq) | Raw Newton | Stabilized Newton |
| ---: | ---: | ---: | ---: |
| 1,000 | 23.45448 ± 6.72988 | 32.27180 ± 12.02955 | 25.21391 ± 6.62584 |
| 10,000 | 4.85274 ± 6.61450 | 4.73043 ± 2.04915 | 2.26203 ± 0.46338 |
| 100,000 | 0.60875 ± 0.20667 | 0.96144 ± 0.44508 | 0.58842 ± 0.24265 |

At 100,000 shots, final errors within 1.6 mHa occurred in 5/5 sequential,
4/5 raw-Newton and 5/5 stabilized-Newton runs. No final trial reached 0.1 mHa.
These are endpoint accuracy counts, not convergence rates.

None of the 150 finite-shot stabilized-Newton updates needed a Hessian shift,
step cap or backtrack, and none was rejected. The observed raw/stabilized
endpoint differences therefore cannot be attributed to the stabilization
controls. The two code paths use pivoted LU and Cholesky respectively;
finite-precision differences can change later stochastic samples even with
matched base seeds. Five seeds and inactive stabilization do not establish a
stabilization benefit or general optimizer superiority.

Validation checked the same 3,831 measured/optimization RDM evaluations in each
run, disjoint trial seed windows, and all exact energies above the FCI lower
bound. The largest recorded orbital orthogonality error was `2.44e-15`.

### Accuracy and paired comparisons

Additional descriptive accuracy statistics and matched-seed comparisons can be
reproduced from the saved results without rerunning the optimizer:

```bash
.paper-env/venv/bin/python scripts/compare_shot_noise_accuracy.py \
    --work local_runs/lih_shot_noise_20261020_independent
```

This creates `accuracy_comparison.md`, `accuracy_comparison.json` and
`figures/paired_accuracy.{png,pdf}`. It reports mean/median/RMSE/range, paired
error differences and two-sided sign tests, with Holm adjustment over the nine
method-pair/shot-level comparisons. The sign test concerns which method more
often has lower error; it does not test equality of mean errors.

At 100,000 shots, OO(seq) has lower error than raw Newton in all five matched
trials, with a mean advantage of `0.35269` mHa. Its mean error differs from
stabilized Newton by `0.02034` mHa. The unadjusted two-sided sign-test p-value
for five wins out of five is `0.0625` (`0.5625` after Holm adjustment here).
These data support a descriptive comparison but do not establish superiority
at the 5% level. Non-rejection also does not establish equivalence.

At 10,000 shots, OO(seq)'s median error is `2.01862` mHa, while its mean is
`4.85274` mHa because one trial has `16.67752` mHa error. Both central tendency
and the tail of the error distribution should be considered.

## Twenty-seed stretched-geometry follow-up

The follow-up uses **1.6, 2.4 and 3.0 Å**, twenty trial seeds, the same three
optimizers and shot counts, and ten fixed macro iterations. There are 549
configurations including nine noiseless controls. The completed 48-run 1.6 Å
pilot is reused after checking evaluator-binary hashes, physical/method input
settings, FCI references and trajectory hashes; 501 new trajectories are run.

The study workspace is `local_runs/lih_shot_noise_20seeds_20261020` with `r160`,
`r240` and `r300` children. `study.json` records array submissions, a dependent
automatic analysis job, and analysis-source hashes. New tasks use one CPU;
a deterministic H2 check reproduced the earlier two-CPU noisy trajectory.
Accuracy statistics can be combined, but mixed-allocation wall times are not
used for performance comparisons.

The preparation script supports geometry and seed-count arguments, for example:

```bash
.paper-env/venv/bin/python scripts/evaluate_lih_shot_noise.py prepare \
    --work "$PWD/local_runs/lih_r240_twenty_fresh" \
    --distance 2.4 --seed-count 20 --cpus 1
```

For a compatible 1.6 Å extension, add
`--reuse local_runs/lih_shot_noise_20261020_independent`. Only exact matches of
physical and method settings are copied. `pending_task_indices` in the new
manifest lists the array indices that still require execution. Submit with
`--cpus-per-task=1` to match this example's manifest.

Twenty base seeds are spaced by one million, beginning at 20261020. This keeps
each trial's `base_seed + evaluation_count` window disjoint. Seeds are matched
across methods, shot counts and geometries.

After completion, the aggregate analysis is reproducible with:

```bash
.paper-env/venv/bin/python scripts/summarize_lih_geometry_study.py \
    --study local_runs/lih_shot_noise_20seeds_20261020
```

It produces a root `report.md`, `study_summary.json`, `geometry_summary.csv`,
validation/accounting records and PNG/PDF figures. These compare final exact
FCI errors, medians, accuracy fractions, excess error above each method's
noiseless **same-budget** control, and stabilization interventions. Stretched
noiseless controls are not assumed converged after ten iterations.

The sign tests use jointly completed trial pairs; failures are retained in
completion/accuracy rates and are not given artificial finite errors.
Multiplicity adjustment covers all 27 geometry/shot/method-pair comparisons.
The follow-up remains exploratory, including the five-seed pilot, rather than
a preregistered confirmatory superiority test.

### Completed twenty-seed results

All 549 trajectories completed, including the 48 reused pilot trajectories
and 501 new runs. Scheduler accounting confirmed that all 501 new jobs
completed successfully. All three geometries used the same evaluator binary,
disjoint trial seed windows, and 3,831 measured/optimization RDM evaluations
per trajectory. The largest orbital orthogonality error was `3.11e-15`.

Final exact-state FCI errors, mean ± sample SD over twenty seeds, in mHa:

| R (Å) | Shots / Pauli word | OO(seq) | Raw Newton | Stabilized Newton |
| ---: | ---: | ---: | ---: | ---: |
| 1.6 | 1,000 | 27.055 ± 9.099 | 26.784 ± 8.144 | 26.507 ± 7.901 |
| 1.6 | 10,000 | 4.587 ± 4.786 | 4.383 ± 3.673 | 3.173 ± 3.784 |
| 1.6 | 100,000 | 0.602 ± 0.215 | 0.842 ± 0.726 | 0.724 ± 0.778 |
| 2.4 | 1,000 | 27.687 ± 20.419 | 23.275 ± 8.029 | 22.344 ± 11.821 |
| 2.4 | 10,000 | 2.501 ± 0.435 | 2.321 ± 0.477 | 2.353 ± 0.391 |
| 2.4 | 100,000 | 0.715 ± 0.211 | 0.781 ± 0.192 | 0.673 ± 0.201 |
| 3.0 | 1,000 | 21.208 ± 3.467 | 21.414 ± 3.853 | 22.668 ± 3.913 |
| 3.0 | 10,000 | 2.736 ± 0.440 | 2.538 ± 0.290 | 2.631 ± 0.378 |
| 3.0 | 100,000 | 0.718 ± 0.160 | 0.728 ± 0.130 | 0.688 ± 0.172 |

The clearest descriptive result favoring sequential OO is at 1.6 Å and
100,000 shots: it has the smallest mean and SD, and all 20 final errors are
within 1.6 mHa, versus 18/20 for raw Newton and 19/20 for stabilized Newton.
This does not establish general superiority. Sequential OO beats raw Newton
in 14/20 matched trials there (two-sided sign-test `p=0.11532`), and beats
stabilized Newton in 9/20. Stabilized Newton has a smaller median error
(`0.529` versus `0.629` mHa); a few larger Newton errors affect its mean and SD.

All 27 Holm-adjusted sign-test p-values are `1.0`. At 10,000 shots, both Newton
variants have smaller mean errors than sequential OO at every geometry.
Failure to detect a difference does not demonstrate equivalence. The evidence
supports reporting achieved accuracy and distributions, but not claiming
general sequential-OO superiority in shot-noise resistance. Its avoidance of
orbital gradient/Hessian construction is a structural advantage; isolated
classical OO runtime was not benchmarked. Quantum measurement budgets are
equal in this experiment.

At 100,000 shots, all 20 sequential-OO trials meet the 1.6 mHa threshold at
each geometry. Both Newton methods also meet it in all trials at 2.4 and
3.0 Å. None of the 1,800 finite-shot stabilized-Newton updates required a
shift, cap, backtrack or rejection, so these conditions do not measure a
benefit from activating stabilization.

The generated study-root `report.md` includes all medians, accuracy fractions,
paired tests, and convergence/endpoint figures in PNG and PDF. The dependent
analysis job initially failed because `sacct` on the compute node could not
connect to the accounting database at `localhost`. The unchanged analysis
pipeline was rerun successfully from the submission environment, where
accounting was available; no trajectories were rerun. `analysis_recovery.json`
and the preserved logs record this recovery.
