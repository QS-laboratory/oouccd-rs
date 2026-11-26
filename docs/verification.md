# Recorded numerical verification

The paper reproduction is supported by actual completed Slurm calculations,
not just prepared scripts. This report distinguishes full-scan validation,
historical numerical agreement and their limits. Public tables are in
[results/paper](../results/paper/).

## Full scan

The 2026-11-26 Slurm array `26306` ran four methods at fourteen LiH distances
(56 tasks) on the CPU partition `defq`. Every array element finished with
`COMPLETED` and exit code `0:0`; each has a completed SQLite result and logs.
Read-only validation checks database integrity, conditions, finite energies,
unique history continuity, convergence flags and the outer stopping rule.

| Method | Points | Converged | Reached the 100-macro cap |
|---|---:|---:|---:|
| UCCD(seq) | 14 | 14 | 0 |
| UCCSD(seq) | 14 | 11 | 3 |
| OO(seq)-UCCD(seq) | 14 | 14 | 0 |
| OO(lbfgs)-UCCD(seq) | 14 | 14 | 0 |

The capped points are UCCSD at 3.0, 3.2 and 3.4 Å. Their final consecutive-energy
changes are approximately `1.72e-8`, `4.28e-8` and `1.50e-8` Ha, respectively.
Completion is therefore not synonymous with convergence. These three tasks were
the longest, at up to 4962 s each with one CPU.

At 1.6 Å the historical numerical checkpoints are:

| Method | Total energy (Ha) | Macro iterations |
|---|---:|---:|
| UCCD(seq) | -7.881875306295823 | 4 |
| UCCSD(seq) | -7.882313807184405 | 7 |
| OO(seq)-UCCD(seq) | -7.882313806735446 | 7 |
| OO(lbfgs)-UCCD(seq) | -7.882313807311107 | 7 |

The checker uses `1e-8` Ha checkpoint tolerance; the largest deviation in this
scan was `1.119e-13` Ha. These values are numerical checkpoints, not a promise
of last-digit identity across platforms.

## Historical comparison

The recovered archive covers all 56 physical inputs. With task names and the
outer iteration cap treated separately, all physical settings match the scan.

- **51 comparable endpoints:** all agree within `1e-8` Ha. The largest final
  energy difference is `5.569678052097515e-11` Ha, for an L-BFGS OO point.
- **Five extended UCCSD runs:** at 2.6, 2.8, 3.0, 3.2 and 3.4 Å, the recovered
  archive ends at 30 macros. The scan continues to 34, 61, 100, 100 and 100.
  Missing replacement histories prevent a full historical final-energy check
  for those five points.
- **662 common history entries:** the largest intermediate difference is
  `1.233981894444014e-8` Ha. One entry exceeds `1e-8` Ha. This does not support
  a claim that every intermediate energy is identical within that tolerance.

The 100-macro scan validates the manuscript stopping protocol at all 56 points,
including capped points. It does not recover unavailable historical artifacts
or establish pixel-identical reproduction of the arXiv v1 figures.

## Provenance and environment

The executable SHA-256 is
`e152f71993bc320c74a676fafb93b0367fd857108b499284b170922d74284c9a`, built from
source commit `5cb6767429a9b08d89a3288bbf4be085f334f2f0` with no uncommitted
changes. Source archives, lockfiles, input JSONs, package versions and logs
were preserved with the run.

The build log records Rust 1.99.1, Cargo 1.99.1, Python 3.12.3, NumPy 1.26.4,
SciPy 1.13.1 and PySCF 2.7.0; plotting used Matplotlib 3.9.4. One CPU was
allocated per task, with BLAS/OpenMP limited to one thread and at most 16
tasks running at once.

## Inspectable artifacts

`results/paper/energies.csv` contains all 56 endpoints and their RHF/FCI
references. `macro_iterations.csv` contains unique histories, including an
explicit recomputed RHF iteration zero. `scheduler.csv` records the 56 array
outcomes. `historical_comparison.json` retains numerical comparisons without
machine-specific database paths. `provenance.json` describes settings and
source, executable, input and result hashes.

The export tool validates raw databases before writing tables. The raw files
are read-only inputs; export and plotting do not resume an optimizer or require
cloud credentials.

## Manuscript figures

The manuscript figures in `docs/figures/` are plotted from the included tables.
[Figure provenance](figures/provenance.json) records the two CSV hashes,
upstream dataset provenance, plotting-script hash, Python/Matplotlib versions
and four output hashes. Fonts are embedded in the PDFs; the layout matches the
arXiv v1 figures, including colors, line styles, markers, legends, font sizes
and axis labels. The PES has the original 1.8:1 panel-height ratio;
convergence retains its original single panel. The arXiv v1 record was not
modified.
