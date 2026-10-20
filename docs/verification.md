# Recorded numerical verification

The paper reproduction is supported by actual completed Slurm calculations,
not just prepared scripts. This report distinguishes full-scan validation,
historical numerical agreement and their limits. Public tables are in
[results/paper](../results/paper/).

## Full scan

The 2026-10-19 Slurm array `10621` ran four methods at fourteen LiH distances
(56 tasks) on the CPU partition `defq`. Every array element finished with
`COMPLETED` and exit code `0:0`; each has a completed SQLite result and logs.
Read-only revalidation checks database integrity, conditions, finite energies,
unique history continuity, convergence flags and the outer stopping rule.

| Method | Points | Converged | Reached the 100-macro cap |
|---|---:|---:|---:|
| UCCD(seq) | 14 | 14 | 0 |
| UCCSD(seq) | 14 | 11 | 3 |
| OO(seq)-UCCD(seq) | 14 | 14 | 0 |
| OO(lbfgs)-UCCD(seq) | 14 | 14 | 0 |

The capped points are UCCSD at 3.0, 3.2 and 3.4 Å. Their final consecutive-energy
changes are approximately `1.72e-8`, `4.28e-8` and `1.50e-8` Ha, respectively.
Completion is therefore not synonymous with convergence.

At 1.6 Å the historical numerical checkpoints are:

| Method | Total energy (Ha) | Macro iterations |
|---|---:|---:|
| UCCD(seq) | -7.881875306295823 | 4 |
| UCCSD(seq) | -7.882313807184405 | 7 |
| OO(seq)-UCCD(seq) | -7.882313806735446 | 7 |
| OO(lbfgs)-UCCD(seq) | -7.882313807311107 | 7 |

The checker uses `1e-8` Ha checkpoint tolerance. These values are numerical
checkpoints, not a promise of last-digit identity across platforms.

## Historical comparison

The recovered archive covers all 56 physical inputs. With task names and the
outer iteration cap treated separately, all physical settings match the rerun.

- **51 comparable endpoints:** all agree within `1e-8` Ha. The largest final
  energy difference is `5.569678052097515e-11` Ha, for an L-BFGS OO point.
- **Five extended UCCSD runs:** at 2.6, 2.8, 3.0, 3.2 and 3.4 Å, the recovered
  archive ends at 30 macros. The rerun continues to 34, 61, 100, 100 and 100.
  Missing replacement histories prevent a full historical final-energy check
  for those five points.
- **662 common history entries:** the largest intermediate difference is
  `1.233981894444014e-8` Ha. One entry exceeds `1e-8` Ha. This does not support
  a claim that every intermediate energy is identical within that tolerance.

The later 100-macro rerun validates the manuscript stopping protocol at all
56 points, including capped points. It does not recover unavailable historical
artifacts or establish pixel-identical reproduction of the manuscript PDFs.

## Provenance and environment

The recorded executable SHA-256 is
`4c400b643a08b90f511f2396252ce3d8dadde41c47a14e8b2067d3a85d71b6a0`.
Its saved source commit is `6cd1b8b0c49c07439f7385eb2d99416fc3587b15`.
The accompanying uncommitted patch changes documentation, ignore rules and
convergence plotting, not numerical Rust sources. Source archives, lockfiles,
input JSONs, package versions and logs were preserved with the run.

The build log records Rust 1.99.1, Cargo 1.99.1, Python 3.12.3, NumPy 1.26.4,
SciPy 1.13.1 and PySCF 2.7.0; plotting used Matplotlib 3.9.4. One CPU was
allocated per full-scan task, with BLAS/OpenMP limited to one thread.

The current main branch subsequently exposes the task API and expresses the
PySCF spin argument directly as `N_alpha-N_beta`. It retains the same numerical
optimization routines. New public-release checks and their results should be
recorded with the actual executable/source snapshot rather than attributed to
the historical binary.

## Inspectable artifacts

`results/paper/energies.csv` contains all 56 endpoints and their RHF/FCI
references. `macro_iterations.csv` contains unique histories, including an
explicit recomputed RHF iteration zero. `scheduler.csv` records the original
56 array outcomes. `historical_comparison.json` retains numerical comparisons
without machine-specific database paths. `provenance.json` describes settings,
input/result hashes and the limits of the recovered reference.

The export tool validates raw databases before writing tables. The raw files
are read-only inputs; export and plotting do not resume an optimizer or require
cloud credentials.

## Publication-preparation recheck (2026-10-20)

The original 56 databases were checked again using read-only connections, and
the historical comparison was regenerated. Its 51 comparable endpoints and
reported limitations were confirmed. The tracked scan template generated inputs
identical to all 56 saved rerun configurations.

At this publication-preparation check, main was rebuilt on Slurm. All 22 library tests passed,
both maintained examples compiled, and the CLI/API H2 examples both converged
to `-1.1372838344884912` Ha in two macros. A fresh four-task LiH array at
1.6 Å passed all checkpoints; its largest reference-energy deviation was
`1.1191048088221578e-13` Ha. Macro counts were 4, 7, 7 and 7 for the four
methods listed above. This is a new four-point check, not a new 56-point scan.

Portable CSV plots were rendered and inspected. Before pipeline consolidation,
the database convergence plotter was run across all 56 histories, producing 28 figures;
the three capped UCCSD histories end at macro 100 after duplicate-row removal.
Citation metadata passed CFF schema 1.2.0 validation, and a clean dependency
resolution of `requirements.txt` succeeded without changing the running
scientific environment.

Machine-readable results and scheduler outcomes are in
[`public_release_checks.json`](../results/paper/public_release_checks.json).
At that earlier check, LaTeX compilation was not rerun because `latexmk` was
not installed. The subsequent figure-update typesetting check is recorded below.

## Unified pipeline recheck (2026-10-20)

After removing the unused custom L-BFGS implementation, the release build
succeeded, all 22 library tests passed and both maintained examples compiled.
The central finite-difference function and the active `argmin` optimizer were retained.

The generated full-analysis job was tested on copies of the 56 existing
completed databases, preserving their original source and executable provenance.
It validated all tasks, exported 56 endpoints and 963 history rows, and plotted
the CSVs. Every exported numeric value matched the included tables exactly.
This postprocessing check does not represent 56 new VQE calculations.

The PES and convergence energy-error curves were checked against the CSV
energies and FCI references at 71 plotted points. Both error plots display
`E - E_FCI` as in the original paper; convergence uses a single panel.
Original databases and published numerical/provenance files were left unchanged.

A fresh four-task Slurm smoke array using the rebuilt executable also passed.
All four methods converged at 1.6 Å with the same macro counts (4, 7, 7, 7);
the largest checkpoint-energy difference was `1.119e-13` Ha.
Smoke analysis validated the four tasks without requesting a full CSV export.

## Manuscript figure update (2026-10-20)

The manuscript's previous arXiv v1 figure assets were replaced by figures
regenerated from the included local-rerun tables. Both PES and 1.6 Å convergence
use the complete Slurm rerun dataset described above. The captions and Results
section identify that dataset, the 53 converged points and three capped points.
The original arXiv v1 record was not modified.

[Figure provenance](figures/provenance.json) records the two CSV hashes,
upstream dataset provenance, plotting-script hash, Python/Matplotlib versions
and four output hashes. The 71 plotted energy-error points were checked against
the tables. Fonts are embedded in the PDFs; the regenerated assets use a
layout matching the original paper, including its colors, line styles,
markers, legends, font sizes and axis labels. The PES has the original
1.8:1 panel-height ratio; convergence retains its original single panel.

The updated manuscript compiled with an isolated Tectonic 0.17.0 installation.
Its eight-page PDF was inspected at the page containing both figures; figure
labels and captions were readable, and no references remained unresolved.
There is an existing 3.88 pt overfull box in a mathematical expression, unrelated
to the figure replacement. The default `docs/build.sh` still uses `latexmk`.

## Local-only output check (2026-10-20)

After replacing the manuscript figures, GCS authentication/upload code and its
direct HTTP/URL-encoding dependencies were removed from main. Remaining package
versions were retained. Numerical integral, circuit and optimizer algorithms
were not changed by this removal.

The release build, 22 library tests and both example builds passed. CLI and API
H2 calculations both returned `-1.1372838344884912` Ha. A CLI run with legacy
cloud-launcher variables still wrote to its explicit local SQLite path; storage
URIs were rejected before calculation. Original numerical tables and their
provenance records remained unchanged.

## History and provenance

Before publication, history was rewritten to remove downloaded article PDFs,
submission correspondence, old drafts, generated manuscript products, caches
and a compiled development executable. Personal paths and a private storage
identifier were replaced in old text files. Author identities, emails and
timestamps were retained.

The [commit mapping](history-commit-map.csv) covers all 42 original commits.
The numerical Rust sources and Cargo lockfile at each mapped original commit
were verified against the pre-rewrite backup. This describes the historical
mapping, not a requirement that later maintenance commits retain unused code.
The annotated `pre-newton-oo` tag retains its baseline meaning and annotation,
now pointing to `e0e0a8e9c419d79f74cdaa0abbab5a06cc04fbc4`.

Original execution identifiers, source patches, binary hashes and result hashes
remain in their run records. They were not rewritten to suggest a new execution.
The dataset's [history mapping](../results/paper/history_rewrite.json) identifies
its public source commits. Keep the mapping while these old identifiers are
referenced. The prose can later be shortened or archived without discarding
numerical data and their provenance.

Private-archive comparison tooling has been retired. The previously recorded
comparison report remains as verification evidence; new reproduction runs use
public inputs, tables and numerical checkpoints.
