# Verified LiH numerical data

These tables come from the completed 56-task Slurm statevector rerun on
2026-10-19. They were exported after read-only database validation. They are
fresh rerun data, not a claim that every original manuscript database was
recovered. See [the verification report](../../docs/verification.md).

| File | Contents |
|---|---|
| `energies.csv` | 56 endpoints: four methods × fourteen bond lengths, RHF/FCI references, stopping information |
| `macro_iterations.csv` | 963 unique history rows, including recomputed RHF iteration zero for each run |
| `scheduler.csv` | Original 56 Slurm array outcomes, exit codes and elapsed times |
| `historical_comparison.json` | Sanitized comparison with recovered historical endpoints and shared histories |
| `history_rewrite.json` | Original-to-public source identifiers after history cleanup; original run records retained |
| `provenance.json` | Executable/source/input/result hashes, settings and recorded scientific-package versions |
| `public_release_checks.json` | Current-main build/tests, four fresh Slurm checkpoints, and publication-preparation checks |

Energies are Hartree; distances are Å. `error_from_fci_hartree` is the signed
energy difference, without a plotting floor or absolute-value transformation.
The plotter displays positive `E - E_FCI` values on logarithmic error axes,
using the original paper's styles without a display floor. Convergence at
1.6 Å uses the original single-panel layout; CSV values are preserved.
The manuscript figures are regenerated from these tables; their
[provenance record](../../docs/figures/provenance.json) identifies input and
output hashes and the plotting implementation.
`converged` records the consecutive-energy criterion, not accuracy relative to
FCI. Three UCCSD points reach the 100-macro cap. A repeated final logger row is
collapsed in the exported histories.

The historical comparison has 51 comparable endpoints, all within `1e-8` Ha
(maximum difference `5.57e-11` Ha). Five recovered UCCSD histories end earlier
than the rerun. One shared intermediate-history entry differs by more than
`1e-8` Ha. These limitations are retained explicitly in the JSON.

Replot the tables without SQLite, PySCF or cloud access:

```bash
python scripts/plot_paper_results.py results/paper \
  --out-dir local_runs/paper-table-figures
```

To export a new full run, choose a fresh output directory:

```bash
python scripts/export_paper_results.py local_runs/paper-full \
  --out-dir local_runs/paper-export
```

The exporter recomputes RHF/FCI with PySCF. Optional `--accounting-file` adds
scheduler records from
`sacct --format=JobID,State,ExitCode,ElapsedRaw,NodeList -P -n`.
The included historical comparison is a retained verification record; the
private-archive comparison tool is no longer part of the reproduction workflow.
The original raw databases, configs and full logs remain in the producing
workspace, and can be regenerated using the tracked scan template.
