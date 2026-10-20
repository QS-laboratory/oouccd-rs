# Reproducing the LiH paper calculations

This guide reproduces the numerical protocol in
[main_rev3.tex](main_rev3.tex). It uses tracked inputs and local SQLite outputs;
neither the historical cloud archive nor a private `infra/` directory is needed.
The existing verification and its limitations are documented in
[verification.md](verification.md).

## Protocol

| Setting | Value |
|---|---|
| Molecule and grid | LiH, 1.0–3.6 Å inclusive, 0.2 Å spacing |
| Basis / charge / PySCF spin | STO-3G / 0 / 0 |
| Active space | Four electrons, six spatial orbitals; no frozen core |
| Register and reference | Twelve qubits; RHF determinant |
| Methods | UCCD(seq), UCCSD(seq), OO(seq)-UCCD(seq), OO(lbfgs)-UCCD(seq) |
| Initialization | Zero amplitudes independently at every geometry |
| Circuit order | Doubles first, then singles; fixed lexicographic order within rank |
| Orbital coordinates | Eight occupied–virtual spatial rotations |
| Sequential updates | One sweep; five points per amplitude, nine per orbital coordinate |
| Outer termination | Consecutive-energy change below `1e-8` Ha, or 100 macros |
| RDM evaluation | Exact statevector; no measurement or hardware noise |

One OO macro updates orbitals at fixed RDM, absorbs the rotation into the
integrals, then sweeps the amplitudes. Amplitudes carry between macros.
Sequential OO makes one sweep per macro; L-BFGS runs its configured inner
optimization, up to 200 iterations. Total energies include nuclear repulsion.
The 100-macro cap is part of the protocol and does not imply convergence.

The complete template is [examples/lih_paper_scan.json](../examples/lih_paper_scan.json).
The generated ordering is:

| Indices | Method | Index at 1.6 Å |
|---|---|---:|
| 0–13 | UCCD(seq) | 3 |
| 14–27 | UCCSD(seq) | 17 |
| 28–41 | OO(seq)-UCCD(seq) | 31 |
| 42–55 | OO(lbfgs)-UCCD(seq) | 45 |

## 1. Prepare Python and Rust

Run commands from the repository root. The verified environment used Python
3.12.3 and the versions in `requirements.txt`. Rust, CMake, a C/C++ toolchain,
`pkg-config`, OpenSSL development libraries and Python's linkable runtime must
be available on the build node.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
source scripts/python_env.sh
```

On a machine where local compilation is permitted:

```bash
cargo build --release --locked
```

On Slurm, use an appropriate CPU partition; `defq` is the tested site's name:

```bash
sbatch --partition=defq scripts/build_paper.sbatch
```

Wait for a successful build before preparing runs. Keep the Python environment
at the same absolute path on all compute nodes while jobs run. For a different
venv, source `scripts/python_env.sh /absolute/path/to/python` before building
and pass that interpreter to the preparer. The helper preserves venv paths
rather than resolving their executable symlinks to system Python.

The build script uses the installed Rust toolchain; it does not install one
or require a cluster-specific Rust directory. Its CPU count and memory/time
limits can be overridden by normal `sbatch` options.

## 2. Run the four-point smoke array

```bash
"$PYO3_PYTHON" scripts/prepare_paper_slurm.py local_runs/paper-smoke \
  --scope smoke --python "$PYO3_PYTHON" \
  --cpus 1 --concurrency 4 --partition defq

sbatch --test-only local_runs/paper-smoke/run.sbatch
job_id=$(sbatch --parsable local_runs/paper-smoke/run.sbatch)
sbatch --dependency="afterok:$job_id" local_runs/paper-smoke/analyze.sbatch
```

The preparer creates a **new** workspace, all 56 input JSONs, a copied binary,
environment/source records, `run.sbatch` and `analyze.sbatch`. The smoke array
selects only indices 3, 17, 31 and 45. It does not submit jobs by itself.
Generated scripts limit BLAS/OpenMP threads to one and use the requested CPUs
for Rayon. Each task selects its local SQLite path explicitly.

Inspect the scheduler and logs:

```bash
squeue -j "$job_id"
sacct -j "$job_id" --format=JobID,State,ExitCode,ElapsedRaw,NodeList -P
cat local_runs/paper-smoke/validation.json
```

The analysis job requires every selected result and validates SQLite integrity,
method/geometry/mode, history continuity, energy monotonicity, stopping flags
and the four 1.6 Å reference energies. Smoke analysis performs validation only;
CSV export and plotting require the full scan.
An `afterok` dependency prevents analysis after failed array tasks.

## 3. Run all 56 calculations

Use another fresh workspace:

```bash
"$PYO3_PYTHON" scripts/prepare_paper_slurm.py local_runs/paper-full \
  --scope full --python "$PYO3_PYTHON" \
  --cpus 1 --concurrency 16 --partition defq
job_id=$(sbatch --parsable local_runs/paper-full/run.sbatch)
sbatch --dependency="afterok:$job_id" local_runs/paper-full/analyze.sbatch
```

Adjust concurrency to your allocation and site policy. The tested full scan
used one CPU per task. Stretched UCCSD points can take substantially longer
than the other methods; inspect task-specific elapsed times instead of assuming
a uniform duration. The supplied two-hour task limit was sufficient for the
recorded scan, not a portable runtime guarantee.

Each task has a fresh-process/fresh-database lock. After a failed attempt,
investigate the failure and prepare a new workspace; the scripts do not silently
overwrite or resume a previous database. Preserve logs, actual configs,
`manifest.json`, `provenance/`, and scheduler accounting with each calculation.
The source snapshot includes uncommitted files, with file hashes, so its
contents are not inferred from a commit ID alone.

## 4. Inspect tables and figures

The full analysis job runs a single pipeline: **SQLite → CSV → figures**.
It first validates all 56 databases, exports portable tables while recomputing
RHF/FCI references, and then plots those tables. The exporter opens databases
read-only. The plotter uses only CSV files and Matplotlib.

| Output | Contents |
|---|---|
| `validation.json` | Per-task validation and numerical checkpoints at 1.6 Å |
| `tables/energies.csv` | Endpoint energies, RHF/FCI references and stopping information |
| `tables/macro_iterations.csv` | Unique energy histories, including recomputed RHF iteration zero |
| `tables/provenance.json` | Actual source, executable, input and result hashes |
| `figures/lih_pes.png`, `figures/lih_pes.pdf` | PES and absolute FCI energy errors for UCCD, UCCSD and sequential OO-UCCD |
| `figures/lih_R016_convergence.png`, `figures/lih_R016_convergence.pdf` | FCI energy errors for all four methods at 1.6 Å, in one panel |
| `figures/provenance.json` | Input-table, plotting-script and output hashes, plus software versions |

The figures retain the original paper's layout, colors, line styles, markers,
legend order, fonts and axis labels. The PES has two panels with height ratio
1.8:1; convergence at 1.6 Å uses a single panel including RHF iteration zero.
Logarithmic error axes show `E - E_FCI` without a display floor or absolute-value
transformation. The plotter requires positive finite errors, as in the original
paper's convergence plot. Input table values are unchanged.

The full analysis creates a new `tables/` directory and refuses to overwrite an
existing export. If analysis fails, inspect its log before retrying. Source
copies of the exporter, checker and plotter are saved in `provenance/` so the
analysis does not depend on later edits to repository scripts.

## 5. Manual validation, export and plotting

For a recheck that preserves an existing validation report:

```bash
"$PYO3_PYTHON" scripts/check_paper_slurm.py local_runs/paper-full \
  --report local_runs/paper-full/recheck.json
```

To export a completed full scan again, choose a new directory:

```bash
"$PYO3_PYTHON" scripts/export_paper_results.py local_runs/paper-full \
  --out-dir local_runs/paper-export
"$PYO3_PYTHON" scripts/plot_paper_results.py local_runs/paper-export \
  --out-dir local_runs/paper-figures
```

The exporter requires all 56 tasks; a four-point smoke scan is not a complete
input. Keep the raw databases, inputs, manifest and source/environment records
alongside exported tables. Optional `--accounting-file` adds validated scheduler
records; see [the dataset description](../results/paper/README.md).

To plot the included validated tables without running VQE:

```bash
"$PYO3_PYTHON" scripts/plot_paper_results.py results/paper \
  --out-dir local_runs/included-paper-figures
```

The recorded verification and its limits are documented in
[verification.md](verification.md). Reproduction uses the public scan inputs,
tables and numerical checkpoints; no private historical archive is required.

## Figures included in the manuscript

The manuscript assets in `docs/figures/` are generated from the same included
56-task local-rerun dataset used by the public numerical tables. To regenerate
them with the pinned Python environment:

```bash
"$PYO3_PYTHON" scripts/plot_paper_results.py results/paper --out-dir docs/figures
```

This replaces the figure assets and their `provenance.json`. Inspect the figures
and their hashes before using a new dataset in the manuscript. The arXiv v1
record is unchanged; see [manuscript version details](manuscript-license.md).
