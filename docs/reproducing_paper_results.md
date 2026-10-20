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
for Rayon. Cloud-output environment variables are unset.

Inspect the scheduler and logs:

```bash
squeue -j "$job_id"
sacct -j "$job_id" --format=JobID,State,ExitCode,ElapsedRaw,NodeList -P
cat local_runs/paper-smoke/validation.json
```

The analysis job requires every selected result and validates SQLite integrity,
method/geometry/mode, history continuity, energy monotonicity, stopping flags
and the four 1.6 Å reference energies. It also plots convergence histories.
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

## 4. Validate and plot

Checks can be run independently of the analysis job:

```bash
"$PYO3_PYTHON" scripts/check_paper_slurm.py local_runs/paper-full
```

Use `--report another/path.json` for a read-only recheck that leaves the old
validation report intact. Checks use read-only SQLite connections.

The existing database plotters are:

```bash
MPLBACKEND=Agg "$PYO3_PYTHON" summary/scripts/plot_lih_pes.py \
  --results-dir local_runs/paper-full/results \
  --output local_runs/paper-full/figures/lih_pes.png
MPLBACKEND=Agg "$PYO3_PYTHON" summary/scripts/plot_convergence.py \
  --include-fixed-orbitals --pdf \
  --results-dir local_runs/paper-full/results \
  --out-dir local_runs/paper-full/figures/convergence
```

They recompute RHF/FCI references with PySCF. The PES displays UCCD, UCCSD and
sequential OO-UCCD. Convergence at 1.6 Å comes from the same scan; a separate
simulation is unnecessary. Duplicate final macro rows at the iteration cap
must be collapsed when processing histories.

To inspect the published compact tables without running quantum simulations:

```bash
"$PYO3_PYTHON" scripts/plot_paper_results.py results/paper \
  --out-dir local_runs/paper-table-figures
```

## 5. Comparison with historical results

The repository contains compact numerical evidence from the verified Slurm
scan under [results/paper](../results/paper/), including endpoint energies,
unique macro histories, scheduler outcomes and sanitized historical comparisons.
Original databases and environment snapshots are retained locally but are not
required to view these tables.

If the historical SQLite/config archive is available independently, compare it
with a full rerun using:

```bash
"$PYO3_PYTHON" scripts/compare_paper_results.py \
  --archive-root /path/to/historical-archive \
  --rerun-workspace local_runs/paper-full \
  --out-dir local_runs/paper-comparison
```

This script does not download data or alter cloud permissions. It checks all
56 physical inputs, separates same-endpoint comparisons from extended runs,
and reports intermediate-history deviations. Five recovered UCCSD histories
end at 30 macros, while the paper protocol allows 100; their later historical
replacement histories were not recovered. Do not describe the available
comparison as complete agreement of all 56 historical final energies.

Numerical reproduction concerns energies, settings and stopping behavior.
Neither bitwise-identical intermediate trajectories nor pixel-identical
historical PDF rendering is established.
