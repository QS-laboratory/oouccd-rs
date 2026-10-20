# oouccd-rs

Rust implementation of sequential orbital optimization for variational quantum
eigensolvers (VQE), accompanying [*Sequential optimization for orbital-optimized
variational quantum eigensolvers*](https://arxiv.org/abs/2610.16032) by
Hiromichi Hayashi and Takeshi Sato.

## Code

The implementation combines PySCF molecular integrals with a roqoqo/QuEST
quantum-circuit simulator. It optimizes unitary coupled-cluster amplitudes and,
optionally, molecular orbitals. Sequential optimization reconstructs the finite
Fourier energy profile for each parameter and determines its minimizing angle.

| Method | Circuit ansatz and amplitude updates | Orbital updates |
|---|---|---|
| `UCCD(seq)` | UCCD, sequential sweeps | Fixed RHF orbitals |
| `UCCSD(seq)` | UCCSD, sequential sweeps | Fixed RHF orbitals |
| `OO(seq)-UCCD(seq)` | UCCD, sequential sweeps | Sequential Givens rotations |
| `OO(lbfgs)-UCCD(seq)` | UCCD, sequential sweeps | L-BFGS with finite-difference gradients |

Each orbital-optimized macro iteration performs one orbital update followed by
one amplitude sweep. Sequential OO visits each occupied–virtual orbital pair
once per macro iteration. Its nine energy evaluations per pair use the same
fixed reduced density matrices (RDMs), so these orbital trial evaluations
require only classical integral transformations and contractions.

The paper calculations use exact statevectors. An ideal finite-shot mode is
also available; see [configuration and outputs](docs/configuration.md) for its
measurement model, task JSON, units and SQLite format. Newton OO and subsequent
shot-noise research are on the `newton-shot-noise` branch.

The library and CLI are in `src/`, task inputs and API examples in `examples/`,
and reproduction tools in `scripts/`. For a small H₂ calculation, use
[`examples/config_h2_manual.json`](examples/config_h2_manual.json) or the
[Rust API example](examples/check_rust_app.rs).

## Paper

Hiromichi Hayashi and Takeshi Sato,
**Sequential optimization for orbital-optimized variational quantum eigensolvers**,
[arXiv:2610.16032](https://arxiv.org/abs/2610.16032) (2026).

The paper introduces sequential orbital-optimized UCCD (SOO-UCCD) and compares
its energies and macro-iteration convergence with UCCD, UCCSD and L-BFGS-based
orbital optimization for LiH in the STO-3G basis.

![LiH/STO-3G potential-energy curves](docs/figures/lih_pes.png)

Citation metadata is available in [CITATION.cff](CITATION.cff). The included
[author manuscript](docs/main_rev3.tex) corrects the excitation-gate order
description relative to arXiv v1 and uses figures regenerated from the included
local-rerun data. See [manuscript version and license](docs/manuscript-license.md)
and [figure provenance](docs/figures/provenance.json).

## Reproduce the paper data

The tracked [scan template](examples/lih_paper_scan.json) defines **56
calculations**: four methods at fourteen Li–H distances from 1.0 to 3.6 Å in
0.2 Å steps. The protocol uses four electrons, six spatial orbitals, no frozen
core, exact statevectors, zero initial amplitudes, a `1e-8` Ha energy-change
tolerance and at most 100 macro iterations.

### 1. Install and build

Use a checkout of `main`. A Rust toolchain, Python 3.12, a C/C++ compiler,
CMake, `pkg-config`, OpenSSL development libraries and a linkable Python
runtime are required. The [verification report](docs/verification.md) records
the tested software versions.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
source scripts/python_env.sh
cargo build --release --locked
```

To use an existing Python environment, source
`scripts/python_env.sh /absolute/path/to/venv/bin/python` instead. If your
cluster requires compilation in a Slurm allocation, submit
`sbatch --partition=defq scripts/build_paper.sbatch` and wait for a successful
build before preparing the calculations. Replace `defq` with your CPU partition.

### 2. Run the full Slurm scan

From the repository root, with the Python environment above selected:

```bash
"$PYO3_PYTHON" scripts/prepare_paper_slurm.py local_runs/paper-full \
  --scope full --python "$PYO3_PYTHON" \
  --cpus 1 --concurrency 16 --partition defq
job_id=$(sbatch --parsable local_runs/paper-full/run.sbatch)
sbatch --dependency="afterok:$job_id" local_runs/paper-full/analyze.sbatch
```

Replace `defq` with your CPU partition and adjust concurrency to your allocation.
The preparer creates inputs, a copied executable, source/environment records
and job scripts; the two `sbatch` commands submit the calculations and dependent
analysis. Use a **new workspace for each attempt**. Existing databases are not
optimizer checkpoints.

For a four-method check at 1.6 Å before the full scan, use `--scope smoke` with
a different workspace. The [full reproduction guide](docs/reproducing_paper_results.md)
explains resource settings, scheduler checks and individual task outputs.

### 3. Inspect the results

The full analysis job validates the databases, exports CSV tables and plots
those tables automatically. After it finishes successfully, the workspace contains:

| Path | Contents |
|---|---|
| `validation.json` | Integrity, stopping-rule and 1.6 Å checkpoint checks |
| `tables/energies.csv` | Final energies and RHF/FCI references |
| `tables/macro_iterations.csv` | Unique macro histories, including RHF iteration zero |
| `tables/provenance.json` | Source, executable and input/result hashes |
| `figures/lih_pes.png`, `.pdf` | Potential-energy curves and errors relative to FCI |
| `figures/lih_R016_convergence.png`, `.pdf` | FCI energy errors at 1.6 Å, in the original single-panel style |
| `figures/provenance.json` | Figure inputs, plotting code, software versions and output hashes |

All plotting follows **SQLite → CSV → figures**. The smoke analysis checks
its four calculations without exporting the full dataset. See the
[reproduction guide](docs/reproducing_paper_results.md) for manual checks,
re-exporting and replotting. Keep the raw databases and run records.

### Inspect the included data

The repository already includes [validated numerical tables](results/paper/)
from a completed 56-task Slurm rerun. To plot them without rerunning VQE,
use the following command with a Python environment containing Matplotlib:

```bash
python scripts/plot_paper_results.py results/paper \
  --out-dir local_runs/included-paper-figures
```

In the recorded rerun, 53 points converged and three stretched-bond UCCSD
points reached the iteration cap. All 51 comparable historical endpoints
agreed within `1e-8` Ha; five historical runs had shorter iteration budgets.
See the [verification report](docs/verification.md) for the numerical checks
and limits of reproduction.

Original software, software documentation and numerical tables use the
[MIT License](LICENSE). The manuscript and scientific figures use
[CC BY 4.0](docs/manuscript-license.md); dependencies retain their own licenses.
