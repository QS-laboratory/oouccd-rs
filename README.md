# oouccd-rs: Newton and shot-noise research

This is the `newton-shot-noise` research branch of **oouccd-rs**. For the
paper's baseline implementation, public numerical tables and verified Slurm
reproduction workflow, use the [main branch](https://github.com/QS-laboratory/oouccd-rs/tree/main).
The accompanying paper is Hiromichi Hayashi and Takeshi Sato,
*Sequential optimization for orbital-optimized variational quantum eigensolvers*,
[arXiv:2610.16032](https://arxiv.org/abs/2610.16032).

## Research scope

This branch adds shared exact/finite-shot Pauli RDM construction, analytic
orbital gradients and Hessians, stabilized or raw Newton updates, constrained
RDM reconstruction, covariance propagation and uncertainty-aware orbital
proposals. Sequential and L-BFGS orbital solvers remain available.

The measurement model uses independent counts for each nonidentity Pauli word.
One estimate is reused in all RDM entries containing that word. Consequently,
the nine classical Fourier energy evaluations from one RDM are correlated.
The uncertainty calculations retain this shared-count structure. Hardware
noise and grouped measurement acquisition are outside that model.

Research protocols and archived development results should be interpreted
with the endpoints and limitations recorded in their guides; they do not
establish a general advantage of one optimizer.

## Build and run

A Rust toolchain, C/C++ compiler, CMake, OpenSSL development libraries and a
linkable Python installation are required. Start with the pinned core Python
packages and build the simulator:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export PYO3_PYTHON="$PWD/.venv/bin/python"
export PYTHONPATH="$($PYO3_PYTHON -c 'import sysconfig; print(sysconfig.get_path("purelib"))')${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
cargo build --release --locked
```

The Python runtime library must also be on the platform's linker and runtime
search paths. See the main branch's environment helper for a portable Slurm
setup. Use a fresh database path for each run:

```bash
mkdir -p local_runs
unset GCS_PATH BATCH_TASK_INDEX DB_FILEPATH
target/release/oouccd-rs --config-filepath examples/config_lih_newton.json \
  --db-filepath local_runs/lih-newton.db
```

- [Analytic Newton OO and exact/finite-shot RDMs](docs/newton_oo.md).
- [RDM reconstruction, uncertainty and research protocol](docs/uncertainty_oo.md).
- [Shot-noise evaluation and scoring](docs/lih_shot_noise_evaluation.md).

Additional analysis dependencies are pinned in `scripts/requirements_uncertainty.txt`.
Read the research guide before installing them or submitting study arrays.
Raw databases, logs, environments and workspaces belong in ignored local
folders. Build products and downloaded article PDFs are not source assets.

## Manuscript, citation and licenses

The included [author manuscript](docs/main_rev3.tex) corrects the excitation-gate
order description relative to arXiv v1. The [version and license notice](docs/manuscript-license.md)
identifies that change. Cite the preprint using [CITATION.cff](CITATION.cff).

Original software and software documentation are under the [MIT License](LICENSE).
The manuscript `docs/main_rev3.tex` and figure assets in `docs/figures/` retain
[CC BY 4.0](docs/manuscript-license.md). Dependencies retain their own licenses.
[History and provenance](docs/history-rewrite.md) explains the pre-publication
history cleanup and preservation of original run identifiers.
