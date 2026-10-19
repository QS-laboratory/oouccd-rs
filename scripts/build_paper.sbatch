#!/usr/bin/env bash
# Submit from the repository root after installing the dedicated environment:
#   sbatch scripts/build_paper.sbatch
#SBATCH --job-name=lih-paper-build
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${1:-$SLURM_SUBMIT_DIR}"
export CARGO_HOME="$PWD/.paper-env/cargo"
export RUSTUP_HOME="$PWD/.paper-env/rustup"
export PATH="$CARGO_HOME/bin:$PATH"
export PYO3_PYTHON="$PWD/.paper-env/venv/bin/python"
export PYTHONPATH="$("$PYO3_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# argo compute nodes have libpython3.12.so.1.0 but lack the development
# libpython3.12.so symlink. Supply only that linker name in our private tree.
"$PYO3_PYTHON" - <<'PY'
from pathlib import Path
import sysconfig
native = Path('.paper-env/lib')
native.mkdir(exist_ok=True)
runtime = (Path(sysconfig.get_config_var('LIBDIR')) / sysconfig.get_config_var('INSTSONAME')).resolve(strict=True)
link = native / sysconfig.get_config_var('LDLIBRARY')
if not link.exists():
    link.symlink_to(runtime)
print(f'Private Python linker name: {link} -> {runtime}')
PY
export LIBRARY_PATH="$PWD/.paper-env/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
hostname
rustc --version
cargo --version
"$PYO3_PYTHON" -c 'import sys,numpy,scipy,pyscf; print(sys.version); print(numpy.__version__,scipy.__version__,pyscf.__version__)'
srun --cpu-bind=cores cargo build --release --locked -j "$SLURM_CPUS_PER_TASK"
ldd target/release/oouccd-rs
