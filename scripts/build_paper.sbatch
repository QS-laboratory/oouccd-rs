#!/usr/bin/env bash
# Submit from the repository root with Rust on PATH and Python in .venv:
#   sbatch --partition=YOUR_CPU_PARTITION scripts/build_paper.sbatch
#SBATCH --job-name=lih-paper-build
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${1:-$SLURM_SUBMIT_DIR}"
source scripts/python_env.sh "${PYO3_PYTHON:-$PWD/.venv/bin/python}"
hostname
rustc --version
cargo --version
"$PYO3_PYTHON" -c 'import sys,numpy,scipy,pyscf; print(sys.version); print(numpy.__version__,scipy.__version__,pyscf.__version__)'
srun --cpu-bind=cores cargo build --release --locked -j "$SLURM_CPUS_PER_TASK"
ldd target/release/oouccd-rs
