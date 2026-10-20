#!/usr/bin/env bash
# Submit after prepare, with --array=0-47%16 and workspace log paths.
#   sbatch --array=0-47%16 scripts/evaluate_lih_shot_noise.sbatch /absolute/workspace
#SBATCH --job-name=lih-shot-noise
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:30:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
WORK="$(realpath "${1:?Pass the prepared experiment workspace}")"
PY="$PWD/.paper-env/venv/bin/python"
srun --cpu-bind=cores "$PY" "$WORK/source/scripts/evaluate_lih_shot_noise.py" \
    task --work "$WORK" --index "${SLURM_ARRAY_TASK_ID:?Use an array submission}"
