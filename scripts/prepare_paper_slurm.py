#!/usr/bin/env python3
"""Prepare (without submitting) isolated LiH paper jobs for Slurm.

Inputs are reconstructed from docs/reproducing_paper_results.md, src/config.rs,
and config_lih_pes_lbfgs_only.json; the ignored historical master is unavailable.
Smoke: all four methods at 1.60 A. Full: four methods x fourteen geometries.
Use a new workspace for every attempt; an existing database is never resumed.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--scope", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--cpus", type=int, default=1, help="CPUs per task (default: 1; argo scaling benchmark)")
    parser.add_argument("--concurrency", type=int, default=4, help="Maximum simultaneous array elements")
    parser.add_argument("--python", type=Path, default=ROOT / ".paper-env/venv/bin/python")
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/oouccd-rs")
    args = parser.parse_args()
    if args.cpus < 1 or args.concurrency < 1:
        parser.error("--cpus and --concurrency must be positive")
    binary = args.binary.resolve(strict=True)
    # Keep the venv path: resolving its symlink would select the system Python.
    python = args.python.absolute()
    subprocess.run([str(python), "-c", "import numpy, scipy, pyscf, matplotlib"], check=True)
    work = args.workspace.resolve()
    work.mkdir(parents=True, exist_ok=False)
    for name in ("configs", "results", "logs", "bin", "provenance"):
        (work / name).mkdir()
    shutil.copy2(binary, work / "bin/oouccd-rs")
    shutil.copy2(Path(__file__), work / "provenance/prepare_paper_slurm.py")
    shutil.copy2(ROOT / "scripts/check_paper_slurm.py", work / "provenance/check_paper_slurm.py")
    shutil.copy2(ROOT / "scripts/build_paper.sbatch", work / "provenance/build_paper.sbatch")
    for name in ("plot_convergence.py", "plot_lih_pes.py"):
        shutil.copy2(ROOT / "summary/scripts" / name, work / "provenance" / name)
    for name in ("Cargo.lock", "Cargo.toml", "requirements.txt",
                 "config_lih_pes_lbfgs_only.json", "docs/reproducing_paper_results.md",
                 "docs/main_rev3.tex"):
        target = work / "provenance" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    versions = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    (work / "provenance/python-freeze.txt").write_text(versions)
    with (work / "provenance/source.patch").open("w") as out:
        subprocess.run(["git", "diff", "HEAD"], cwd=ROOT, stdout=out, check=True)
    with (work / "provenance/source.tar").open("wb") as out:
        subprocess.run(["git", "archive", "HEAD"], cwd=ROOT, stdout=out, check=True)

    master = json.loads((ROOT / "config_lih_pes_lbfgs_only.json").read_text())
    base = copy.deepcopy(master["methods"][0])
    base.update(max_macro_iterations=100, energy_tolerance=1e-8,
                rdm_evaluation={"type": "statevector"})
    methods = []
    for label, ansatz, orbital in (
        ("UCCD(seq)", "UCCD", None),
        ("UCCSD(seq)", "UCCSD", None),
        ("OO(seq)-UCCD(seq)", "UCCD", {"type": "sequential"}),
        ("OO(lbfgs)-UCCD(seq)", "UCCD", {"type": "lbfgs", "max_iter": 200}),
    ):
        method = copy.deepcopy(base)
        method["label"] = label
        method["ucc_optimizer"]["ansatz_config"]["type"] = ansatz
        method["orbital_optimizer"] = {"solver_config": orbital} if orbital else None
        methods.append(method)
    indices = []
    for rank, method in enumerate(methods):
        for point in range(14):
            index = 14 * rank + point
            chemistry = copy.deepcopy(master["chemistry"])
            chemistry["geometry"] = chemistry["geometry"].replace("{R}", f"{1 + .2 * point:.5f}")
            task = {"task_name": "PaperReproduction_" + method["label"],
                    "chemistry": chemistry, "method": method}
            (work / "configs" / f"task_{index:03d}.json").write_text(json.dumps(task, indent=2) + "\n")
            if args.scope == "full" or point == 3:
                indices.append(index)
    manifest = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "scope": args.scope, "task_indices": indices, "python": str(python),
        "cpus_per_task": args.cpus, "concurrency": args.concurrency,
        "input_provenance": "Reconstructed from tracked L-BFGS master, manuscript guide, and config schema; not recovered historical JSON",
        "note": "Python environment must remain at this path and unchanged while jobs run. Binary is copied; source.tar plus source.patch record the checkout.",
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    script = f'''#!/usr/bin/env bash
#SBATCH --job-name=lih-paper-{args.scope}
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array={','.join(map(str, indices))}%{args.concurrency}
#SBATCH --output={shlex.quote(str(work / 'logs/%A_%a.out'))}
#SBATCH --error={shlex.quote(str(work / 'logs/%A_%a.err'))}
set -euo pipefail
WORK={shlex.quote(str(work))}
PY={shlex.quote(str(python))}
unset GCS_PATH BATCH_TASK_INDEX DB_FILEPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export RAYON_NUM_THREADS="$SLURM_CPUS_PER_TASK"
printf -v task 'task_%03d' "$SLURM_ARRAY_TASK_ID"
# This lock remains even after failure: retry in a fresh workspace.
mkdir "$WORK/results/$task.lock"
test ! -e "$WORK/results/$task.db"
hostname
date --iso-8601=seconds
"$PY" -c 'import sys,numpy,scipy,pyscf; print(sys.version); print(numpy.__version__,scipy.__version__,pyscf.__version__)'
srun --cpu-bind=cores /usr/bin/time -v "$WORK/bin/oouccd-rs" \\
  --config-filepath "$WORK/configs/$task.json" \\
  --db-filepath "$WORK/results/$task.db"
'''
    (work / "run.sbatch").write_text(script)
    analysis = f'''#!/usr/bin/env bash
#SBATCH --job-name=lih-paper-check
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output={shlex.quote(str(work / 'logs/check-%j.out'))}
#SBATCH --error={shlex.quote(str(work / 'logs/check-%j.err'))}
set -euo pipefail
WORK={shlex.quote(str(work))}
PY={shlex.quote(str(python))}
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
"$PY" "$WORK/provenance/check_paper_slurm.py" "$WORK"
mkdir -p "$WORK/figures"
"$PY" "$WORK/provenance/plot_convergence.py" --include-fixed-orbitals --pdf --results-dir "$WORK/results" --out-dir "$WORK/figures/convergence"
'''
    if args.scope == "full":
        analysis += '"$PY" "$WORK/provenance/plot_lih_pes.py" --results-dir "$WORK/results" --output "$WORK/figures/lih_pes.png"\n'
    (work / "analyze.sbatch").write_text(analysis)
    print(f"Prepared {args.scope}: {len(indices)} tasks in {work}")
    print(shlex.join(["sbatch", "--test-only", str(work / "run.sbatch")]))
    print(shlex.join(["sbatch", "--parsable", str(work / "run.sbatch")]))
    print(shlex.join(["sbatch", "--dependency=afterok:JOB_ID", str(work / "analyze.sbatch")]))


if __name__ == "__main__":
    main()
