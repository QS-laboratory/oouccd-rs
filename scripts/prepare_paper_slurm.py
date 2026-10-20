#!/usr/bin/env python3
"""Prepare (without submitting) isolated LiH paper jobs for Slurm.

Inputs come from the tracked examples/lih_paper_scan.json template.
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
import re
import tarfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--scope", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--cpus", type=int, default=1, help="CPUs per task (default: 1; argo scaling benchmark)")
    parser.add_argument("--concurrency", type=int, default=4, help="Maximum simultaneous array elements")
    parser.add_argument("--partition", default="defq", help="Slurm CPU partition (default: defq)")
    parser.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/oouccd-rs")
    args = parser.parse_args()
    if args.cpus < 1 or args.concurrency < 1:
        parser.error("--cpus and --concurrency must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.partition):
        parser.error("--partition must be a single Slurm partition name")
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
    for name in ("export_paper_results.py", "plot_paper_results.py"):
        shutil.copy2(ROOT / "scripts" / name, work / "provenance" / name)
    for name in ("Cargo.lock", "Cargo.toml", "requirements.txt",
                 "examples/lih_paper_scan.json", "docs/reproducing_paper_results.md",
                 "docs/main_rev3.tex"):
        target = work / "provenance" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    versions = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    (work / "provenance/python-freeze.txt").write_text(versions)
    with (work / "provenance/source.patch").open("w") as out:
        subprocess.run(["git", "diff", "HEAD"], cwd=ROOT, stdout=out, check=True)
    paths = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT).split(b"\0")
    source_hashes = {}
    with tarfile.open(work / "provenance/source.tar", "w") as archive:
        for raw in sorted(set(p for p in paths if p)):
            name = raw.decode(); path = ROOT / name
            if path == work or work in path.parents:
                continue
            if path.is_file():
                archive.add(path, arcname=name, recursive=False)
                source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    master = json.loads((ROOT / "examples/lih_paper_scan.json").read_text())
    methods = master["methods"]
    if len(methods) != 4 or len(master['distances_angstrom']) != 14:
        raise ValueError('The paper protocol requires four methods and fourteen distances')
    indices = []
    for rank, method in enumerate(methods):
        for point, distance in enumerate(master["distances_angstrom"]):
            index = 14 * rank + point
            chemistry = copy.deepcopy(master["chemistry"])
            chemistry["geometry"] = chemistry["geometry"].replace("{R}", f"{distance:.5f}")
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
        "partition": args.partition,
        "input_provenance": "Tracked examples/lih_paper_scan.json; reconstructed manuscript protocol, not historical launch JSONs",
        "source_files_sha256": source_hashes,
        "source_archive_sha256": hashlib.sha256((work / "provenance/source.tar").read_bytes()).hexdigest(),
        "note": "Python must remain available at this path on compute nodes. Binary and actual working-tree source are copied, including uncommitted files.",
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    script = f'''#!/usr/bin/env bash
#SBATCH --job-name=lih-paper-{args.scope}
#SBATCH --partition={args.partition}
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
unset DB_FILEPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
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
#SBATCH --partition={args.partition}
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
'''
    if args.scope == "full":
        analysis += '''"$PY" "$WORK/provenance/export_paper_results.py" "$WORK" --out-dir "$WORK/tables"
"$PY" "$WORK/provenance/plot_paper_results.py" "$WORK/tables" --out-dir "$WORK/figures"
'''
    (work / "analyze.sbatch").write_text(analysis)
    print(f"Prepared {args.scope}: {len(indices)} tasks in {work}")
    print(shlex.join(["sbatch", "--test-only", str(work / "run.sbatch")]))
    print(shlex.join(["sbatch", "--parsable", str(work / "run.sbatch")]))
    print(shlex.join(["sbatch", "--dependency=afterok:JOB_ID", str(work / "analyze.sbatch")]))


if __name__ == "__main__":
    main()
