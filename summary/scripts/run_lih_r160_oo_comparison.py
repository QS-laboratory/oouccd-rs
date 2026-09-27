"""Run fresh R=1.6 comparisons of sequential and L-BFGS orbital optimization."""

import argparse
import json
import os
import sqlite3
import subprocess
import time


METHODS = {
    "seq": {
        "label": "OO(seq)-UCCD(seq)",
        "orbital_solver": {"type": "sequential"},
    },
    "lbfgs": {
        "label": "OO(lbfgs)-UCCD(seq)",
        "orbital_solver": {"type": "lbfgs", "max_iter": 200},
    },
}


def task_config(method_key, mode, shots, seed, macro_iterations):
    method = METHODS[method_key]
    rdm_evaluation = {"type": "statevector"}
    if mode == "finite_shot":
        rdm_evaluation = {"type": "finite_shot", "shots": shots, "seed": seed}
    return {
        "task_name": f"LiH_R160_{method_key}_{mode}_{seed if seed is not None else 'exact'}",
        "chemistry": {
            "geometry": "Li 0 0 0; H 0 0 1.60000",
            "basis": "sto-3g",
            "charge": 0,
            "spin": 0,
            "freeze_core": 0,
        },
        "method": {
            "label": method["label"],
            "max_macro_iterations": macro_iterations,
            "energy_tolerance": -1.0,
            "rdm_evaluation": rdm_evaluation,
            "ucc_optimizer": {
                "ansatz_config": {"type": "UCCD"},
                "solver_config": {"type": "sequential"},
            },
            "orbital_optimizer": {"solver_config": method["orbital_solver"]},
        },
    }


def load_result(db_path):
    connection = sqlite3.connect(db_path)
    run = connection.execute(
        "SELECT id, final_energy, is_completed FROM runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if run is None:
        connection.close()
        raise RuntimeError(f"No run stored in {db_path}")
    run_id, final_energy, is_completed = run
    rows = connection.execute(
        "SELECT macro_idx, energy FROM macro_iterations WHERE run_id = ? ORDER BY rowid",
        (run_id,),
    ).fetchall()
    connection.close()
    history = {}
    for macro_idx, energy in rows:
        history[macro_idx] = energy
    return {
        "final_energy_ha": final_energy,
        "is_completed": bool(is_completed),
        "iterations": sorted(history),
        "energies_ha": [history[index] for index in sorted(history)],
    }


def run_task(binary, config_path, db_path, environment):
    if os.path.exists(db_path):
        os.remove(db_path)
    started = time.perf_counter()
    completed = subprocess.run(
        [binary, "--config-filepath", config_path, "--db-filepath", db_path],
        env=environment,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Run failed ({config_path}):\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    result = load_result(db_path)
    result["elapsed_seconds"] = elapsed
    result["stdout"] = completed.stdout
    result["stderr"] = completed.stderr
    return result


def save_summary(path, results):
    summary_dir = os.path.dirname(path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


parser = argparse.ArgumentParser(description="Run fresh LiH R=1.6 OO optimizer comparison")
parser.add_argument("--binary", default="target/debug/oouccd-rs")
parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "results", "lih_r160_oo_comparison"))
parser.add_argument("--summary", default=os.path.join(os.path.dirname(__file__), "..", "results", "lih_r160_oo_comparison.json"))
parser.add_argument("--shots", type=int, default=10000)
parser.add_argument("--seeds", type=int, nargs="*", default=[20260910, 20260911, 20260912, 20260913, 20260914])
parser.add_argument("--macro-iterations", type=int, default=7)
parser.add_argument("--skip-statevector", action="store_true")
parser.add_argument("--skip-finite-shot", action="store_true")
parser.add_argument("--resume", action="store_true", help="Keep completed runs and continue missing runs")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
environment = os.environ.copy()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
venv_lib = os.path.join(project_root, "seq-venv", "lib")
if os.path.isdir(venv_lib):
    python_dirs = sorted(
        entry for entry in os.listdir(venv_lib)
        if entry.startswith("python") and os.path.isdir(os.path.join(venv_lib, entry, "site-packages"))
    )
    if python_dirs:
        site_packages = os.path.join(venv_lib, python_dirs[-1], "site-packages")
        current_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            site_packages if not current_pythonpath
            else os.pathsep.join([site_packages, current_pythonpath])
        )
if args.resume and os.path.exists(args.summary):
    with open(args.summary, encoding="utf-8") as handle:
        results = json.load(handle)
else:
    results = {
        "bond_length_angstrom": 1.6,
        "basis": "sto-3g",
        "shots": args.shots,
        "seeds": args.seeds,
        "macro_iterations": args.macro_iterations,
        "statevector": {},
        "finite_shot": {method_key: [] for method_key in METHODS},
    }

if not args.skip_statevector:
    for method_key in METHODS:
        if args.resume and method_key in results["statevector"]:
            continue
        config = task_config(method_key, "statevector", args.shots, None, args.macro_iterations)
        config_path = os.path.join(args.out_dir, f"{method_key}_statevector.json")
        db_path = os.path.join(args.out_dir, f"{method_key}_statevector.db")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")
        print(f"Running {method_key} statevector ...", flush=True)
        results["statevector"][method_key] = run_task(args.binary, config_path, db_path, environment)
        save_summary(args.summary, results)

if not args.skip_finite_shot:
    for seed in args.seeds:
        for method_key in METHODS:
            results["finite_shot"].setdefault(method_key, [])
            completed_seeds = {run["seed"] for run in results["finite_shot"][method_key]}
            if args.resume and seed in completed_seeds:
                continue
            config = task_config(method_key, "finite_shot", args.shots, seed, args.macro_iterations)
            config_path = os.path.join(args.out_dir, f"{method_key}_finite_shot_{seed}.json")
            db_path = os.path.join(args.out_dir, f"{method_key}_finite_shot_{seed}.db")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)
                handle.write("\n")
            print(f"Running {method_key} finite-shot seed={seed} ...", flush=True)
            run = run_task(args.binary, config_path, db_path, environment)
            run["seed"] = seed
            results["finite_shot"][method_key].append(run)
            save_summary(args.summary, results)

save_summary(args.summary, results)
print(f"Saved: {os.path.abspath(args.summary)}")