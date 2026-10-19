"""Small OO smoke/regression runs; no paper-scale seed or shot sweep."""
import argparse
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sysconfig
import time


def run(binary, config, directory, name, environment):
    config_path = directory / f"{name}.json"
    db_path = directory / f"{name}.db"
    if db_path.exists():
        raise RuntimeError(f"Refusing to overwrite {db_path}; choose a fresh --out-dir")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    started = time.perf_counter()
    with (directory / f"{name}.log").open("w") as log:
        subprocess.run(
            [str(binary), "--config-filepath", str(config_path), "--db-filepath", str(db_path)],
            env=environment, stdout=log, stderr=log, check=True,
        )
    with sqlite3.connect(db_path) as connection:
        energies = [row[0] for row in connection.execute(
            "SELECT energy FROM macro_iterations ORDER BY rowid"
        )]
        completed = connection.execute("SELECT is_completed FROM runs").fetchone()[0]
    if not completed or not energies or not all(math.isfinite(e) for e in energies):
        raise RuntimeError(f"Incomplete or non-finite result: {name}")
    return {"energies": energies, "seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/oouccd-rs"))
    parser.add_argument("--baseline", type=Path, help="Optional pre-newton-oo binary for seq/L-BFGS regression")
    parser.add_argument("--out-dir", type=Path, default=Path("local_runs") / f"newton_smoke_{time.time_ns()}")
    parser.add_argument("--molecules", nargs="+", choices=["h2", "lih"], default=["h2", "lih"])
    parser.add_argument("--modes", nargs="+", choices=["statevector", "finite_shot"], default=["statevector", "finite_shot"])
    parser.add_argument("--solvers", nargs="+", choices=["sequential", "lbfgs", "newton_raphson", "newton_raw"], default=["sequential", "lbfgs", "newton_raphson"])
    parser.add_argument("--macro-iterations", type=int, default=2)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--regression-tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    if args.macro_iterations < 1 or args.shots < 1:
        parser.error("macro-iterations and shots must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "1"
    environment.setdefault("RAYON_NUM_THREADS", "4")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        sysconfig.get_paths()["purelib"], environment.get("PYTHONPATH"),
    )))
    geometries = {"h2": "H 0 0 0; H 0 0 0.735", "lih": "Li 0 0 0; H 0 0 1.6"}
    results = {}
    for molecule in args.molecules:
        for mode in args.modes:
            for solver in args.solvers:
                name = f"{molecule}_{mode}_{solver}"
                orbital = {"type": solver}
                if solver == "lbfgs":
                    orbital["max_iter"] = 20
                elif solver == "newton_raw":
                    orbital = {"type": "newton_raphson", "stabilized": False}
                rdm = {"type": mode}
                if mode == "finite_shot":
                    rdm.update(shots=args.shots, seed=args.seed)
                config = {
                    "task_name": name,
                    "chemistry": {"geometry": geometries[molecule], "basis": "sto-3g"},
                    "method": {
                        "label": name, "max_macro_iterations": args.macro_iterations,
                        "energy_tolerance": -1.0, "rdm_evaluation": rdm,
                        "ucc_optimizer": {"ansatz_config": {"type": "UCCD"}, "solver_config": {"type": "sequential"}},
                        "orbital_optimizer": {"solver_config": orbital},
                    },
                }
                result = run(args.binary.resolve(), config, args.out_dir, name, environment)
                if args.baseline and solver in ("sequential", "lbfgs"):
                    reference = run(args.baseline.resolve(), config, args.out_dir, name + "_baseline", environment)
                    if len(result["energies"]) != len(reference["energies"]):
                        raise RuntimeError(f"Iteration count differs: {name}")
                    difference = max(abs(a-b) for a,b in zip(result["energies"], reference["energies"]))
                    result.update(baseline=reference, max_energy_difference=difference)
                    if difference > args.regression_tolerance:
                        raise RuntimeError(f"Regression mismatch: {name}: {difference} Ha")
                results[name] = result
                (args.out_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
                print(f"{name}: {result['energies'][-1]:.12f} Ha ({result['seconds']:.2f}s)", flush=True)
    print(f"Results: {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
