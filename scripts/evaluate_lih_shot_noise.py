"""Prepare, run and summarize matched-budget LiH OO shot-noise trajectories."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import sysconfig
import time

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "seq": {"type": "sequential"},
    "newton_raw": {"type": "newton_raphson", "stabilized": False},
    "newton_stable": {"type": "newton_raphson", "stabilized": True},
}
LABELS = {"seq": "OO(seq)", "newton_raw": "OO(Newton, raw)", "newton_stable": "OO(Newton, stabilized)"}


def geometry_label(manifest):
    return f"LiH {manifest.get('bond_length_angstrom', 1.6):g} Å"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if (work / "manifest.json").exists():
        raise RuntimeError("This workspace is already prepared")
    for directory in ("configs", "results", "logs", "bin", "source", "figures"):
        (work / directory).mkdir(exist_ok=True)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    import numpy as np
    import pyscf
    from pyscf import ao2mo, fci, gto, scf
    geometry = f"Li 0 0 0; H 0 0 {args.distance:g}"
    distance_tag = f"r{args.distance:.3f}".replace(".", "p")
    mol = gto.M(atom=geometry, basis="sto-3g", spin=0, charge=0, symmetry=False, verbose=0)
    mf = scf.RHF(mol).run()
    if not mf.converged:
        raise RuntimeError("RHF reference did not converge")
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.incore.full(mol.intor("int2e"), mf.mo_coeff), mol.nao)
    solver = fci.direct_spin0.FCI()
    solver.max_cycle = 200
    electronic, _ = solver.kernel(h1, eri, mol.nao, mol.nelectron)
    if not solver.converged:
        raise RuntimeError("FCI reference did not converge")
    binary = args.binary.resolve()
    copied_binary = work / "bin/evaluate_shot_noise"
    shutil.copy2(binary, copied_binary)
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True,
    ).splitlines()
    source_hashes = {}
    for relative in files:
        if relative.startswith(("src/", "scripts/", "examples/")) or relative in ("Cargo.toml", "Cargo.lock"):
            path = ROOT / relative
            if path.is_file():
                destination = work / "source" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                source_hashes[relative] = sha256(path)
    (work / "source/diff-from-head.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT))
    tasks = []
    for shots in [None, *args.shots]:
        for seed in ([None] if shots is None else args.seeds):
            for method, orbital in METHODS.items():
                index = len(tasks)
                task_name = f"lih_{distance_tag}_{method}_{'exact' if shots is None else f'n{shots}_seed{seed}'}"
                rdm = {"type": "statevector"} if shots is None else {"type": "finite_shot", "shots": shots, "seed": seed}
                config = {
                    "task_name": task_name,
                    "chemistry": {"geometry": geometry, "basis": "sto-3g", "charge": 0, "spin": 0, "freeze_core": 0},
                    "method": {
                        "label": LABELS[method] + "-UCCD(seq)", "max_macro_iterations": args.iterations,
                        "energy_tolerance": -1.0, "rdm_evaluation": rdm,
                        "ucc_optimizer": {"ansatz_config": {"type": "UCCD"}, "solver_config": {"type": "sequential"}},
                        "orbital_optimizer": {"solver_config": orbital},
                    },
                }
                config_path = work / "configs" / f"task_{index:03d}.json"
                write_json(config_path, config)
                tasks.append({"index": index, "name": task_name, "method": method, "shots": shots, "seed": seed,
                              "config": str(config_path), "result": str(work / 'results' / f'task_{index:03d}.json')})
    manifest = {
        "geometry": f"LiH {args.distance:g} Angstrom", "bond_length_angstrom": args.distance,
        "basis": "STO-3G", "spatial_orbitals": int(mol.nao),
        "electrons": int(mol.nelectron), "freeze_core": 0, "fci_energy_ha": float(electronic + mol.energy_nuc()),
        "rhf_energy_ha": float(mf.e_tot), "iterations": args.iterations, "shots": args.shots, "seeds": args.seeds,
        "measurement_model": "Independent finite-shot Pauli words; full real spin-summed 1-/2-RDM; no hardware noise",
        "initialization": "RHF orbitals, zero UCCD amplitudes", "ucc_optimizer": "one sequential five-point sweep per macro iteration",
        "trial_seed_policy": "Disjoint base_seed + evaluation_count windows across trial seeds; matched trial seeds reused across methods and shot counts",
        "newton_updates_per_macro": 1,
        "stabilization": {"hessian_floor": 1e-4, "max_step_norm": 0.2, "gradient_tolerance": 1e-6, "max_backtracks": 20},
        "diagnostics": "Exact statevector energies are recorded after updates and never enter optimization or stopping",
        "python": sys.executable, "numpy": np.__version__, "pyscf": pyscf.__version__,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "binary": str(copied_binary), "binary_sha256": sha256(copied_binary), "source_sha256": source_hashes,
        "cpus_per_task": args.cpus, "tasks": tasks,
    }
    reused = import_results(work, manifest, args.reuse) if args.reuse else []
    manifest["pending_task_indices"] = [item["index"] for item in tasks if "reused_from" not in item]
    write_json(work / "manifest.json", manifest)
    write_json(work / "reused_results.json", reused)
    print(json.dumps({"workspace": str(work), "tasks": len(tasks), "reused": len(reused),
                      "new_tasks": len(manifest["pending_task_indices"]), "fci_energy_ha": manifest["fci_energy_ha"]}, indent=2))


def import_results(work, manifest, previous):
    previous = previous.resolve()
    old = json.loads((previous / "manifest.json").read_text())
    if old["binary_sha256"] != manifest["binary_sha256"]:
        raise ValueError("Cannot reuse results from a different evaluator binary")
    old_tasks = {(item["method"], item["shots"], item["seed"]): item for item in old["tasks"]}
    reused = []
    for item in manifest["tasks"]:
        source = old_tasks.get((item["method"], item["shots"], item["seed"]))
        if source is None:
            continue
        old_config = json.loads(Path(source["config"]).read_text())
        new_config = json.loads(Path(item["config"]).read_text())
        if any(old_config[key] != new_config[key] for key in ("chemistry", "method")):
            continue
        result_path = Path(source["result"])
        execution_path = result_path.with_name(result_path.stem + "_execution.json")
        if not result_path.exists() or not execution_path.exists():
            continue
        result = json.loads(result_path.read_text())
        execution = json.loads(execution_path.read_text())
        if result["status"] != "completed" or execution["exit_code"] != 0:
            continue
        if result["iterations"][-1]["macro_iteration"] != manifest["iterations"]:
            raise ValueError("Reused result does not have the requested budget")
        if abs(old["fci_energy_ha"] - manifest["fci_energy_ha"]) > 1e-10:
            raise ValueError("FCI references differ for a reused configuration")
        destination = Path(item["result"])
        shutil.copy2(result_path, destination)
        shutil.copy2(execution_path, destination.with_name(destination.stem + "_execution.json"))
        provenance = {"source_workspace": str(previous), "source_task_index": source["index"],
                      "source_result": str(result_path), "result_sha256": sha256(result_path),
                      "execution_sha256": sha256(execution_path), "original_cpus_per_task": old["cpus_per_task"]}
        if sha256(destination) != provenance["result_sha256"]:
            raise ValueError("Reused result hash mismatch")
        item["reused_from"] = provenance
        reused.append({"task_index": item["index"], **provenance})
    return reused


def task(args):
    work = args.work.resolve()
    manifest = json.loads((work / "manifest.json").read_text())
    item = manifest["tasks"][args.index]
    result = Path(item["result"])
    if result.exists():
        raise RuntimeError(f"Refusing to overwrite {result}")
    environment = os.environ.copy()
    for variable in ("GCS_PATH", "BATCH_TASK_INDEX", "DB_FILEPATH", "PYTHONHOME"):
        environment.pop(variable, None)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        environment[variable] = "1"
    environment["OMP_DYNAMIC"] = "FALSE"
    environment["RAYON_NUM_THREADS"] = str(manifest["cpus_per_task"])
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = sysconfig.get_paths()["purelib"]
    started = time.perf_counter()
    with (work / "logs" / f"task_{args.index:03d}.log").open("w") as log:
        process = subprocess.run([manifest["binary"], "--config", item["config"], "--output", str(result)],
                                 env=environment, stdout=log, stderr=log)
    execution = {"exit_code": process.returncode, "seconds": time.perf_counter() - started,
                 "hostname": socket.gethostname(), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                 "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"), "affinity": sorted(os.sched_getaffinity(0))}
    write_json(work / "results" / f"task_{args.index:03d}_execution.json", execution)
    if process.returncode:
        print(f"Task {args.index} process failure: {process.returncode}", file=sys.stderr)
        return process.returncode
    output = json.loads(result.read_text())
    print(f"Task {args.index}: {output['status']}")
    return 0


def load_rows(work):
    manifest = json.loads((work / "manifest.json").read_text())
    reference = manifest["fci_energy_ha"]
    rows = []
    for item in manifest["tasks"]:
        path = Path(item["result"])
        execution_path = path.with_name(path.stem + "_execution.json")
        result = json.loads(path.read_text()) if path.exists() else None
        execution = json.loads(execution_path.read_text()) if execution_path.exists() else None
        terminal = result is not None and result["status"] in ("completed", "failed")
        if execution and execution["exit_code"]:
            status = "process_failure"
        elif terminal:
            status = result["status"]
        else:
            status = "pending" if result is None else "running"
        endpoint = result["iterations"][-1] if result and result["iterations"] else None
        complete = status == "completed"
        error = endpoint["exact_energy_ha"] - reference if endpoint else None
        if error is not None and error < -1e-7:
            raise RuntimeError(f"Variational bound violation in task {item['index']}: {error}")
        if complete:
            if endpoint["macro_iteration"] != manifest["iterations"]:
                raise RuntimeError(f"Wrong endpoint in task {item['index']}")
            if endpoint["rdm_evaluations"] != 1 + 383 * manifest["iterations"]:
                raise RuntimeError(f"Unexpected measurement budget in task {item['index']}")
        rows.append(dict(item, status=status, error_ha=error, completed=complete,
                         chemical_accuracy=complete and error <= 1.6e-3,
                         accuracy_0p1_mha=complete and error <= 1e-4,
                         failure_reason=(result or {}).get("error") or (f"Process exit {execution['exit_code']}" if execution and execution["exit_code"] else None),
                         failed_macro_iteration=result.get("failed_macro_iteration") if result else None,
                         trajectory=result, execution=execution))
    return manifest, rows


def summarize(args):
    import numpy as np
    work = args.work.resolve()
    if Path(__file__).resolve() != work / "analysis_source.py":
        shutil.copy2(Path(__file__).resolve(), work / "analysis_source.py")
    manifest, rows = load_rows(work)
    if not args.allow_partial and any(row["status"] in ("pending", "running") for row in rows):
        raise RuntimeError("Some tasks have not finished")
    grouped = defaultdict(list)
    for row in rows:
        if row["shots"] is not None:
            grouped[row["method"], row["shots"]].append(row)
    summary = []
    baseline_errors = {row["method"]: row["error_ha"] for row in rows if row["shots"] is None and row["completed"]}
    for (method, shots), group in sorted(grouped.items(), key=lambda x: (x[0][1], list(METHODS).index(x[0][0]))):
        errors = np.array([row["error_ha"] * 1000 for row in group if row["completed"]])
        updates = [point["oo"] for row in group if row["trajectory"] for point in row["trajectory"]["iterations"] if point["oo"]]
        summary.append({"method": method, "shots": shots, "trials": len(group),
                        "completed": sum(row["completed"] for row in group),
                        "failed": sum(row["status"] in ("failed", "process_failure") for row in group),
                        "chemical_accuracy_count": sum(row["chemical_accuracy"] for row in group),
                        "accuracy_0p1_mha_count": sum(row["accuracy_0p1_mha"] for row in group),
                        "mean_error_mha_completed": float(np.mean(errors)) if errors.size else None,
                        "std_error_mha_completed": float(np.std(errors, ddof=1)) if errors.size > 1 else None,
                        "median_error_mha_completed": float(np.median(errors)) if errors.size else None,
                        "max_error_mha_completed": float(np.max(errors)) if errors.size else None,
                        "mean_noise_penalty_mha": float(np.mean(errors) - 1000 * baseline_errors[method]) if errors.size and method in baseline_errors else None,
                        "oo_updates": len(updates),
                        "shifted_updates": sum((point["hessian_shift"] or 0) > 0 for point in updates),
                        "capped_updates": sum(point["line_search_scale"] is not None and point["line_search_scale"] > 0 and abs(point["step_norm"]/point["line_search_scale"] - 0.2) < 1e-8 for point in updates) if method=="newton_stable" else 0,
                        "backtracked_updates": sum(point["line_search_scale"] is not None and 0 < point["line_search_scale"] < 1 for point in updates),
                        "rejected_updates": sum(point["status"]=="LineSearchRejected" for point in updates),
                        "exact_uphill_oo_updates": sum(point["exact_energy_after_ha"] > point["exact_energy_before_ha"] + 1e-10 for point in updates)})
    status_counts = {status: sum(row["status"]==status for row in rows) for status in sorted({row["status"] for row in rows})}
    report = {"fci_energy_ha": manifest["fci_energy_ha"], "task_status_counts": status_counts, "summary": summary,
              "runs": [{key: value for key, value in row.items() if key != "trajectory"} for row in rows]}
    write_json(work / "summary.json", report)
    make_plots(work, manifest, rows)
    import matplotlib
    write_json(work / "analysis_provenance.json", {"script": "analysis_source.py", "sha256": sha256(work / "analysis_source.py"),
               "python": sys.executable, "numpy": np.__version__, "matplotlib": matplotlib.__version__})
    lines = [f"# {geometry_label(manifest)}: shot-noise evaluation", "",
             f"Run status: {status_counts}." + (" **Partial preview; do not interpret unfinished accuracy rates.**" if any(row["status"] in ("pending", "running") for row in rows) else ""), "",
             f"FCI/STO-3G reference: **{manifest['fci_energy_ha']:.12f} Ha**.", "",
             f"Three orbital optimizers, {manifest['iterations']} fixed macro iterations, UCCD(seq), zero amplitudes and RHF orbitals.",
             f"Shots per independent nonidentity Pauli word: {manifest['shots']}; seeds: {manifest['seeds']}.",
             "Exact statevector diagnostics are excluded from optimization and stopping. No hardware noise or measurement grouping.", "",
             f"Each full run uses {1+383*manifest['iterations']:,} optimization/measurement RDM evaluations, each with 2,778 nonidentity Pauli words.",
             f"The {manifest['iterations']+1} extra exact diagnostic evaluations have no modeled shots and do not advance the sampling seed.", "",
             "## Noiseless controls", "", "| Method | Final exact energy (Ha) | FCI error (mHa) | Status |", "|---|---:|---:|---|"]
    for row in rows:
        if row["shots"] is None:
            last = row["trajectory"]["iterations"][-1] if row["trajectory"] else None
            energy = f"{last['exact_energy_ha']:.12f}" if last else "—"
            error = f"{row['error_ha']*1000:.6f}" if last else "—"
            lines.append(f"| {LABELS[row['method']]} | {energy} | {error} | {row['status']} |")
    lines += ["", "## Final exact-state errors", "", "Mean and sample standard deviation use completed trajectories only; accuracy counts use all requested trials.", "",
              "| Shots/word | Method | Completed | Error mean ± SD (mHa) | Median (mHa) | Within 1.6 mHa | Within 0.1 mHa |", "|---:|---|---:|---:|---:|---:|---:|"]
    for group in summary:
        mean, std, median = (group[k] for k in ("mean_error_mha_completed", "std_error_mha_completed", "median_error_mha_completed"))
        estimate = "—" if mean is None else f"{mean:.5f} ± {std:.5f}" if std is not None else f"{mean:.5f}"
        med = "—" if median is None else f"{median:.5f}"
        n = group['trials']
        lines.append(f"| {group['shots']:,} | {LABELS[group['method']]} | {group['completed']}/{n} | {estimate} | {med} | {group['chemical_accuracy_count']}/{n} | {group['accuracy_0p1_mha_count']}/{n} |")
    lines += ["", "## Orbital-update diagnostics", "",
              "Counts are over recorded OO updates. An exact uphill step can occur even if the sampled frozen-RDM energy decreases.", "",
              "| Shots/word | Method | Updates | Hessian shifted | Step capped | Backtracked | Rejected | Exact OO energy increased |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for group in summary:
        lines.append(f"| {group['shots']:,} | {LABELS[group['method']]} | {group['oo_updates']} | {group['shifted_updates']} | {group['capped_updates']} | {group['backtracked_updates']} | {group['rejected_updates']} | {group['exact_uphill_oo_updates']} |")
    lines += ["", "## Figures", "", "![Exact energy trajectories](figures/convergence.png)", "", "![Endpoint errors](figures/endpoints.png)", "", "![Newton diagnostics](figures/newton_diagnostics.png)", "",
              "## Interpretation and limits", "",
              "- OO(Newton) is this repository's Mizukami-type analytic-gradient/Hessian implementation, with one orbital update per macro iteration.",
              "- Stabilized Newton uses Hessian floor 1e-4, step-norm cap 0.2, and Armijo backtracking; settings were held fixed across shot counts.",
              "- A completed fixed-budget run is not proof of convergence. The accuracy thresholds describe the final exact state, not a lucky noisy energy estimate.",
              f"- {len(manifest['seeds'])} seeds quantify this fixed-budget comparison; they do not establish optimizer superiority for other geometries or measurement models.",
              "- All methods use the same measurement protocol and fixed macro budget; full-budget runs have identical modeled shot costs at each shot count.",
              "- Raw data, configs, source snapshot, binary hashes, execution metadata and numerical summaries are preserved beside this report."]
    stable_groups = [group for group in summary if group["method"]=="newton_stable"]
    interventions = sum(group[key] for group in stable_groups for key in ("shifted_updates", "capped_updates", "backtracked_updates", "rejected_updates"))
    if interventions==0 and all(row["status"] not in ("pending", "running") for row in rows):
        lines += ["", "No Hessian shifts, step caps, backtracks or rejected searches were observed in stabilized Newton.",
                  "The stabilization controls therefore did not intervene in this parameter range; endpoint differences cannot be credited to those controls.",
                  "The raw and stabilized implementations solve the same unshifted Newton system using pivoted LU and Cholesky respectively.",
                  "Finite-precision differences can lead to different subsequent binomial samples even with matched base seeds, so numerical trajectories need not be identical."]
    (work / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Report: {work / 'report.md'}")


def make_plots(work, manifest, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    colors = {"seq": "#0072B2", "newton_raw": "#D55E00", "newton_stable": "#009E73"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(manifest["shots"]), figsize=(15, 4.5), sharey=True, squeeze=False)
    for ax, shots in zip(axes[0], manifest["shots"]):
        for method in METHODS:
            group = [r for r in rows if r["method"] == method and r["shots"] == shots and r["trajectory"]]
            control = next((r for r in rows if r["method"]==method and r["shots"] is None and r["trajectory"]), None)
            if control:
                points = control["trajectory"]["iterations"]
                ax.plot([p["macro_iteration"] for p in points],
                        [max(1e-6,1000*(p["exact_energy_ha"]-manifest["fci_energy_ha"])) for p in points],
                        color=colors[method], linestyle="--", linewidth=1.0, alpha=.7)
            for row in group:
                trajectory = row["trajectory"]["iterations"]
                x = [p["macro_iteration"] for p in trajectory]
                y = [max(1e-6, 1000*(p["exact_energy_ha"]-manifest["fci_energy_ha"])) for p in trajectory]
                ax.plot(x, y, color=colors[method], alpha=0.2, linewidth=0.8)
                if row["status"] in ("failed", "process_failure"):
                    ax.plot(x[-1], y[-1], "x", color=colors[method], markersize=7)
            medians = []
            for iteration in range(manifest["iterations"]+1):
                values = [1000*(p["exact_energy_ha"]-manifest["fci_energy_ha"]) for row in group for p in row["trajectory"]["iterations"] if p["macro_iteration"]==iteration]
                medians.append(max(1e-6, float(np.median(values))) if values else np.nan)
            ax.plot(range(len(medians)), medians, color=colors[method], label=LABELS[method], linewidth=2)
        ax.axhline(1.6, color="0.4", linestyle="--", linewidth=0.9)
        ax.axhline(0.1, color="0.6", linestyle=":", linewidth=0.9)
        ax.set(title=f"{shots:,} shots / Pauli word", xlabel="Macro iteration", yscale="log")
        ax.grid(alpha=0.2, which="both")
    axes[0,0].set_ylabel("Exact state energy error from FCI (mHa)")
    axes[0,-1].legend(fontsize=8)
    fig.suptitle(f"{geometry_label(manifest)} / STO-3G: noisy medians (solid), noiseless controls (dashed)")
    fig.tight_layout()
    for extension in ("png", "pdf"): fig.savefig(work / "figures" / f"convergence.{extension}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for si, shots in enumerate(manifest["shots"]):
        for mi, method in enumerate(METHODS):
            values = [r["error_ha"]*1000 for r in rows if r["shots"]==shots and r["method"]==method and r["completed"]]
            position = si + (mi-1)*0.24
            if values:
                ax.scatter(position+np.linspace(-0.045,0.045,len(values)), values, color=colors[method], alpha=0.8, s=35, label=LABELS[method] if si==0 else None)
                ax.plot([position-.07,position+.07], [np.median(values)]*2, color=colors[method], linewidth=3)
    ax.axhline(1.6, color="0.4", linestyle="--", label="1.6 mHa")
    ax.axhline(0.1, color="0.6", linestyle=":", label="0.1 mHa")
    controls = [r["error_ha"]*1000 for r in rows if r["shots"] is None and r["completed"]]
    if controls:
        ax.axhline(np.median(controls), color="0.25", linestyle="-.", linewidth=.9, label="Median noiseless endpoint")
    ax.set_xticks(range(len(manifest["shots"])), [f"{n:,}" for n in manifest["shots"]])
    ax.set(xlabel="Shots per independent Pauli word", ylabel="Final exact state energy error from FCI (mHa)", yscale="log",
           title=f"{geometry_label(manifest)}: fixed-budget endpoints (points: trials; bars: medians)")
    ax.legend(fontsize=8);ax.grid(alpha=0.2, which="both");fig.tight_layout()
    for extension in ("png", "pdf"): fig.savefig(work / "figures" / f"endpoints.{extension}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, len(manifest["shots"]), figsize=(15, 7), squeeze=False, sharex=True)
    for si, shots in enumerate(manifest["shots"]):
        for method in ("newton_raw", "newton_stable"):
            group = [r for r in rows if r["method"]==method and r["shots"]==shots and r["trajectory"]]
            for row in group:
                points=[p for p in row["trajectory"]["iterations"] if p["oo"]]
                x=[p["macro_iteration"] for p in points]
                axes[0,si].plot(x,[max(1e-10,p["oo"]["step_norm"]) for p in points], color=colors[method], alpha=.45)
                axes[1,si].plot(x,[p["oo"]["hessian_shift"] for p in points], color=colors[method], alpha=.45)
            axes[0,si].plot([],[],color=colors[method],label=LABELS[method])
        axes[0,si].set(title=f"{shots:,} shots / word", yscale="log")
        axes[0,si].axhline(0.2,color="0.4",linestyle="--",linewidth=.8)
        axes[1,si].set_xlabel("Macro iteration")
        for ax in axes[:,si]:ax.grid(alpha=.2)
    axes[0,0].set_ylabel("Accepted Newton step norm (rad)");axes[1,0].set_ylabel("Hessian shift")
    axes[0,-1].legend(fontsize=8);fig.tight_layout()
    for extension in ("png", "pdf"):fig.savefig(work / "figures" / f"newton_diagnostics.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--work", type=Path, required=True)
    prep.add_argument("--binary", type=Path, default=ROOT / "target/release/examples/evaluate_shot_noise")
    prep.add_argument("--shots", type=int, nargs="+", default=[1000,10000,100000])
    prep.add_argument("--distance", type=float, default=1.6)
    prep.add_argument("--seeds", type=int, nargs="+")
    prep.add_argument("--seed-count", type=int, default=5)
    prep.add_argument("--reuse", type=Path, help="Import compatible completed runs from a prior workspace")
    prep.add_argument("--iterations", type=int, default=10)
    prep.add_argument("--cpus", type=int, default=2)
    worker = sub.add_parser("task")
    worker.add_argument("--work", type=Path, required=True);worker.add_argument("--index",type=int,required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--work",type=Path,required=True);summary.add_argument("--allow-partial",action="store_true")
    args = parser.parse_args()
    if args.command=="prepare":
        if not math.isfinite(args.distance) or args.distance <= 0:parser.error("Distance must be positive and finite")
        if args.seed_count < 1:parser.error("seed-count must be positive")
        if args.seeds is None:args.seeds=[20261020 + 1_000_000*i for i in range(args.seed_count)]
        if args.iterations<1 or args.cpus<1 or any(n<1 for n in args.shots):parser.error("Counts must be positive")
        budget = 1 + 383 * args.iterations
        seeds = sorted(args.seeds)
        if any(seed < 0 or seed + budget > 2**64 for seed in seeds):parser.error("Seed range exceeds u64")
        if any(b-a < budget for a,b in zip(seeds,seeds[1:])):parser.error("Trial seed windows overlap under base_seed + evaluation_count")
        if len(set(args.shots)) != len(args.shots):parser.error("Shot counts must be distinct")
        prepare(args)
    elif args.command=="task":return task(args)
    else:summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
