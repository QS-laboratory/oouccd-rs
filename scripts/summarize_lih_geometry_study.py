"""Validate and analyze a multi-geometry, matched-budget LiH shot-noise study."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
METHODS = ("seq", "newton_raw", "newton_stable")
LABELS = {"seq": "OO(seq)", "newton_raw": "Newton raw", "newton_stable": "Newton stabilized"}
COLORS = {"seq": "#0072B2", "newton_raw": "#D55E00", "newton_stable": "#009E73"}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slurm_audit(root, study):
    """Capture terminal scheduler states, including jobs killed before writing metadata."""
    records = []
    for job in study["submitted_jobs"]:
        work = Path(job["workspace"])
        manifest = json.loads((work / "manifest.json").read_text())
        for attempt in range(12):
            text = subprocess.check_output(
                ["sacct", "-j", job["job_id"], "--format=JobID,State%40,ExitCode,ElapsedRaw,NodeList", "-P", "-n"], text=True,
            )
            states = {}
            for line in text.splitlines():
                fields = line.split("|")
                match = re.fullmatch(re.escape(job["job_id"]) + r"_(\d+)", fields[0])
                if match:
                    states[int(match[1])] = fields
            ready = all(index in states and states[index][1].split()[0].rstrip("+") not in
                        ("PENDING","RUNNING","COMPLETING","CONFIGURING","REQUEUED","SUSPENDED")
                        for index in manifest["pending_task_indices"])
            if ready or attempt == 11:
                break
            time.sleep(5)
        (work / "slurm_accounting.txt").write_text(text)
        for index in manifest["pending_task_indices"]:
            if index not in states:
                raise RuntimeError(f"No scheduler accounting for {job['job_id']}_{index}")
            fields = states[index]
            state = fields[1].split()[0].rstrip("+")
            if state in ("PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "REQUEUED", "SUSPENDED"):
                raise RuntimeError(f"Task is still active: {fields[0]}")
            exit_status, signal = map(int, fields[2].split(":"))
            successful = state == "COMPLETED" and exit_status == 0 and signal == 0
            item = manifest["tasks"][index]
            result = Path(item["result"])
            execution = result.with_name(result.stem + "_execution.json")
            if not execution.exists():
                if successful:
                    raise RuntimeError(f"Completed scheduler task has no execution record: {fields[0]}")
                write_json(execution, {
                    "exit_code": exit_status or -signal or 1,
                    "seconds": int(fields[3]), "hostname": fields[4], "slurm_job_id": fields[0],
                    "slurm_array_task_id": str(index), "affinity": None,
                    "recovered_from_scheduler": True, "scheduler_state": state,
                })
            records.append({"geometry": job["distance"], "task_index": index,
                            "job_id": fields[0], "state": state, "exit_code": fields[2]})
    write_json(root / "slurm_accounting.json", records)
    return records


def number(value, precision=3):
    return "—" if value is None else f"{value:.{precision}f}"


def analyze(root, skip_slurm=False):
    study = json.loads((root / "study.json").read_text())
    scheduler = [] if skip_slurm else slurm_audit(root, study)
    cells, paired, controls, all_runs, validations = [], [], [], [], []
    manifests = {}
    for entry in study["workspaces"]:
        work = Path(entry["work"])
        for script, command in (("evaluate_lih_shot_noise.py", "summarize"), ("compare_shot_noise_accuracy.py", None)):
            arguments = [sys.executable, str(HERE / script)]
            if command:
                arguments.append(command)
            arguments += ["--work", str(work)]
            with (work / "logs" / f"final_{Path(script).stem}.log").open("w") as log:
                subprocess.run(arguments, check=True, stdout=log, stderr=log)
        manifest = json.loads((work / "manifest.json").read_text())
        report = json.loads((work / "summary.json").read_text())
        accuracy = json.loads((work / "accuracy_comparison.json").read_text())
        distance = entry["distance"]
        manifests[distance] = manifest
        if len(manifest["seeds"]) != study["seed_count"]:
            raise ValueError("Seed counts do not match study protocol")
        if manifest["iterations"] != study["macro_iterations"]:
            raise ValueError("Macro budgets do not match study protocol")
        if digest(Path(manifest["binary"])) != manifest["binary_sha256"]:
            raise ValueError("Evaluator binary hash mismatch")
        for group in report["summary"]:
            if group["trials"] != study["seed_count"]:
                raise ValueError("A result cell has an incorrect number of trials")
            cells.append({"distance": distance, **group})
        paired.extend({"distance": distance, **row} for row in accuracy["paired"])
        for row in report["runs"]:
            native = json.loads(Path(row["result"]).read_text()) if Path(row["result"]).exists() else None
            all_runs.append({"distance": distance, **row, "trajectory": native})
            if row["shots"] is None:
                controls.append({"distance": distance, "method": row["method"], "status": row["status"],
                                 "error_mha": row["error_ha"]*1000 if row["error_ha"] is not None else None})
            if row["completed"]:
                expected = 1 + 383 * manifest["iterations"]
                if native["rdm_evaluations"] != expected or native["diagnostic_rdm_evaluations"] != manifest["iterations"]+1:
                    raise ValueError("Unexpected evaluation/diagnostic count")
                if native["unique_measured_pauli_words"] != 2778:
                    raise ValueError("Unexpected measurement plan")
                if row["shots"] is not None and native["modeled_shots"] != expected*2778*row["shots"]:
                    raise ValueError("Unexpected modeled shot budget")
            if native:
                for point in native["iterations"]:
                    if point["orthogonality_error"] >= 1e-8:
                        raise ValueError("Lost orbital orthogonality")
                    if point["exact_energy_ha"] < manifest["fci_energy_ha"] - 1e-7:
                        raise ValueError("Variational bound violation")
            if row.get("reused_from"):
                if digest(Path(row["result"])) != row["reused_from"]["result_sha256"]:
                    raise ValueError("Reused trajectory hash mismatch")
        validations.append({"distance": distance, "tasks": len(report["runs"]),
                            "statuses": report["task_status_counts"], "binary_sha256": manifest["binary_sha256"],
                            "seed_windows_disjoint": all(b-a >= 1+383*manifest["iterations"]
                                                         for a,b in zip(sorted(manifest["seeds"]), sorted(manifest["seeds"])[1:]))})
    if any(not row["seed_windows_disjoint"] for row in validations):
        raise ValueError("Overlapping trial seed windows")
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(paired)), key=lambda i: paired[i]["sign_test_p_two_sided"])):
        previous = max(previous, min(1.0, (len(paired)-rank)*paired[index]["sign_test_p_two_sided"]))
        paired[index]["holm_p_across_all_geometries"] = previous
    if len({row["binary_sha256"] for row in validations}) != 1:
        raise ValueError("Geometry workspaces use different evaluator binaries")
    aggregate = {"study": study, "controls": controls, "cells": cells,
                 "paired_comparisons": paired, "statuses": dict(Counter(row["status"] for row in all_runs)),
                 "runs": [{key:value for key,value in row.items() if key!="trajectory"} for row in all_runs],
                 "validation": validations}
    write_json(root / "study_summary.json", aggregate)
    write_json(root / "study_validation.json", {"geometries": validations, "total_tasks": len(all_runs),
               "new_scheduler_tasks": len(scheduler), "reused_tasks": sum(bool(row.get("reused_from")) for row in all_runs),
               "maximum_orthogonality_error": max((point["orthogonality_error"] for row in all_runs if row["trajectory"] for point in row["trajectory"]["iterations"]), default=0.0)})
    with (root / "geometry_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader(); writer.writerows(cells)
    make_plots(root, study, manifests, all_runs, cells)
    write_report(root, aggregate)
    write_json(root / "analysis_provenance.json", {"python": sys.executable,
               "scripts": {script: digest(HERE / script) for script in ("evaluate_lih_shot_noise.py", "compare_shot_noise_accuracy.py", Path(__file__).name)}})
    print(json.dumps({"tasks": len(all_runs), "statuses": aggregate["statuses"], "report": str(root/'report.md')}, indent=2))


def write_report(root, aggregate):
    study, cells, controls = aggregate["study"], aggregate["cells"], aggregate["controls"]
    distances = [entry["distance"] for entry in study["workspaces"]]
    lookup = {(row["distance"],row["shots"],row["method"]): row for row in cells}
    control = {(row["distance"],row["method"]): row for row in controls}
    lines = ["# LiH stretched-geometry shot-noise comparison", "",
             f"Distances: {distances} Å; STO-3G, 4 electrons / 6 spatial orbitals, no frozen core.",
             f"{study['seed_count']} independent trial seeds per method/shot level; {study['macro_iterations']} fixed macro iterations; UCCD(seq).",
             f"Final task statuses: {aggregate['statuses']}.", "",
             "OO(seq): one orbital sweep; Newton: one simultaneous step per macro. UCC: one sequential amplitude sweep. Both OO and UCC use noisy full RDMs.",
             "Exact diagnostic energies are excluded from optimization and stopping. No hardware noise or measurement grouping is modeled.", "",
             "## Noiseless controls at the same iteration budget", "",
             "These are fixed-budget endpoints, not assumed fully converged solutions at stretched geometries.", "",
             "| R (Å) | OO(seq) error (mHa) | Raw Newton | Stabilized Newton |", "|---:|---:|---:|---:|"]
    for distance in distances:
        values = [number(control[distance,method]["error_mha"]) + (" (failed)" if control[distance,method]["status"]!="completed" else "") for method in METHODS]
        lines.append(f"| {distance:g} | " + " | ".join(values) + " |")
    lines += ["", "## Final exact-state error from FCI", "",
              "Mean ± sample SD (mHa), over completed trials only. Accuracy fractions below include every requested trial.", "",
              "| R (Å) | Shots/word | OO(seq) | Raw Newton | Stabilized Newton |", "|---:|---:|---:|---:|---:|"]
    for distance in distances:
        for shots in study["shots"]:
            values=[]
            for method in METHODS:
                row=lookup[distance,shots,method]
                value=number(row["mean_error_mha_completed"])+" ± "+number(row["std_error_mha_completed"])
                if row["completed"]!=row["trials"]:value+=f" ({row['completed']}/{row['trials']} complete)"
                values.append(value)
            lines.append(f"| {distance:g} | {shots:,} | "+" | ".join(values)+" |")
    for title, key in (("Median final exact-state error from FCI (mHa)", "median_error_mha_completed"),
                       ("Final exact-state accuracy within 1.6 mHa", "chemical_accuracy_count"),
                       ("Mean excess error above each method's noiseless fixed-budget control (mHa)", "mean_noise_penalty_mha")):
        lines += ["", "## "+title, "", "| R (Å) | Shots/word | OO(seq) | Raw Newton | Stabilized Newton |", "|---:|---:|---:|---:|---:|"]
        for distance in distances:
            for shots in study["shots"]:
                values=[]
                for method in METHODS:
                    row=lookup[distance,shots,method]
                    values.append(f"{row[key]}/{row['trials']}" if key=="chemical_accuracy_count" else number(row[key]))
                lines.append(f"| {distance:g} | {shots:,} | "+" | ".join(values)+" |")
    lines += ["", "## Stabilized Newton interventions", "",
              "Operations can co-occur within an update. A decrease in sampled frozen-RDM energy does not guarantee a decrease in exact physical energy.", "",
              "| R (Å) | Shots/word | Updates | Shifted | Capped | Backtracked | Rejected |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for distance in distances:
        for shots in study["shots"]:
            row=lookup[distance,shots,"newton_stable"]
            lines.append(f"| {distance:g} | {shots:,} | {row['oo_updates']} | {row['shifted_updates']} | {row['capped_updates']} | {row['backtracked_updates']} | {row['rejected_updates']} |")
    lines += ["", "## Figures", "", "![Endpoint accuracy](figures/geometry_endpoints.png)", "",
              "![Median and interquartile convergence](figures/geometry_convergence.png)", "",
              "![Stabilization frequency](figures/stabilization_usage.png)", "", "## Statistical interpretation", "",
              "Paired, two-sided sign tests compare which method more often has lower final error, among jointly completed pairs. They do not test equality of mean errors.",
              f"Holm adjustment in study_summary.json covers all {len(aggregate['paired_comparisons'])} geometry/shot/method-pair comparisons. Per-geometry reports also show their within-geometry adjustments.",
              "Failures and endpoint accuracy rates must be read with completed-only mean/median statistics; no artificial finite error is assigned to failed runs.",
              f"This is an exploratory follow-up with {study['seed_count']} trial seeds per condition, including {sum(bool(row.get('reused_from')) for row in aggregate['runs'])} documented reused trajectories. It is not a preregistered confirmatory test.",
              "Raw Newton uses pivoted LU; stabilized Newton uses Cholesky after any Hessian shift. Numerical roundoff can change later stochastic trajectories even for matched trial seeds.", "",
              "| R (Å) | Shots/word | A vs B | Mean A−B (mHa) | A wins / joint completions | Raw p | Holm p, full study |",
              "|---:|---:|---|---:|---:|---:|---:|"]
    for row in aggregate["paired_comparisons"]:
        lines.append(f"| {row['distance']:g} | {row['shots']:,} | {LABELS[row['a']]} vs {LABELS[row['b']]} | {number(row['mean_a_minus_b_mha'])} | {row['a_wins']}/{row['n']} | {row['sign_test_p_two_sided']:.5f} | {row['holm_p_across_all_geometries']:.5f} |")
    lines += ["",
              "## Provenance", "",
              "study.json records job IDs and workspace paths. Each workspace preserves source/binary hashes, configs, all per-iteration states and energies, execution metadata, reference energies and seed windows.",
              "Reused trajectories retain their original execution metadata; their accuracy contributes once. Runtime comparisons are not made across mixed CPU allocations."]
    failures=[row for row in aggregate.get("runs",[]) if row["status"]!="completed"]
    if failures:lines.append(f"Failed trajectories: {len(failures)}.")
    (root/"report.md").write_text("\n".join(lines)+"\n")


def make_plots(root, study, manifests, rows, cells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    output=root/"figures";output.mkdir(exist_ok=True)
    distances=[entry["distance"]for entry in study["workspaces"]]
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,len(study["shots"]),figsize=(15,4.8),sharey=True,squeeze=False)
    for ax,shots in zip(axes[0],study["shots"]):
        for mi,method in enumerate(METHODS):
            medians=[]
            for distance in distances:
                errors=[row["error_ha"]*1000 for row in rows if row["distance"]==distance and row["shots"]==shots and row["method"]==method and row["completed"]]
                position=distance+(mi-1)*.045
                if errors:ax.scatter(position+np.linspace(-.018,.018,len(errors)),errors,color=COLORS[method],s=12,alpha=.4)
                medians.append(np.median(errors)if errors else np.nan)
            ax.plot(np.array(distances)+(mi-1)*.045,medians,"o-",color=COLORS[method],label=LABELS[method],linewidth=2)
        ax.axhline(1.6,color=".4",linestyle="--",linewidth=.8)
        ax.set_xticks(distances);ax.set(title=f"{shots:,} shots / Pauli word",xlabel="R (Å)",yscale="log");ax.grid(alpha=.2,which="both")
    axes[0,0].set_ylabel("Final exact energy error from FCI (mHa)");axes[0,-1].legend(fontsize=8)
    fig.suptitle(f"LiH/STO-3G: {study['seed_count']} trials, {study['macro_iterations']} macro iterations; markers: medians")
    fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(output/f"geometry_endpoints.{ext}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(len(distances),len(study["shots"]),figsize=(15,3.4*len(distances)),sharey=True,squeeze=False)
    for ri,distance in enumerate(distances):
        reference=manifests[distance]["fci_energy_ha"]
        for si,shots in enumerate(study["shots"]):
            ax=axes[ri,si]
            for method in METHODS:
                group=[row for row in rows if row["distance"]==distance and row["shots"]==shots and row["method"]==method and row["trajectory"]]
                quantiles=[]
                for iteration in range(study["macro_iterations"]+1):
                    values=[1000*(point["exact_energy_ha"]-reference)for row in group for point in row["trajectory"]["iterations"]if point["macro_iteration"]==iteration]
                    quantiles.append(np.quantile(values,[.25,.5,.75])if values else [np.nan]*3)
                quantiles=np.array(quantiles);x=np.arange(len(quantiles))
                ax.plot(x,np.maximum(1e-6,quantiles[:,1]),color=COLORS[method],label=LABELS[method])
                ax.fill_between(x,np.maximum(1e-6,quantiles[:,0]),np.maximum(1e-6,quantiles[:,2]),color=COLORS[method],alpha=.13)
                control=next((row for row in rows if row["distance"]==distance and row["shots"]is None and row["method"]==method and row["trajectory"]),None)
                if control:
                    points=control["trajectory"]["iterations"]
                    ax.plot([p["macro_iteration"]for p in points],[max(1e-6,1000*(p["exact_energy_ha"]-reference))for p in points],color=COLORS[method],linestyle="--",alpha=.7,linewidth=.8)
            ax.axhline(1.6,color=".5",linestyle=":",linewidth=.8);ax.set(title=f"R={distance:g} Å; {shots:,} shots",yscale="log");ax.grid(alpha=.2,which="both")
            if ri==len(distances)-1:ax.set_xlabel("Macro iteration")
            if si==0:ax.set_ylabel("FCI error (mHa)")
    axes[0,-1].legend(fontsize=7);fig.suptitle("Solid: noisy median / IQR; dashed: noiseless fixed-budget controls")
    fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(output/f"geometry_convergence.{ext}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),squeeze=False,layout="constrained")
    for ax,key,title in zip(axes[0],("shifted_updates","capped_updates","backtracked_updates"),("Hessian shifted","Step capped","Backtracked")):
        matrix=np.zeros((len(distances),len(study["shots"])))
        for i,distance in enumerate(distances):
            for j,shots in enumerate(study["shots"]):
                row=next(row for row in cells if row["distance"]==distance and row["shots"]==shots and row["method"]=="newton_stable")
                matrix[i,j]=row[key]/row["oo_updates"]if row["oo_updates"]else 0
        image=ax.imshow(matrix,vmin=0,vmax=1,cmap="Blues",aspect="auto")
        for(i,j),value in np.ndenumerate(matrix):ax.text(j,i,f"{value:.0%}",ha="center",va="center",color="white"if value>.5 else"black")
        ax.set_xticks(range(len(study["shots"])),[f"{n:,}"for n in study["shots"]]);ax.set_yticks(range(len(distances)),[f"{d:g} Å"for d in distances]);ax.set(title=title,xlabel="Shots / word")
    fig.colorbar(image,ax=list(axes[0]),label="Fraction of recorded stabilized-Newton updates")
    for ext in ("png","pdf"):fig.savefig(output/f"stabilization_usage.{ext}",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--skip-slurm",action="store_true",help="Analyze a local test fixture without scheduler accounting")
    args=parser.parse_args();analyze(args.study.resolve(),args.skip_slurm)
