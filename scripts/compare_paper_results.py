#!/usr/bin/env python3
"""Compare downloaded GCS PES tasks with a prepared argo rerun.

The archive layout is configs/task_N.json and task_N/results.db; the argo
layout is configs/task_NNN.json and results/task_NNN.db. Both are read-only.
A shorter archived iteration budget is reported separately from final-energy
agreement. This program does not download data or change GCS permissions.
"""

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import sqlite3


METHODS = ("UCCD(seq)", "UCCSD(seq)", "OO(seq)-UCCD(seq)", "OO(lbfgs)-UCCD(seq)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_run(database, config_path):
    config = json.loads(config_path.read_text())
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as conn:
        require(conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)], f"Corrupt DB: {database}")
        rows = conn.execute("SELECT id,conditions_json,final_energy,is_completed FROM runs").fetchall()
        require(len(rows) == 1, f"Expected one run: {database}")
        run_id, raw, energy, completed = rows[0]
        require(completed and energy is not None and math.isfinite(energy), f"Incomplete run: {database}")
        conditions = json.loads(raw)
        rows = conn.execute(
            "SELECT macro_idx,energy,converged FROM macro_iterations WHERE run_id=? ORDER BY macro_idx,rowid",
            (run_id,),
        ).fetchall()
    history = {index: (value, bool(flag)) for index, value, flag in rows}
    indices = sorted(history)
    limit = config["method"]["max_macro_iterations"]
    require(len(indices) >= 2 and indices == list(range(1, indices[-1] + 1)), f"Invalid history: {database}")
    require(all(math.isfinite(value) for _, value, _ in rows), f"Nonfinite history: {database}")
    last = indices[-1]
    if len(rows) != len(history):
        require(len(rows) == len(history) + 1 and last == limit and rows[-1] == rows[-2],
                f"Unexpected duplicate iterations: {database}")
    delta = abs(history[last][0] - history[last - 1][0])
    require(last <= limit and (history[last][1] or last == limit), f"Invalid stopping iteration: {database}")
    require(history[last][1] == (delta < config["method"]["energy_tolerance"]), f"Invalid convergence flag: {database}")
    require(abs(energy - history[last][0]) < 1e-12, f"Final/history mismatch: {database}")
    require(conditions["method"] == config["method"]["label"], f"Method mismatch: {database}")
    for key in ("geometry", "basis", "freeze_core"):
        require(conditions[key] == config["chemistry"][key], f"Wrong {key}: {database}")
    key = (conditions["method"], round(float(conditions["geometry"].split()[-1]), 8))
    return key, {"path": str(database), "config": config, "energy": energy,
                 "history": history, "iterations": last, "converged": history[last][1]}


def load_runs(root, archive):
    result = {}
    paths = sorted(root.glob("task_*/results.db") if archive else (root / "results").glob("task_*.db"))
    for database in paths:
        task = database.parent.name if archive else database.stem
        key, run = read_run(database, root / "configs" / f"{task}.json")
        require(key not in result, f"Duplicate method/geometry: {key}")
        result[key] = run
    expected = {(method, round(1 + 0.2 * point, 8)) for method in METHODS for point in range(14)}
    require(result.keys() == expected, f"Coverage mismatch: missing={expected - result.keys()}, extra={result.keys() - expected}")
    return result


def physical_settings(config):
    value = copy.deepcopy(config)
    value.pop("task_name")
    value["method"].pop("max_macro_iterations")
    value["method"].setdefault("rdm_evaluation", {"type": "statevector"})
    value["method"].setdefault("orbital_optimizer", None)
    return value


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--rerun-workspace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    require(math.isfinite(args.tolerance) and args.tolerance > 0, "Tolerance must be positive and finite")
    archive_root, rerun_root = args.archive_root.resolve(), args.rerun_workspace.resolve()
    old, new = load_runs(archive_root, True), load_runs(rerun_root, False)
    rows, histories = [], []
    for key in sorted(old, key=lambda k: (METHODS.index(k[0]), k[1])):
        reference, rerun = old[key], new[key]
        common = sorted(reference["history"].keys() & rerun["history"].keys())
        local_history = []
        for index in common:
            before, after = reference["history"][index][0], rerun["history"][index][0]
            local_history.append({"method": key[0], "R_angstrom": key[1], "macro_idx": index,
                                  "gcs_energy_hartree": before, "argo_energy_hartree": after,
                                  "absolute_difference_hartree": abs(after - before)})
        histories.extend(local_history)
        worst = max(local_history, key=lambda row: row["absolute_difference_hartree"])
        extended = (not reference["converged"] and reference["iterations"] == reference["config"]["method"]["max_macro_iterations"]
                    and rerun["iterations"] > reference["iterations"])
        same_endpoint = reference["iterations"] == rerun["iterations"]
        require(same_endpoint or extended, f"Unexpected iteration-count change for {key}")
        require(extended or reference["converged"] == rerun["converged"], f"Unexpected stopping-status change for {key}")
        final_error = abs(rerun["energy"] - reference["energy"])
        rows.append({"method": key[0], "R_angstrom": key[1], "archive_db": reference["path"],
                     "argo_db": rerun["path"], "input_settings_match": physical_settings(reference["config"]) == physical_settings(rerun["config"]),
                     "gcs_iteration_limit": reference["config"]["method"]["max_macro_iterations"],
                     "argo_iteration_limit": rerun["config"]["method"]["max_macro_iterations"],
                     "gcs_iterations": reference["iterations"], "argo_iterations": rerun["iterations"],
                     "gcs_converged": reference["converged"], "argo_converged": rerun["converged"],
                     "continued_beyond_archived_cap": extended, "same_final_iteration": same_endpoint,
                     "gcs_final_energy_hartree": reference["energy"], "argo_final_energy_hartree": rerun["energy"],
                     "final_absolute_difference_hartree": final_error,
                     "final_energy_matches": same_endpoint and final_error < args.tolerance,
                     "common_iterations": len(common), "max_history_difference_hartree": worst["absolute_difference_hartree"],
                     "max_history_difference_macro_idx": worst["macro_idx"]})
    comparable = [row for row in rows if row["same_final_iteration"]]
    report = {"archive_root": str(archive_root), "rerun_workspace": str(rerun_root),
              "tolerance_hartree": args.tolerance, "points": len(rows),
              "all_physical_inputs_match": all(row["input_settings_match"] for row in rows),
              "comparable_final_energies": len(comparable),
              "matching_final_energies": sum(row["final_energy_matches"] for row in comparable),
              "max_comparable_final_difference_hartree": max(row["final_absolute_difference_hartree"] for row in comparable),
              "common_history_entries": len(histories),
              "max_history_difference_hartree": max(row["absolute_difference_hartree"] for row in histories),
              "history_entries_over_tolerance": [row for row in histories if row["absolute_difference_hartree"] >= args.tolerance],
              "extended_runs": [row for row in rows if row["continued_beyond_archived_cap"]],
              "full_final_reference_available": len(comparable) == len(rows),
              "by_method": {}, "results": rows}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        final = [row for row in selected if row["same_final_iteration"]]
        report["by_method"][method] = {
            "points": len(selected), "comparable_final_energies": len(final),
            "matching_final_energies": sum(row["final_energy_matches"] for row in final),
            "max_comparable_final_difference_hartree": max(row["final_absolute_difference_hartree"] for row in final),
            "max_common_history_difference_hartree": max(row["max_history_difference_hartree"] for row in selected)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(args.out_dir / "points.csv", rows)
    write_csv(args.out_dir / "macro_iterations.csv", histories)
    print(json.dumps({key: value for key, value in report.items() if key not in ("results", "extended_runs")}, indent=2))
    print("Runs continued beyond archived cap:", [(row["method"], row["R_angstrom"]) for row in report["extended_runs"]])
    # Exit status reports comparable input/final-energy failures. Missing later
    # reference iterations and trajectory deviations remain explicit report fields.
    require(report["all_physical_inputs_match"] and report["matching_final_energies"] == len(comparable),
            "Physical settings or comparable final energies disagree; inspect comparison.json")


if __name__ == "__main__":
    main()
