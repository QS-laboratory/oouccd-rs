#!/usr/bin/env python3
"""Check a prepared Slurm workspace, including numerical 1.60 A checkpoints.

Writes validation.json; exit status is nonzero for missing/invalid results or
checkpoint energy disagreement. Full historical comparison needs the old DBs.
"""

import argparse
import json
import math
from pathlib import Path
import sqlite3


CHECKPOINTS = {
    "UCCD(seq)": (-7.881875306295823, 4),
    "UCCSD(seq)": (-7.882313807184405, 7),
    "OO(seq)-UCCD(seq)": (-7.882313806735446, 7),
    "OO(lbfgs)-UCCD(seq)": (-7.882313807311107, 7),
}


def check(path, task):
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)], "SQLite integrity"
        runs = conn.execute("SELECT id, conditions_json, final_energy, is_completed FROM runs").fetchall()
        assert len(runs) == 1, f"Expected one run, got {len(runs)}"
        run_id, raw, energy, completed = runs[0]
        assert completed and energy is not None and math.isfinite(energy), "Incomplete/nonfinite run"
        conditions = json.loads(raw)
        assert conditions["method"] == task["method"]["label"], "Wrong method"
        for key in ("geometry", "basis", "freeze_core"):
            assert conditions[key] == task["chemistry"][key], f"Wrong {key}"
        assert conditions["rdm_evaluation"] == {"type": "statevector"}, "Wrong RDM mode"
        rows = conn.execute(
            "SELECT macro_idx, energy, converged FROM macro_iterations WHERE run_id=? ORDER BY macro_idx, rowid",
            (run_id,),
        ).fetchall()
    assert rows, "Empty history"
    assert all(math.isfinite(row[1]) for row in rows), "Nonfinite history"
    history = {index: (value, flag) for index, value, flag in rows}
    indices = sorted(history)
    last = indices[-1]
    limit = task["method"]["max_macro_iterations"]
    tolerance = task["method"]["energy_tolerance"]
    assert indices == list(range(1, last + 1)) and 2 <= last <= limit, "Invalid indices"
    assert abs(energy - history[last][0]) < 1e-12, "Final/history mismatch"
    if len(rows) != len(history):
        assert len(rows) == len(history) + 1 and last == limit, "Unexpected duplicate rows"
        assert rows[-1] == rows[-2], "Unexpected duplicate final iteration"
    delta = abs(history[last][0] - history[last - 1][0])
    converged = bool(history[last][1])
    assert converged == (delta < tolerance), "Wrong convergence flag"
    assert converged or last == limit, "Stopped early"
    assert not any(flag for _, flag in list(history.values())[:-1]), "Continued after convergence"
    for previous, current in zip(indices, indices[1:]):
        assert history[current][0] <= history[previous][0] + 1e-8, "Energy increased"
    result = {"method": conditions["method"], "geometry": conditions["geometry"],
              "energy_hartree": energy, "iterations": last, "converged": converged,
              "final_delta_hartree": delta}
    if math.isclose(float(task["chemistry"]["geometry"].split()[-1]), 1.6, abs_tol=1e-10):
        reference, expected_iterations = CHECKPOINTS[conditions["method"]]
        result.update(checkpoint_error_hartree=energy - reference,
                      expected_iterations=expected_iterations,
                      iteration_count_matches=(last == expected_iterations))
        assert abs(energy - reference) < 1e-8, f"Checkpoint mismatch: {energy - reference:.3e} Ha"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument('--report', type=Path, help='Write the report here instead of workspace/validation.json')
    args = parser.parse_args()
    work = args.workspace.resolve()
    manifest = json.loads((work / "manifest.json").read_text())
    expected = list(range(56)) if manifest["scope"] == "full" else [3, 17, 31, 45]
    assert manifest["task_indices"] == expected, "Unexpected task coverage"
    report = {"scope": manifest["scope"], "results": [], "errors": [],
              "reference_scope": "Structural/stopping checks at every point; numerical checkpoints at 1.60 A. Full historical comparison is a separate check."}
    for index in manifest["task_indices"]:
        name = f"task_{index:03d}"
        try:
            task = json.loads((work / "configs" / f"{name}.json").read_text())
            result = check(work / "results" / f"{name}.db", task)
            report["results"].append({"task": name, **result})
            print(f"{name} {result['method']:24s} E={result['energy_hartree']:.15f} "
                  f"iterations={result['iterations']} converged={result['converged']}")
        except (AssertionError, OSError, sqlite3.Error, ValueError, KeyError) as error:
            report["errors"].append({"task": name, "error": str(error)})
            print(f"FAIL {name}: {error}")
    report["passed"] = not report["errors"]
    destination = args.report if args.report else work / 'validation.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
