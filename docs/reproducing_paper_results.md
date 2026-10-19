# Reproducing the paper results

## Scope and verification status

This guide covers the LiH potential-energy curve and the convergence comparison
at 1.60 Å in the working manuscript
`<repository-root>/docs/main_rev3.tex`.
The other manuscript versions are independent historical documents and have not
been corrected here. Repository inspection and archive checks were performed on
October 19, 2026. The source revision before these documentation changes was
`d740107`.

There are two distinct reproduction tasks:

1. **Replot archived results:** read the existing SQLite databases and recompute
   the FCI references. This reproduces the available plotting pipeline.
2. **Rerun the calculation:** use the current implementation with the paper's
   stopping rule. Compare numerical results, not necessarily bitwise-identical
   output or identical figure styling.

The archive histories have been checked; a complete fresh 56-calculation scan
has not been rerun as part of preparing this guide. The exact historical Python
environment and PDF export commands are not recorded in the available metadata.

All shell blocks below assume Bash. Set these variables in the same shell:

```bash
export ROOT=<repository-root>
export PY="$ROOT/seq-venv/bin/python3"
```

## 1. Physical model and algorithm

| Item | Setting |
| --- | --- |
| Geometry | `Li 0 0 0; H 0 0 R`, distances in Å |
| Grid | 1.0 through 3.6 Å, 14 points, spacing 0.2 Å |
| Basis | STO-3G |
| Charge / spin | Neutral, PySCF `spin=0` |
| Frozen core | None: four electrons, six spatial orbitals |
| Circuit register | 12 qubits |
| Reference | Restricted Hartree–Fock determinant |
| Initial amplitudes | Zero at each geometry; no cross-geometry warm start |
| Circuit evaluation | Exact state vector; no shot noise |
| Ansatz | One fixed-order product of excitation gates |
| Circuit and amplitude-sweep order | Doubles first, then singles when present; lexicographic within rank |
| Orbital pairs | Occupied–virtual spatial pairs, lexicographic order; eight pairs |
| Sequential orbital update | Nine equally spaced angular shifts over a period |
| Sequential amplitude update | Five equally spaced angular shifts over a period |
| Stopping rule | `abs(E_m - E_(m-1)) < 1e-8` Ha, or 100 macro-iterations |

The four methods are UCCD(seq), UCCSD(seq), OO(seq)-UCCD(seq), and
OO(lbfgs)-UCCD(seq). The PES plot displays the first three; the current
convergence script displays the two orbital-optimized methods.

For OO calculations, each macro-iteration first optimizes orbitals with the
current state's RDMs held fixed, then performs one amplitude sweep with the
updated integrals. Amplitudes carry over between macro-iterations. Orbital
rotations are absorbed into the integrals, and the temporary orbital parameters
are reset. Sequential updates reconstruct Fourier energy profiles and minimize
them using polynomial roots from Aberth's method, with sample points also
included as candidates. This is coordinate-wise minimization, not a guarantee
of the global optimum over all parameters simultaneously.

The orbital L-BFGS comparison uses central finite differences with step `1e-5`,
gradient tolerance `1e-6`, history length 10, More–Thuente line search, and at most
200 inner iterations per orbital optimization in the supplied templates.

Total energies include nuclear repulsion. The code's two-particle RDM includes
the factor 1/2; the energy contraction therefore has no additional explicit
factor 1/2. FCI is exact within this finite basis, not the complete-basis limit.

Implementation sources, all under the absolute repository root above:

- `<repository-root>/src/chemistry/pyscf_driver.rs`
- `<repository-root>/src/chemistry/rdm.rs`
- `<repository-root>/src/chemistry/engine.rs`
- `<repository-root>/src/optimizer/solver.rs`
- `<repository-root>/src/optimizer/trigonometric.rs`
- `<repository-root>/src/quantum/unitary_coupled_cluster_gates.rs`

## 2. Input provenance and the 100-iteration limit

The available master inputs are:

- `<repository-root>/infra/config_templates/config_lih_pes.json`:
  all four methods on the 14-point grid; limits of 30 macro-iterations for the
  sequential methods and 50 for the L-BFGS orbital comparison.
- `<repository-root>/infra/config_templates/config_lih_pes_uccsd_recalc.json`:
  UCCSD at 2.6, 2.8, 3.0, 3.2, and 3.4 Å, with a 100-iteration limit.
- `<repository-root>/config_lih_pes_lbfgs_only.json`:
  the L-BFGS orbital comparison on all 14 geometries, with a 50-iteration limit.

The databases do not record the original JSON filename or full optimizer
settings. Attribution is based on matching task names, geometries, and methods.
In particular, both the main template and the L-BFGS-only template define the
same L-BFGS method, so its launch filename cannot be proved from the database.

Exactly five original UCCSD runs failed to converge within 30 iterations.
All five have replacement histories in
`<repository-root>/summary/results/lih_pes/task_zz_recalc.db`.
Their first 30 iteration energies agree with the original histories within
`9.1e-13` Ha. The replacements are consistent with rerunning the same procedure
with a larger budget; this does not imply checkpoint continuation.

| R (Å) | Original final change at iteration 30 (Ha) | Replacement final iteration | Replacement final change (Ha) | Status |
| --- | --- | --- | --- | --- |
| 2.6 | 2.2362e-8 | 34 | 8.7299e-9 | Converged |
| 2.8 | 2.7860e-7 | 61 | 9.8154e-9 | Converged |
| 3.0 | 3.9682e-7 | 100 | 1.7197e-8 | Iteration limit |
| 3.2 | 1.4503e-7 | 100 | 4.2753e-8 | Iteration limit |
| 3.4 | 2.3201e-8 | 100 | 1.4965e-8 | Iteration limit |

All other original runs converged before their original limits. In the final
selected dataset, the largest final iteration is 6 for UCCD, 23 for sequential
OO-UCCD, 15 for L-BFGS OO-UCCD, and 100 for UCCSD. All 56 selected results satisfy
the paper's stopping rule. **The paper's 100-iteration description is consistent
with the final histories; it does not mean every point converged.** Using 100
from the start is the straightforward reproduction protocol.

The separate convergence template at
`<repository-root>/infra/config_templates/config_lih_convergence.json`
uses 1.5945 Å and different settings; it is not the input corresponding to the
paper's 1.60 Å convergence histories. Current generated task files are from
later experiments and should not be treated as archived paper inputs.

## 3. Environment

Use a Python interpreter compatible with the pinned scientific packages and
the Rust/PyO3 build. The existing environment can be inspected with:

```bash
"$PY" -c 'import sys,numpy,scipy,pyscf,matplotlib; print(sys.version); print(numpy.__version__,scipy.__version__,pyscf.__version__,matplotlib.__version__)'
```

The requirements file pins NumPy 1.26.4, SciPy 1.13.1, PySCF 2.7.0, and
OpenFermion 1.6.1, but does not include Matplotlib. The inspected local environment
instead has Python 3.9.6, NumPy 1.26.4, SciPy 1.13.1, PySCF 2.12.1, and Matplotlib
3.9.4. Neither the current environment nor the requirements alone proves which
versions produced the archived results. Do not silently replace an existing
environment when investigating numerical differences.

For a fresh environment, create it at a new location, then set `PY` accordingly:

```bash
python3 -m venv "$ROOT/paper-venv"
export PY="$ROOT/paper-venv/bin/python3"
"$PY" -m pip install -r "$ROOT/requirements.txt" 'matplotlib==3.9.4'
export PYO3_PYTHON="$PY"
export PYTHONPATH="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')${PYTHONPATH:+:$PYTHONPATH}"
cargo build --release --locked --manifest-path "$ROOT/Cargo.toml"
```

Native compilation requires Rust, a C/C++ toolchain, and Python development
libraries compatible with the selected interpreter. On macOS, Python framework
discovery may also be needed; the existing local runner contains framework-path
setup. Archive the source commit, `Cargo.lock`, Python package versions, actual
task JSONs, and logs with any new results. The original environment is not fully
locked by the Rust lockfile.

## 4. Replot the archived data

These commands write new images to a fresh temporary directory, not the
manuscript figure directory. FCI references are recomputed using PySCF RHF
orbitals and `fci.direct_spin0.FCI`, with `max_cycle=200`; nuclear repulsion is
added to the electronic FCI energy.

```bash
export FIGURES="$(mktemp -d "${TMPDIR:-/tmp}/oouccd-paper-figures.XXXXXX")"
MPLBACKEND=Agg "$PY" "$ROOT/summary/scripts/plot_lih_pes.py" \
  --results-dir "$ROOT/summary/results/lih_pes" \
  --output "$FIGURES/lih_pes.png"
MPLBACKEND=Agg "$PY" "$ROOT/summary/scripts/plot_convergence.py" \
  --results-dir "$ROOT/summary/results/lih_pes" \
  --out-dir "$FIGURES/convergence"
printf 'Figures: %s\n' "$FIGURES"
```

Important plotting details:

- Inputs are files matching `task_*.db`, in lexicographically sorted order.
  For duplicate method–geometry entries, later entries replace earlier ones;
  they are not averaged. `task_zz_recalc.db` supplies the replacement UCCSD data.
- The PES script includes only completed runs with non-null final energy.
  Its lower panel plots positive `E - E_FCI` values on a log scale, not an
  explicit absolute-value transformation.
- The convergence script reads histories without a completion filter, sorts
  by `macro_idx`, and adds the RHF energy as iteration zero. Its current
  `METHOD_ORDER` includes only the two OO methods. UCCD/UCCSD entries are
  commented out; enable them in a separate script copy if those curves are
  needed. At the iteration cap, the logger writes a duplicate final iteration;
  this matters if plotting capped histories, since the script enumerates rows.
- The manuscript includes PDF figures; the existing scripts write PNGs. The
  historical PDF export procedure is not established here. Recreating these
  images is not proof of pixel-identical reproduction of the embedded PDFs.
- The convergence plot at 1.60 Å uses histories from the same PES dataset;
  no separate quantum calculation is needed for that figure.

## 5. Rerun using the manuscript protocol

Use a fresh workspace and one fresh database per task. The following generates
56 task inputs from the existing main template, overriding only the outer limit
to 100 and making state-vector evaluation explicit. This avoids the existing
generator's destructive replacement of its shared generated-config directory.
It also preserves the original templates and archived results.

```bash
export WORK="$(mktemp -d "${TMPDIR:-/tmp}/oouccd-paper-rerun.XXXXXX")"
"$PY" - <<'PY'
import copy
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
work = Path(os.environ["WORK"])
template = json.loads((root / "infra/config_templates/config_lih_pes.json").read_text())
for subdir in ("configs", "results", "logs"):
    (work / subdir).mkdir()
count = 0
for method in template["methods"]:
    for point in range(14):
        task = {
            "task_name": "PaperReproduction_" + method["label"],
            "chemistry": copy.deepcopy(template["chemistry"]),
            "method": copy.deepcopy(method),
        }
        task["chemistry"]["geometry"] = task["chemistry"]["geometry"].replace(
            "{R}", f"{1.0 + 0.2 * point:.5f}"
        )
        task["method"]["max_macro_iterations"] = 100
        task["method"]["rdm_evaluation"] = {"type": "statevector"}
        (work / "configs" / f"task_{count:03d}.json").write_text(
            json.dumps(task, indent=2) + "\n"
        )
        count += 1
assert count == 56
print(f"Generated {count} tasks in {work}")
PY

# Prevent cloud environment variables from overriding the local DB destination.
unset GCS_PATH BATCH_TASK_INDEX
for task in "$WORK"/configs/task_*.json; do
  name="$(basename "$task" .json)"
  "$ROOT/target/release/oouccd-rs" \
    --config-filepath "$task" \
    --db-filepath "$WORK/results/$name.db" \
    >"$WORK/logs/$name.log" 2>&1 || break
done
printf 'Rerun workspace: %s\n' "$WORK"
```

Check that all 56 tasks completed successfully before plotting. Do not rerun
tasks into the same database: matching conditions reuse a run record but append
history rows rather than resuming the optimizer. Completion alone does not mean
convergence: a capped run is also marked completed. Inspect the final history's
`converged` flag and actual energy change.

Use the plotting commands in section 4 with `--results-dir "$WORK/results"` and
a fresh figure directory. A complete scan may be expensive; no runtime estimate
is asserted. For a smoke check, task `031` is sequential OO-UCCD at 1.60 Å in
this newly generated numbering (not the original archive numbering).

## 6. Audit the archived stopping rule

This read-only check applies the PES selection rule, collapses duplicate
macro-indices, and verifies all 56 selected histories. It checks stopping
behavior, not agreement with FCI or the correctness of every circuit gate.

```bash
"$PY" - <<'PY'
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

directory = Path(os.environ["ROOT"]) / "summary/results/lih_pes"
selected = {}
for path in sorted(directory.glob("task_*.db")):
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        for run_id, raw, energy, completed in conn.execute(
            "SELECT id, conditions_json, final_energy, is_completed FROM runs ORDER BY id"
        ):
            if not completed or energy is None:
                continue
            condition = json.loads(raw)
            rows = conn.execute(
                "SELECT macro_idx, energy, converged FROM macro_iterations "
                "WHERE run_id=? ORDER BY macro_idx, rowid", (run_id,)
            ).fetchall()
            history = {index: (value, flag) for index, value, flag in rows}
            selected[condition["method"], condition["geometry"]] = (energy, history)
assert len(selected) == 56, len(selected)
counts = Counter()
capped = []
for (method, geometry), (energy, history) in selected.items():
    indices = sorted(history)
    assert len(indices) >= 2
    last, previous = indices[-1], indices[-2]
    delta = abs(history[last][0] - history[previous][0])
    assert abs(energy - history[last][0]) < 1e-12
    assert bool(history[last][1]) == (delta < 1e-8)
    assert delta < 1e-8 or last == 100, (method, geometry, last, delta)
    counts[method] += 1
    if not history[last][1]:
        capped.append((method, geometry, last, delta))
assert len(counts) == 4 and all(count == 14 for count in counts.values())
assert len(capped) == 3
print("PASS: all 56 selected histories satisfy the paper's stopping rule.")
print("Iteration-capped points:")
for row in sorted(capped):
    print(row)
PY
```

At 1.60 Å, the archived reference values are:

| Method | Total energy (Ha) | Macro-iterations |
| --- | --- | --- |
| UCCD(seq) | -7.881875306295823 | 4 |
| UCCSD(seq) | -7.882313807184405 | 7 |
| OO(seq)-UCCD(seq) | -7.882313806735446 | 7 |
| OO(lbfgs)-UCCD(seq) | -7.882313807311107 | 7 |

Use these as numerical checkpoints, not a promise of last-digit identity across
software environments. The `1e-8` stopping tolerance concerns consecutive
energies, not the energy error relative to FCI. Investigate discrepancies in
environment, excitation order, initialization, and selection of replacement
runs before attributing them to the algorithm.

## 7. Manuscript correction and remaining limitations

The working manuscript previously said singles were applied before doubles.
The circuit builder appends doubles before singles and stores parameters in
that same order. The file's Git history shows no changes since its introduction
on June 3, 2026. The working manuscript is corrected to describe that circuit
and amplitude-sweep order; the numerical implementation and archived results
are unchanged. This evidence supports a description correction, not silently
changing the ansatz used for reproduction. The database itself does not embed
a source commit, so exact historical executable provenance remains incomplete.

The numerical histories support retaining the manuscript's 100-iteration
termination description. They do not support saying that every UCCSD point
converged. The guide does not constitute a full independent verification of
all theoretical claims or of the original PDF figure provenance.

Build the corrected working manuscript with:

```bash
sh "$ROOT/docs/build.sh" main_rev3
```

This requires TeX Live, `latexmk`, and the manuscript's packages, and writes
build products below the repository's ignored documentation build directory.