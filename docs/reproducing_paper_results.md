# Reproducing the paper results

For the argo Slurm environment and the fresh-checkout limitations discovered on
October 19, 2026, see section 8. The absolute macOS paths below describe the
earlier archive inspection; those ignored inputs and databases are not present
in a fresh clone.

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

## 8. Running on argo with Slurm

### Availability and missing historical artifacts

The October 19, 2026 inspection found a login host `argo00` and Slurm CPU
nodes `argo01` and `argo02`. The default partition is `defq`; each compute node
advertises 144 CPUs and 2,310,000 MB of schedulable memory. CPU jobs suffice
for the supplied state-vector backend. Resource availability varies with the
other jobs; capacity alone does not establish numerical reproducibility.

This checkout (`6cd1b8b` before the argo additions) has neither `infra/` nor
`summary/results/lih_pes/`: both are ignored by Git. Therefore sections 4--6
cannot be executed literally on a fresh clone. The original template and
archived iteration histories must be recovered for a full historical comparison.
The user identified the bucket `gs://example-results-bucket`; no object prefix has
yet been established. Anonymous object listing returned HTTP 401, and no
`gcloud` executable or user GCP credential configuration was found on argo.

`scripts/prepare_paper_slurm.py` reconstructs all 56 inputs using the tracked
L-BFGS-only master, the settings documented above, and the current config schema.
These are explicitly reconstructed inputs, not recovered historical JSONs.
The four-method numbering is the same as in section 5: 003, 017, 031, and 045
are the four methods at 1.60 Å. Every task uses the 100-iteration cap from the
manuscript, including the slowly converging stretched-bond UCCSD points.

### Dedicated environment and build

The dedicated scientific Python environment is under `.paper-env/venv`; the
shared `/opt/venvs/py312` environment is unchanged. The inspected shared
environment has different package versions and should not be substituted
silently. The argo environment uses Python 3.12.3, Rust 1.99.1, and the pinned
requirements plus Matplotlib 3.9.4. `pip check` passed.

The argo development setup was subsequently simplified: Rust now uses the
standard `~/.cargo` and `~/.rustup` locations, with `~/.cargo/env` sourced by
the user's shell profiles. The old `.paper-env/cargo` and `.paper-env/rustup`
paths are compatibility symlinks, so the existing paper build scripts still
work. Code Runner invokes Cargo directly; no `run_rust_file.sh` is needed.
The Python environment remains at its original path while paper jobs use it.
The fresh-install recipe below also supports an isolated Rust installation.

For a new installation from the repository root (skip installation steps when
the environment already exists):

```bash
export ROOT="$(git rev-parse --show-toplevel)"
mkdir -p "$ROOT/.paper-env"
python3 -m venv "$ROOT/.paper-env/venv"
"$ROOT/.paper-env/venv/bin/python" -m pip install \
  -r "$ROOT/requirements.txt" 'matplotlib==3.9.4'
export CARGO_HOME="$ROOT/.paper-env/cargo"
export RUSTUP_HOME="$ROOT/.paper-env/rustup"
curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs \
  -o "$ROOT/.paper-env/rustup-init.sh"
sh "$ROOT/.paper-env/rustup-init.sh" -y --no-modify-path \
  --profile minimal --default-toolchain 1.99.1
sbatch scripts/build_paper.sbatch
```

Build on a compute node, then wait for successful completion before preparing
tasks. `scripts/build_paper.sbatch` requests four CPUs, 8 GB, and 30 minutes,
sets the exact PyO3 interpreter, and runs `cargo build --release --locked`.
An actual first build failed because argo01 has the Python runtime library
`libpython3.12.so.1.0` but lacks the development linker name `libpython3.12.so`.
The supplied build script creates that name as a symlink in `.paper-env/lib`
and sets `LIBRARY_PATH` for linking. The following build succeeded; no system
package installation or numerical-code change was needed. Runtime linking uses
the system's existing `libpython3.12.so.1.0`.

### Prepare, submit, and validate

Preparation creates a new workspace and **does not submit jobs**:

```bash
python3 scripts/prepare_paper_slurm.py local_runs/paper-smoke-NEW --scope smoke
sbatch --test-only local_runs/paper-smoke-NEW/run.sbatch
job_id="$(sbatch --parsable local_runs/paper-smoke-NEW/run.sbatch)"
sbatch --dependency="afterok:$job_id" --kill-on-invalid-dep=yes \
  local_runs/paper-smoke-NEW/analyze.sbatch
squeue -u "$USER"
```

The smoke array contains four complete 1.60 Å calculations. To prepare the full
scan, choose a new workspace and use `--scope full`. Submission is the same;
the full scan has 56 array elements. Defaults are one CPU, 4 GB, and two hours
per element, with at most four elements running simultaneously. Set `--cpus`
and `--concurrency` during preparation to change these independently; for
example, `--cpus 1 --concurrency 16` reserves at most 16 CPUs for the array.
The one-CPU default follows the argo scaling benchmark below. Use measured
timings before selecting production limits for 100-iteration points.
The code parallelizes RDM evaluation with
Rayon; `RAYON_NUM_THREADS` follows the allocation and BLAS/OpenMP threads are
set to one to avoid nested thread oversubscription. No MPI ranks or GPUs are
required.

Each workspace contains 56 task JSONs, a private binary copy and its SHA-256,
the source commit archive and tracked diff, Python package versions, plotting
and validation script snapshots, separate databases, and Slurm logs. Preserve
the dedicated Python environment while jobs are pending/running; it is recorded
but is not copied into each workspace. A task lock and existing-file check
prevent accidental DB reuse. A failed run must be retried in a new workspace.

The dependent analysis job checks all selected databases, their completion,
history indices, final energies and convergence flags, and the stopping rule.
Duplicate final rows at the iteration cap are collapsed. The four 1.60 Å
checkpoints are compared with a numerical tolerance of `1e-8` Ha; reference
iteration counts are reported separately. This comparison tolerance is distinct
from both the optimizer stopping rule and the FCI error. The job writes
`validation.json` and regenerates convergence plots with freshly computed FCI
references. It uses the new `--include-fixed-orbitals --pdf` options to produce
all four manuscript curves in both PNG and PDF. Without these options the
convergence script retains its previous two-OO-method, PNG-only behavior.
Full-scan analysis also generates the PES PNG. This establishes a new PDF
export procedure; it does not recover the historical PDF export commands.

Validation can also be rerun without submitting another chemistry calculation:

```bash
python3 scripts/check_paper_slurm.py local_runs/paper-smoke-NEW
sacct -j "$job_id" --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
```

Passing the smoke check establishes reproducibility of the documented numerical
checkpoints on the tested node. Establishing the entire paper's PES and
iteration-history agreement still requires the full scan and the GCS archives.

### Actual argo smoke result, October 19, 2026

Array job `10610` ran all four 1.60 Å methods on argo01 with four CPUs each;
analysis job `10617` passed every database/checkpoint check and generated the
four-curve convergence PNG/PDF. All four iteration counts matched section 6.

| Method | Absolute difference from section 6 (Ha) | Iterations | Slurm elapsed |
| --- | --- | --- | --- |
| UCCD(seq) | 4.1e-14 | 4 | 2 min 32 s |
| UCCSD(seq) | 1.3e-14 | 7 | 5 min 22 s |
| OO(seq)-UCCD(seq) | 1.1e-13 | 7 | 4 min 17 s |
| OO(lbfgs)-UCCD(seq) | 1.1e-13 | 7 | 4 min 17 s |

The workspace is `local_runs/argo-smoke-20261019/`; retain `validation.json`,
`slurm-accounting.txt`, the configs, databases, provenance, and figures. Sampled
step MaxRSS values were approximately 92--95 MiB. A separate argo02 preflight
verified Python/package imports, binary startup, and runtime library resolution;
the chemistry calculations above were performed on argo01.

All 56 inputs have also been prepared in
`local_runs/argo-full-prepared-20261019/`, but **the full scan has not been
submitted**. Following the benchmark and the user's choice, this prepared array
uses one CPU per task and at most 16 simultaneous tasks (16 CPUs and 64 GB
reserved in total), with a two-hour limit per task. The directory includes
`preflight.json`, and `sbatch --test-only` passed with the final one-CPU script.
Fresh calculations do not need GCP authentication; archive retrieval and
historical comparison remain separate work.

### Comparing one and four CPUs

```bash
sbatch --nodelist=argo01 scripts/benchmark_paper_cpus.sbatch \
  "$PWD/local_runs/argo-smoke-20261019"
```

This uses one four-CPU allocation and runs all trials sequentially on that node.
Each Slurm step binds exactly one or four CPUs, with the matching Rayon thread
count. BLAS/OpenMP remain at one thread. Both UCCD and UCCSD perform exactly
one complete first macro-iteration from zero amplitudes at 1.60 Å. The order
for each method is 1, 4, 4, 1 CPUs, yielding two trials per CPU count with fresh
processes/databases. The one-iteration cap is intentional benchmark work and
must not be mistaken for a converged paper calculation.

The unique `local_runs/paper-cpu-bench-JOB_ID.XXXXXX/` workspace contains CPU
affinities, package versions, inputs, databases, individual wall/CPU/RSS timings,
and `benchmark.json`. The report checks that changing CPU count changes the
energy by less than `1e-12` Ha. It reports median wall time, speedup, parallel
efficiency (speedup/4), and the CPU reservation cost ratio (4/speedup). This last
ratio estimates separate one-CPU versus four-CPU production jobs; the benchmark
itself holds its four-CPU allocation throughout for a controlled comparison.

Job `10619` completed the benchmark on argo01. Results (medians of two trials
per setting, including Python initialization, integral generation, and DB output):

| Method | 1 CPU (s) | 4 CPUs (s) | Speedup | Reserved CPU-time ratio, 4/1 |
| --- | --- | --- | --- | --- |
| UCCD(seq) | 40.40 | 37.66 | 1.073 | 3.73 |
| UCCSD(seq) | 50.20 | 47.73 | 1.052 | 3.80 |

The trial energies were identical across CPU counts for each method. The raw
report is `local_runs/paper-cpu-bench-10619.JhqKdk/benchmark.json`. The limited
speedup is consistent with the code structure: RDM elements use Rayon, while
the default QuEST CPU build does not enable OpenMP and the amplitude updates
are sequential. Measured user+system CPU time divided by wall time was about
1.3 for the four-CPU trials, so most of the four-CPU reservation was unused.
This is not a function-level profile or a scaling result for larger molecules.

For this 12-qubit LiH workload, use one CPU per production task. With a fixed
16-CPU budget, 16 independent one-CPU tasks use the resources more efficiently
than four four-CPU tasks. Batch completion time also depends on the iteration
counts, queueing, and the longest task; it has not yet been measured for the
full scan.
