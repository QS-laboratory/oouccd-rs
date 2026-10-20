# Configuration and outputs

The CLI accepts one task JSON, not a scan template:

```bash
target/release/oouccd-rs --config-filepath task.json --db-filepath fresh.db
```

`examples/config_h2_manual.json` is a complete task. The following enables
sequential OO for LiH:

```json
{
  "task_name": "LiH_R160_SOO_UCCD",
  "chemistry": {
    "geometry": "Li 0 0 0; H 0 0 1.6",
    "basis": "sto-3g",
    "charge": 0,
    "spin": 0,
    "freeze_core": 0
  },
  "method": {
    "label": "OO(seq)-UCCD(seq)",
    "max_macro_iterations": 100,
    "energy_tolerance": 1e-8,
    "rdm_evaluation": { "type": "statevector" },
    "ucc_optimizer": {
      "ansatz_config": { "type": "UCCD" },
      "solver_config": { "type": "sequential" }
    },
    "orbital_optimizer": {
      "solver_config": { "type": "sequential" }
    }
  }
}
```

## Chemistry and conventions

Distances in the PySCF geometry string are Å; energies and energy tolerances
are Hartree. `spin` is PySCF's `N_alpha - N_beta`, not multiplicity. The
implemented reference and occupied–virtual orbital partition target closed-shell
RHF calculations; arbitrary open-shell accuracy is not established.

`freeze_core` is a number of doubly occupied spatial orbitals. The supplied LiH
protocol uses zero, hence four electrons, six spatial orbitals and twelve
qubits. The prepared state starts at the RHF determinant with zero amplitudes.
Doubles precede singles in the fixed circuit order and the amplitude sweep.

The real spatial RDMs are spin summed. The two-particle RDM includes a factor
of one half, so the energy is
`E_nuc + sum(h1 * rdm1) + sum(h2 * rdm2)` with no further factor of one half.

## Optimization

`ansatz_config.type` accepts `UCCD` or `UCCSD`. Both amplitude and orbital
`solver_config` accept `{"type":"sequential"}` or
`{"type":"lbfgs","max_iter":200}`. Omitting `orbital_optimizer`, or setting it
to `null`, keeps orbitals fixed. The paper uses sequential amplitudes throughout.

One sequential call is one coordinate sweep, with five angular evaluations per
amplitude coordinate and nine per orbital coordinate. It is not an inner loop
until convergence. Orbital trial energies hold the RDM fixed. L-BFGS orbital
updates instead run their inner optimization, using central finite differences
with step `1e-5`, history length ten, gradient tolerance `1e-6`, and More–Thuente
line search, up to the configured limit.

The outer loop stops at `abs(E_m - E_(m-1)) < energy_tolerance` or the macro
limit. Defaults are 50 macros and `1e-6` Ha; the paper template explicitly uses
100 and `1e-8` Ha. Reaching the cap is recorded as completed but not converged.

## RDM evaluation

`{"type":"statevector"}` is the default and the paper setting. Finite shots
are selected by, for example,
`{"type":"finite_shot","shots":10000,"seed":42}`.

Finite-shot evaluation prepares an ideal statevector, computes unique Pauli
expectations, and samples independent binomial counts per word. Reused words
share their estimate across RDM elements; the identity is exact. It models
independent term-by-term ideal measurements, not grouped bitstrings, gate errors
or readout errors. Simulator wall time is not a quantum measurement-cost estimate.
In this main-branch implementation, exact statevector RDMs are computed directly
from amplitudes; measurement unification and newer noise methods live on the
development branch.

## SQLite and environment variables

The application writes `runs`, `macro_iterations` and parameter-related tables.
`runs.conditions_json` identifies chemistry, method label and RDM mode;
`final_energy` includes nuclear repulsion. `macro_iterations` stores `macro_idx`,
energy and the convergence flag. See [the schema](../src/checkpoint/sqlite.rs).
At an iteration cap the logger can append a duplicate final row; analysis tools
collapse identical macro indices. A database is not an optimizer checkpoint:
reusing matching conditions appends histories from a new zero-initialized run.

`--db-filepath` defaults to `results.db`; `DB_FILEPATH` can supply that default.
The application writes local SQLite files only and rejects storage URIs.
Choose a fresh path for each run. Environment variables from old cloud
launchers do not override the selected local output path.
