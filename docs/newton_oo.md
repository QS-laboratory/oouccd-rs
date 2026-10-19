# Pauli RDM evaluation and Newton orbital optimization

The `pre-newton-oo` tag records the implementation before these changes.
The `statevector` and `finite_shot` settings now share the same compiled
Jordan–Wigner observables and RDM reconstruction. Sequential and L-BFGS orbital
optimization retain their existing ordered Givens rotations.

## Measurement model

For each parameter vector, prepare the statevector once and evaluate every
unique Pauli expectation `mu = <psi|P|psi>`. In `statevector` mode, reconstruct
the RDMs directly from these exact values. In `finite_shot` mode, independently
draw `K ~ Binomial(shots, (1 + mu)/2)` for each Pauli word and use
`2*K/shots - 1` in the same reconstruction.

This is the sampling distribution of independent term-by-term ideal Pauli
measurements. It does not model shared bitstrings from simultaneous measurement,
gate errors, or readout errors. A Pauli estimate is reused wherever that word
appears, so the reconstructed RDM elements are correlated. No RDM purification
or trace correction is applied. The identity is exactly one and needs no shots.
Word order and the `base_seed + evaluation_count` seed sequence are unchanged.

The RDM convention is real and spin-summed. The two-particle RDM already
contains the factor `1/2`, so the energy remains
`E_nuc + sum(h1 * rdm1) + sum(h2 * rdm2)`.
The original amplitude-based functions in `chemistry::rdm` remain available as
an independent reference.

LiH/STO-3G with six spatial orbitals requires 2,778 unique nonidentity words for
these full RDMs. At `N` shots per word the modeled cost is `2778*N` shots per
RDM evaluation. The binomial sampler avoids generating those shots individually;
simulation wall time is not a hardware measurement-cost estimate.

## Newton OO

The orbital pairs remain the closed-shell occupied–virtual spatial pairs.
`OrbitalProblem` holds the current integrals, frozen RDMs, and rotation pairs;
its interface contains no quantum backend. An `OrbitalUpdate` returns either
ordered Givens angles, a rotation matrix, or an unchanged result. Only an adopted
update is applied to the integrals.

For pair `(i,j)`, the real generator has `G[i,j]=+1`, `G[j,i]=-1`, matching the
existing Givens sign. If `L_a` applies that generator to every index of an
integral tensor, the derivatives of the exponential rotation at the current
orbital origin are

```
g_a    = contract(L_a h, RDM)
H_ab   = contract((L_a L_b + L_b L_a) h / 2, RDM)
H d    = -g
R      = exp(sum_a d_a G_a)
```

These are analytic contractions, not finite differences or derivatives of the
ordered Givens product. Mixed derivatives explicitly include both generator
orders. First derivatives are cached; second derivatives are contracted one
pair at a time. This implements the fixed-amplitude orbital Newton formulation
of [Mizukami et al., Phys. Rev. Research 2, 033421 (2020), equations 6–8](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.033421).

Each macro iteration makes **one** Newton OO update, followed by the existing
UCC amplitude optimization. Trial rotations and line searches reuse the frozen
RDMs and perform no additional quantum evaluations. Both OO and UCC use noisy
RDMs when `finite_shot` is selected.

The stabilized solver shifts `H` by
`max(0, hessian_floor - lambda_min(H)) * I`, limits the Euclidean norm of `d`,
and tests `alpha=1, 1/2, ...` against the frozen-RDM Armijo condition
`E(alpha*d) <= E(0) + 1e-4 * alpha * dot(g,d)`.
`max_backtracks=20` allows the initial trial plus twenty halvings.
An unaccepted search produces an unchanged update, with a logged reason.
Energy decrease here refers to the **sampled, frozen-RDM objective**; it is
not a guarantee that the exact physical energy decreases with noisy RDMs.

## Configuration and API

Use `orbital_optimizer.solver_config`:

```json
{
  "type": "newton_raphson",
  "stabilized": true,
  "hessian_floor": 0.0001,
  "max_step_norm": 0.2,
  "gradient_tolerance": 0.000001,
  "max_backtracks": 20
}
```

All fields except `type` are optional with the defaults shown. Set
`stabilized: false` for raw Newton: the Hessian shift, step cap, and line search
are all bypassed. A singular raw Hessian or nonfinite derivatives/step/energy
returns an error. A gradient norm below tolerance or an empty pair list returns
an unchanged update. The stabilized mode requires positive finite Hessian floor
and step norm; gradient tolerance must always be finite and nonnegative.

Newton logs show the before/after energies, gradient norm, Hessian shift,
accepted step norm, line-search scale, and status. `gradient_tolerance` controls
only that OO update; the outer stopping rule remains the configured energy
tolerance.

Existing sequential/L-BFGS JSON shapes and RDM configuration names still work.
Rust callers now use `OrbitalSolverConfig` for OO, while `SolverConfig` remains
the UCC enum. `Optimizer::build` returns `anyhow::Result<Self>`; callers use `?`.
Newton is not a UCC solver. The complete effective OO/UCC settings and outer
iteration limits are included in run conditions, without changing DB tables.
Historical condition records are preserved; newly recorded runs use the
expanded condition keys.

Runnable examples:

- `examples/config_lih_newton.json`: noiseless LiH, stabilized Newton.
- `examples/config_h2_newton_finite_shot.json`: three fixed H2 iterations with
  finite shots. `energy_tolerance: -1` disables early stopping for this smoke
  calculation; it is not a proposed noise-aware convergence criterion.

## Verification

Use the scientific Python environment described in `reproducing_paper_results.md`:

```bash
export PYO3_PYTHON="$PWD/.paper-env/venv/bin/python"
export LIBRARY_PATH="$PWD/.paper-env/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export PYTHONPATH="$(.paper-env/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=4
cargo test --release --locked --lib
cargo build --release --locked --bin oouccd-rs --example benchmark_rdm
.paper-env/venv/bin/python scripts/check_newton_oo.py
```

The smoke runner creates a fresh directory below `local_runs`, saves configs,
logs, databases and a JSON summary, and refuses to overwrite an existing DB.
It uses two fixed macro iterations so the second OO update sees a correlated
UCC state. Add `--solvers newton_raphson newton_raw` to check both Newton modes.
An optional `--baseline /path/to/pre-newton-oo-binary` compares sequential and
L-BFGS energy histories within `1e-8` Ha. That binary must be built from the tag
before changing sources, or from a separate checkout of the tag.

Unit tests compare complete exact RDMs with the direct reference at H2/LiH
sizes, check complex-state real-part conventions and seed reproducibility,
compare analytic derivatives with finite differences of exponential rotations
including noncommuting pairs, and exercise positive/indefinite/singular
Hessians, zero gradients, empty pair lists and rejected line searches.

RDM-only performance measurements, excluding state preparation:

```bash
RAYON_NUM_THREADS=1 target/release/examples/benchmark_rdm
RAYON_NUM_THREADS=4 target/release/examples/benchmark_rdm
```

The JSON report separates one-time plan construction from median direct,
exact-Pauli and finite-shot evaluations. It records the thread count, number
of words, shots and maximum RDM discrepancy. Benchmarks are diagnostic, not
timing assertions in the test suite. Large seed/shot sweeps, statistical
comparisons and paper figures are left for a subsequent evaluation stage.

### Recorded validation (2026-10-19)

On Intel Xeon 6521P with Rust 1.99.1, the 35 library tests passed and all
examples compiled. H2 completed two macro iterations for sequential, L-BFGS,
stabilized Newton and raw Newton in both measurement modes. LiH completed two
iterations for sequential/L-BFGS with exact RDMs and stabilized Newton in both
measurement modes. All eight logged stabilized Newton updates either left the
frozen-RDM energy unchanged or decreased it.

Against the saved `pre-newton-oo` binary, the largest exact-energy history
difference was `2.31e-12` Ha (LiH L-BFGS); H2 finite-shot histories were identical
for both legacy optimizers at seed 20261019 and 1,000 shots per word.
These short runs verify implementation behavior, not statistical superiority
or full convergence.

For the deterministic six-spatial-orbital, four-electron benchmark state,
11 timed evaluations after warm-up gave these median milliseconds:

| Threads | Direct RDM | Exact Pauli RDM | Finite-shot RDM (1,000/word) | Plan setup |
| --- | ---: | ---: | ---: | ---: |
| 1 | 17.86 | 15.09 | 15.85 | 48.30 |
| 4 | 5.40 | 4.30 | 4.93 | 47.76 |

The maximum exact RDM discrepancy was `8.88e-16`. Plan setup is a one-time
engine cost, and state preparation is excluded from all three RDM timings.
Raw logs, configs, DBs, benchmark JSON and `validation_summary.json` are stored
under the ignored local directory `local_runs/newton_validation`.
