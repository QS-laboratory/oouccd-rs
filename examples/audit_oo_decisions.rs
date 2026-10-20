//! Replay OO decisions without measurements; optionally recompute ideal-state diagnostics.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use ndarray::{Array2, Array4};
use oouccd_rs::chemistry::engine::ClassicalEngine;
use oouccd_rs::config::{AnsatzConfig, RdmEvaluationConfig, TaskConfig};
use oouccd_rs::optimizer::orbital::OrbitalRotation;
use oouccd_rs::optimizer::solver::build_orbital_optimizer;
use oouccd_rs::optimizer::statistical_oo::{self, OoStatisticsRecord};
use oouccd_rs::optimizer::utils::givens_rotate_2d;
use oouccd_rs::quantum::engine::QuantumEngine;
use oouccd_rs::quantum::rdm_measurement::RdmMeasurementPlan;
use oouccd_rs::quantum::rdm_statistics::{AffineConstraints, PauliSamples};
use serde::Deserialize;
use serde_json::json;
use std::{fs, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    trajectory: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 1e-8)]
    tolerance: f64,
    #[arg(long)]
    verify_energies: bool,
    #[arg(long, requires = "verify_energies")]
    energies_only: bool,
}

#[derive(Deserialize)]
struct Snapshot {
    n_spatial: usize,
    n_elec: usize,
    pairs: Vec<(usize, usize)>,
    plan: RdmMeasurementPlan,
    h1: Array2<f64>,
    h2: Array4<f64>,
    nuclear_repulsion: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.output.exists(),
        "Refusing to overwrite a decision audit"
    );
    ensure!(
        args.tolerance.is_finite() && args.tolerance > 0.,
        "Invalid audit tolerance"
    );
    let initial: Snapshot = serde_json::from_slice(&fs::read(&args.snapshot)?)?;
    let task: TaskConfig = serde_json::from_slice(&fs::read(&args.config)?)?;
    let native: serde_json::Value = serde_json::from_slice(&fs::read(&args.trajectory)?)?;
    let solver = build_orbital_optimizer(
        task.method
            .orbital_optimizer
            .as_ref()
            .context("No orbital optimizer")?,
    )?;
    let constraints = AffineConstraints::new(&initial.plan, initial.n_elec);
    let mut engine = ClassicalEngine {
        h1: initial.h1,
        h2: initial.h2,
        nuclear_repulsion: initial.nuclear_repulsion,
    };
    let original_h1 = engine.h1.clone();
    let original_h2 = engine.h2.clone();
    let mut rotation = Array2::<f64>::eye(initial.n_spatial);
    let mut largest_rotation_error = 0f64;
    let mut largest_energy_error = 0f64;
    let mut checked = 0;
    let mut accepted = 0;
    for point in native["iterations"]
        .as_array()
        .context("Missing trajectory")?
        .iter()
        .skip(1)
    {
        if args.energies_only {
            break;
        }
        let macro_iteration = point["macro_iteration"]
            .as_u64()
            .context("Missing macro index")? as usize;
        let saved: OoStatisticsRecord = serde_json::from_value(point["oo"]["statistics"].clone())?;
        let counts = saved
            .counts
            .as_ref()
            .context("A finite-shot decision audit requires saved raw counts")?;
        let mut samples = PauliSamples::from_counts(counts.shots, counts.plus_counts.clone());
        samples.sampling_seed = counts.sampling_seed;
        let (update, replay) = statistical_oo::optimize(
            solver.as_ref(),
            &engine,
            &initial.pairs,
            &initial.plan,
            &samples,
            &constraints,
            &task.method.orbital_statistics,
            task.method.max_macro_iterations.unwrap_or(50),
            macro_iteration,
        )?;
        let rotation_error = saved
            .proposed_rotation
            .iter()
            .flatten()
            .zip(replay.proposed_rotation.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        largest_rotation_error = largest_rotation_error.max(rotation_error);
        ensure!(
            rotation_error <= args.tolerance,
            "OO proposal replay differs at macro {macro_iteration}: {rotation_error:e}"
        );
        for (a, b) in [
            (
                saved.proposal_energy_before_ha,
                replay.proposal_energy_before_ha,
            ),
            (
                saved.proposal_energy_after_ha,
                replay.proposal_energy_after_ha,
            ),
        ] {
            largest_energy_error = largest_energy_error.max((a - b).abs());
            ensure!(
                (a - b).abs() <= args.tolerance,
                "Training energy replay differs at macro {macro_iteration}"
            );
        }
        for (a, b) in [
            (saved.validation_delta_ha, replay.validation_delta_ha),
            (
                saved.validation_upper_bound_ha,
                replay.validation_upper_bound_ha,
            ),
        ] {
            match (a, b) {
                (Some(a), Some(b)) => {
                    largest_energy_error = largest_energy_error.max((a - b).abs());
                    ensure!(
                        (a - b).abs() <= args.tolerance,
                        "Validation replay differs at macro {macro_iteration}"
                    );
                }
                (None, None) => {}
                _ => anyhow::bail!("Validation policy replay differs"),
            }
        }
        ensure!(
            saved.accepted == replay.accepted && saved.reason == replay.reason,
            "Acceptance replay differs at macro {macro_iteration}"
        );
        ensure!(
            saved.split_seed == replay.split_seed
                && saved.training_shots == replay.training_shots
                && saved.validation_shots == replay.validation_shots,
            "Partition replay differs at macro {macro_iteration}"
        );
        match &update.rotation {
            OrbitalRotation::Givens(angles) => {
                for (&(i, j), &angle) in initial.pairs.iter().zip(angles) {
                    givens_rotate_2d(&mut rotation, i, j, 0, angle);
                }
            }
            OrbitalRotation::Matrix(r) => rotation = r.dot(&rotation),
            OrbitalRotation::Unchanged => {}
        }
        let saved_rotation: Vec<Vec<f64>> =
            serde_json::from_value(point["orbital_rotation"].clone())?;
        let applied_error = rotation
            .indexed_iter()
            .map(|((i, j), &v)| (v - saved_rotation[i][j]).abs())
            .fold(0., f64::max);
        largest_rotation_error = largest_rotation_error.max(applied_error);
        ensure!(
            applied_error <= args.tolerance,
            "Applied rotation replay differs at macro {macro_iteration}"
        );
        update.apply(&mut engine, &initial.pairs);
        checked += 1;
        accepted += usize::from(saved.accepted);
    }
    ensure!(
        checked > 0 || args.energies_only,
        "No recorded decisions to audit"
    );
    let mut verified_energies = 0;
    let mut largest_physical_energy_error = 0f64;
    if args.verify_energies {
        let quantum = QuantumEngine::new(
            initial.n_spatial,
            initial.n_elec,
            task.method.ucc_optimizer.ansatz_config == AnsatzConfig::Uccsd,
            RdmEvaluationConfig::Statevector {},
        );
        let original = ClassicalEngine {
            h1: original_h1,
            h2: original_h2,
            nuclear_repulsion: initial.nuclear_repulsion,
        };
        for point in native["iterations"].as_array().unwrap() {
            let theta: Vec<f64> = serde_json::from_value(point["theta"].clone())?;
            let rows: Vec<Vec<f64>> = serde_json::from_value(point["orbital_rotation"].clone())?;
            let rotation =
                Array2::from_shape_fn((initial.n_spatial, initial.n_spatial), |(i, j)| rows[i][j]);
            let (one, two) = initial
                .plan
                .evaluate_exact(&quantum.get_state_vector(&theta));
            let (h1, h2) = original.rotated_integrals(&rotation);
            let energy = original.nuclear_repulsion + (&h1 * &one).sum() + (&h2 * &two).sum();
            let saved = point["exact_energy_ha"]
                .as_f64()
                .context("Missing exact diagnostic")?;
            let error = (energy - saved).abs();
            largest_physical_energy_error = largest_physical_energy_error.max(error);
            ensure!(
                error <= args.tolerance,
                "Independent physical-energy check differs: {error:e}"
            );
            verified_energies += 1;
        }
    }
    fs::write(
        args.output,
        serde_json::to_vec_pretty(&json!({"passed":true,"checked_decisions":checked,
        "accepted":accepted,"largest_rotation_error":largest_rotation_error,"largest_energy_error_ha":largest_energy_error,
        "additional_quantum_measurements":0,"statevector_diagnostic_calls":verified_energies,
        "largest_physical_energy_error_ha":largest_physical_energy_error,
        "trajectory":args.trajectory,"config":args.config,"snapshot":args.snapshot}))?,
    )?;
    Ok(())
}
