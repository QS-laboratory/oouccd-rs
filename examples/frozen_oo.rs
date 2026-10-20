//! Export existing physical states, then compare OO algorithms on identical counts.
use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use ndarray::{Array2, Array4};
use oouccd_rs::chemistry::engine::ClassicalEngine;
use oouccd_rs::chemistry::pyscf_driver::PyScfDriver;
use oouccd_rs::config::{
    OoAcceptance, OrbitalOptimizerConfig, OrbitalSolverConfig, OrbitalStatisticsConfig,
    RdmMitigation, TaskConfig,
};
use oouccd_rs::optimizer::orbital::OrbitalRotation;
use oouccd_rs::optimizer::solver::{build_orbital_optimizer, Optimizer};
use oouccd_rs::optimizer::statistical_oo::{self, StoredCounts};
use oouccd_rs::optimizer::uncertainty::{self, FourierProfile};
use oouccd_rs::quantum::rdm_measurement::{RdmMeasurementPlan, SpinRdmPlans};
use oouccd_rs::quantum::rdm_statistics::{covariance, AffineConstraints};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Assess the actual OO updates induced by saved physical reconstructions.
    Reconstructed {
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long)]
        reconstruction: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Export {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        trajectory: PathBuf,
        #[arg(long)]
        iteration: usize,
        #[arg(long)]
        output: PathBuf,
    },
    Study {
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long)]
        shots: usize,
        #[arg(long, default_value_t = 1000)]
        replicates: usize,
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Serialize, Deserialize)]
struct Snapshot {
    n_elec: usize,
    n_spatial: usize,
    pairs: Vec<(usize, usize)>,
    plan: RdmMeasurementPlan,
    spin_plans: SpinRdmPlans,
    constraints: AffineConstraints,
    h1: Array2<f64>,
    h2: Array4<f64>,
    nuclear_repulsion: f64,
    exact_means: Vec<f64>,
    exact_energy_ha: f64,
    origin_trajectory: String,
    origin_iteration: usize,
}

fn export(config: PathBuf, trajectory: PathBuf, iteration: usize, output: PathBuf) -> Result<()> {
    ensure!(!output.exists(), "Refusing to overwrite snapshot");
    let task: TaskConfig = serde_json::from_slice(&fs::read(config)?)?;
    let context = PyScfDriver::compute(&task.chemistry)?;
    let mut optimizer = Optimizer::build(&context, &task.method)?;
    let native: serde_json::Value = serde_json::from_slice(&fs::read(&trajectory)?)?;
    let point = native["iterations"]
        .as_array()
        .context("Missing trajectory")?
        .iter()
        .find(|p| p["macro_iteration"].as_u64() == Some(iteration as u64))
        .context("Missing saved iteration")?;
    let theta: Vec<f64> = serde_json::from_value(point["theta"].clone())?;
    let rows: Vec<Vec<f64>> = serde_json::from_value(point["orbital_rotation"].clone())?;
    ensure!(
        rows.len() == context.n_orb && rows.iter().all(|r| r.len() == context.n_orb),
        "Invalid saved rotation"
    );
    let rotation = Array2::from_shape_fn((context.n_orb, context.n_orb), |(i, j)| rows[i][j]);
    optimizer.classical_engine.apply_orbital_rotation(&rotation);
    let plan = optimizer.quantum_engine.measurement_plan().clone();
    let exact_means = plan.exact_expectations(&optimizer.quantum_engine.get_state_vector(&theta));
    let (one, two) = plan.reconstruct(&exact_means);
    let energy = optimizer.classical_engine.evaluate_energy(&one, &two);
    ensure!(
        (energy
            - point["exact_energy_ha"]
                .as_f64()
                .context("Missing reference energy")?)
        .abs()
            < 1e-8,
        "Saved-state replay does not reproduce its exact energy"
    );
    let snapshot = Snapshot {
        n_elec: context.n_elec,
        n_spatial: context.n_orb,
        pairs: optimizer.oo_params_pairs,
        spin_plans: plan.spin_plans()?,
        constraints: AffineConstraints::new(&plan, context.n_elec),
        plan,
        h1: optimizer.classical_engine.h1,
        h2: optimizer.classical_engine.h2,
        nuclear_repulsion: optimizer.classical_engine.nuclear_repulsion,
        exact_means,
        exact_energy_ha: energy,
        origin_trajectory: trajectory.display().to_string(),
        origin_iteration: iteration,
    };
    fs::write(output, serde_json::to_vec(&snapshot)?)?;
    Ok(())
}

fn study(
    snapshot_path: PathBuf,
    shots: usize,
    replicates: usize,
    seed: u64,
    output: PathBuf,
) -> Result<()> {
    ensure!(
        !output.exists() && shots >= 2 && replicates > 0,
        "Invalid fixed-state request or existing output"
    );
    let snapshot: Snapshot = serde_json::from_slice(&fs::read(&snapshot_path)?)?;
    let engine = ClassicalEngine {
        h1: snapshot.h1,
        h2: snapshot.h2,
        nuclear_repulsion: snapshot.nuclear_repulsion,
    };
    let plan = &snapshot.plan;
    let (exact_one, exact_two) = plan.reconstruct(&snapshot.exact_means);
    let exact_energy = engine.evaluate_energy(&exact_one, &exact_two);
    let mut profile_weights = Vec::new();
    for j in 0..9 {
        let mut angles = vec![0.0; snapshot.pairs.len()];
        if !angles.is_empty() {
            angles[0] = 2.0 * std::f64::consts::PI * j as f64 / 9.0;
        }
        let (h1, h2) = engine.integrals_with_givens(&snapshot.pairs, &angles);
        profile_weights.push(plan.energy_weights(&h1, &h2));
    }
    let true_variances: Vec<_> = snapshot
        .exact_means
        .iter()
        .zip(plan.pauli_words())
        .map(|(&mu, w)| {
            if w.x_mask | w.y_mask | w.z_mask == 0 {
                0.0
            } else {
                (1.0 - mu * mu) / shots as f64
            }
        })
        .collect();
    let exact_energies: Vec<_> = profile_weights
        .iter()
        .map(|w| {
            engine.nuclear_repulsion
                + w.iter()
                    .zip(&snapshot.exact_means)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
        })
        .collect();
    let true_profile =
        FourierProfile::from_measurements(&exact_energies, &profile_weights, &true_variances);
    let mut writer = BufWriter::new(fs::File::create(&output)?);
    writeln!(
        writer,
        "{}",
        json!({"record_type":"metadata","snapshot":snapshot_path,"shots":shots,
        "replicates":replicates,"base_seed":seed,"exact_energy_ha":exact_energy,
        "profile_exact_coefficients":true_profile.coefficients,"profile_true_covariance":true_profile.covariance,
        "energy_true_covariance":covariance(&profile_weights,&true_variances),
        "measurement_model":"Independent Pauli words, shared RDM and shared counts across all variants"})
    )?;
    let solvers = [
        (
            "seq",
            build_orbital_optimizer(&OrbitalOptimizerConfig {
                solver_config: OrbitalSolverConfig::Sequential {},
            })?,
        ),
        (
            "newton_stable",
            build_orbital_optimizer(&OrbitalOptimizerConfig {
                solver_config: OrbitalSolverConfig::NewtonRaphson {
                    options: Default::default(),
                },
            })?,
        ),
    ];
    for repetition in 0..replicates {
        let trial_seed = seed
            .checked_add(repetition as u64)
            .context("Seed overflow")?;
        let samples = plan.sample_expectations(&snapshot.exact_means, shots, trial_seed);
        let energies: Vec<_> = profile_weights
            .iter()
            .map(|w| engine.nuclear_repulsion + samples.dot(w))
            .collect();
        let profile = FourierProfile::from_measurements(
            &energies,
            &profile_weights,
            &samples.variances(plan),
        );
        let mut outcomes = Vec::new();
        for (method, solver) in &solvers {
            for mitigation in [RdmMitigation::None, RdmMitigation::Affine] {
                for acceptance in [
                    OoAcceptance::Nominal,
                    OoAcceptance::SplitNominal,
                    OoAcceptance::Heldout,
                ] {
                    let config = OrbitalStatisticsConfig {
                        mitigation,
                        acceptance,
                        ..Default::default()
                    };
                    let start = Instant::now();
                    let outcome = statistical_oo::optimize(
                        solver.as_ref(),
                        &engine,
                        &snapshot.pairs,
                        plan,
                        &samples,
                        &snapshot.constraints,
                        &config,
                        10,
                        1,
                    );
                    let label = format!("{method}_{mitigation:?}_{acceptance:?}").to_lowercase();
                    match outcome {
                    Ok((update,record))=>{
                        let r=Array2::from_shape_fn((snapshot.n_spatial,snapshot.n_spatial),|(i,j)|record.proposed_rotation[i][j]);
                        let (h1,h2)=engine.rotated_integrals(&r);
                        let proposed=engine.nuclear_repulsion+(&h1*&exact_one).sum()+(&h2*&exact_two).sum();
                        let applied=match update.rotation {OrbitalRotation::Unchanged=>exact_energy,_=>proposed};
                        outcomes.push(json!({"variant":label,"method":method,"mitigation":mitigation,"acceptance":acceptance,
                            "status":"completed","proposed_exact_delta_ha":proposed-exact_energy,
                            "applied_exact_delta_ha":applied-exact_energy,"statistics":record}));
                    }
                    Err(error)=>outcomes.push(json!({"variant":label,"method":method,"mitigation":mitigation,"acceptance":acceptance,
                        "status":"failed","error":format!("{error:#}"),"seconds":start.elapsed().as_secs_f64()})),
                }
                }
            }
        }
        let raw_rdm = plan.reconstruct(&samples.means);
        let correction = snapshot.constraints.correct(&samples, plan)?;
        let corrected_rdm = plan.reconstruct(&correction.means);
        let rdm_squared_error = |r: &(Array2<f64>, Array4<f64>)| {
            r.0.iter()
                .zip(&exact_one)
                .chain(r.1.iter().zip(&exact_two))
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
        };
        let posterior =
            uncertainty::posterior_draw(&samples, plan, trial_seed ^ 0x94d049bb133111eb)?;
        let raw_norm = rdm_squared_error(&raw_rdm);
        writeln!(
            writer,
            "{}",
            json!({"record_type":"trial","index":repetition,"seed":trial_seed,
            "counts":StoredCounts {shots:shots as u64,plus_counts:samples.plus_counts.clone(),sampling_seed:samples.sampling_seed},
            "profile_energies":energies,"profile_coefficients":profile.coefficients,
            "profile_estimated_covariance":profile.covariance,"robust_candidate":profile.robust_minimum(2.0,128)?,
            "posterior_example_energy_ha":engine.nuclear_repulsion+plan.energy_weights(&engine.h1,&engine.h2).iter()
                .zip(posterior).map(|(a,b)|a*b).sum::<f64>(),
            "rdm_raw_squared_error":raw_norm,"rdm_affine_squared_error":rdm_squared_error(&corrected_rdm),
            "raw_energy_error_ha":engine.evaluate_energy(&raw_rdm.0,&raw_rdm.1)-exact_energy,
            "affine_energy_error_ha":engine.evaluate_energy(&corrected_rdm.0,&corrected_rdm.1)-exact_energy,
            "outcomes":outcomes})
        )?;
        if repetition % 25 == 0 {
            writer.flush()?;
        }
    }
    writer.flush()?;
    fs::write(
        output.with_extension("completed.json"),
        serde_json::to_vec(&json!({"status":"completed","replicates":replicates}))?,
    )?;
    Ok(())
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Reconstructed {
            snapshot,
            reconstruction,
            output,
        } => reconstructed(snapshot, reconstruction, output),
        Command::Export {
            config,
            trajectory,
            iteration,
            output,
        } => export(config, trajectory, iteration, output),
        Command::Study {
            snapshot,
            shots,
            replicates,
            seed,
            output,
        } => study(snapshot, shots, replicates, seed, output),
    }
}

fn reconstructed(snapshot_path: PathBuf, data: PathBuf, output: PathBuf) -> Result<()> {
    ensure!(
        !output.exists(),
        "Refusing to overwrite reconstruction OO results"
    );
    let snapshot: Snapshot = serde_json::from_slice(&fs::read(&snapshot_path)?)?;
    let engine = ClassicalEngine {
        h1: snapshot.h1,
        h2: snapshot.h2,
        nuclear_repulsion: snapshot.nuclear_repulsion,
    };
    let (exact_one, exact_two) = snapshot.plan.reconstruct(&snapshot.exact_means);
    let before = engine.evaluate_energy(&exact_one, &exact_two);
    let solvers = [
        (
            "seq",
            build_orbital_optimizer(&OrbitalOptimizerConfig {
                solver_config: OrbitalSolverConfig::Sequential {},
            })?,
        ),
        (
            "newton_stable",
            build_orbital_optimizer(&OrbitalOptimizerConfig {
                solver_config: OrbitalSolverConfig::NewtonRaphson {
                    options: Default::default(),
                },
            })?,
        ),
    ];
    let mut writer = BufWriter::new(fs::File::create(&output)?);
    writeln!(
        writer,
        "{}",
        json!({"record_type":"metadata","snapshot":snapshot_path,"reconstruction":data,
        "additional_quantum_shots":0,"acceptance":"nominal_full_data",
        "interpretation":"Reconstructions used all counts; no held-out safety claim is made. Exact RDMs are diagnostic only."})
    )?;
    let content = fs::read_to_string(&data)?;
    let mut trials = 0;
    for line in content.lines() {
        let row: serde_json::Value = serde_json::from_str(line)?;
        if row["record_type"] != "trial" {
            continue;
        }
        let mut outcomes = Vec::new();
        for fit in row["results"]
            .as_array()
            .context("Missing reconstruction results")?
        {
            if fit["completed"] != true {
                for (solver_name, _) in &solvers {
                    outcomes.push(json!({"method":fit["method"],"solver":solver_name,
                        "status":"fit_failure","error":fit["error"]}));
                }
                continue;
            }
            let means: Vec<f64> = serde_json::from_value(fit["means"].clone())?;
            ensure!(
                means.iter().all(|x| x.is_finite()),
                "Nonfinite reconstructed expectations"
            );
            let (one, two) = snapshot.plan.reconstruct(&means);
            let problem = oouccd_rs::optimizer::orbital::OrbitalProblem {
                engine: &engine,
                rdm1: &one,
                rdm2: &two,
                pairs: &snapshot.pairs,
            };
            for (solver_name, solver) in &solvers {
                let started = Instant::now();
                match solver.optimize(&problem) {
                    Ok(update) => {
                        let seconds = started.elapsed().as_secs_f64();
                        let rotation = statistical_oo::rotation_matrix(
                            &update,
                            snapshot.n_spatial,
                            &snapshot.pairs,
                        );
                        let (h1, h2) = engine.rotated_integrals(&rotation);
                        let after = engine.nuclear_repulsion
                            + (&h1 * &exact_one).sum()
                            + (&h2 * &exact_two).sum();
                        outcomes.push(json!({"method":fit["method"],"solver":solver_name,"status":"completed",
                            "exact_delta_ha":after-before,"estimator_delta_ha":update.energy_after-update.energy_before,
                            "classical_oo_seconds":seconds}));
                    }
                    Err(error) => outcomes
                        .push(json!({"method":fit["method"],"solver":solver_name,
                        "status":"optimizer_failure","error":format!("{error:#}")})),
                }
            }
        }
        writeln!(
            writer,
            "{}",
            json!({"record_type":"trial","index":row["index"],"seed":row["seed"],"outcomes":outcomes})
        )?;
        trials += 1;
    }
    writer.flush()?;
    fs::write(
        output.with_extension("completed.json"),
        serde_json::to_vec(&json!({
        "status":"completed","trials":trials,"additional_quantum_shots":0}))?,
    )?;
    Ok(())
}
