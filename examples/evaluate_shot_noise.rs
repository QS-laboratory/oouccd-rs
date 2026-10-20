//! One fixed-budget noisy OO trajectory, with exact diagnostics excluded from optimization.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use ndarray::{Array2, Array4};
use oouccd_rs::chemistry::pyscf_driver::PyScfDriver;
use oouccd_rs::config::{RdmEvaluationConfig, TaskConfig};
use oouccd_rs::optimizer::orbital::OrbitalRotation;
use oouccd_rs::optimizer::solver::Optimizer;
use oouccd_rs::optimizer::statistical_oo::OoStatisticsRecord;
use oouccd_rs::optimizer::utils::givens_rotate_2d;
use oouccd_rs::quantum::rdm_measurement::RdmMeasurementPlan;
use serde::Serialize;
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Controlled experiment: keep UCC energy evaluations exact while OO remains noisy.
    #[arg(long)]
    exact_ucc: bool,
    /// Development-only comparison: apply the joint affine correction to UCC
    /// energy evaluations too. Estimated weights can introduce finite-shot bias.
    #[arg(long, conflicts_with = "exact_ucc")]
    affine_ucc: bool,
}

#[derive(Serialize)]
struct Iteration {
    macro_iteration: usize,
    measured_energy_ha: f64,
    exact_energy_ha: f64,
    rdm_evaluations: u64,
    modeled_shots: u64,
    elapsed_seconds: f64,
    theta: Vec<f64>,
    orbital_rotation: Vec<Vec<f64>>,
    orthogonality_error: f64,
    oo: Option<OoRecord>,
}

#[derive(Serialize)]
struct OoRecord {
    proposed_exact_energy_after_ha: f64,
    statistics: OoStatisticsRecord,
    measured_energy_before_ha: f64,
    measured_energy_after_ha: f64,
    exact_energy_before_ha: f64,
    exact_energy_after_ha: f64,
    gradient_norm: Option<f64>,
    hessian_shift: Option<f64>,
    step_norm: f64,
    line_search_scale: Option<f64>,
    status: String,
}

#[derive(Serialize)]
struct Report {
    ucc_mitigation: String,
    ucc_measurement_mode: String,
    task_name: String,
    method: String,
    status: String,
    error: Option<String>,
    failed_macro_iteration: Option<usize>,
    unique_measured_pauli_words: usize,
    shots_per_pauli_word: Option<usize>,
    diagnostic_rdm_evaluations: usize,
    rdm_evaluations: u64,
    modeled_shots: u64,
    iterations: Vec<Iteration>,
}

fn matrix_rows(matrix: &Array2<f64>) -> Vec<Vec<f64>> {
    matrix.rows().into_iter().map(|r| r.to_vec()).collect()
}

fn save(path: &PathBuf, report: &Report) -> Result<()> {
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(report)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn exact_rdms(optimizer: &Optimizer, plan: &RdmMeasurementPlan) -> (Array2<f64>, Array4<f64>) {
    let state = optimizer
        .quantum_engine
        .get_state_vector(optimizer.vqe_params.as_slice().unwrap());
    plan.evaluate_exact(&state)
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.output.exists(),
        "Refusing to overwrite {}",
        args.output.display()
    );
    let config: TaskConfig = serde_json::from_slice(&fs::read(&args.config)?)?;
    let context = PyScfDriver::compute(&config.chemistry)?;
    let mut optimizer = Optimizer::build(&context, &config.method)?;
    ensure!(
        optimizer.orb_opt.is_some(),
        "This evaluator requires an orbital optimizer"
    );
    let plan = RdmMeasurementPlan::new(context.n_orb);
    let shots = match config.method.rdm_evaluation {
        RdmEvaluationConfig::Statevector {} => None,
        RdmEvaluationConfig::FiniteShot { shots, .. } => Some(shots),
    };
    let count_shots = |evaluations: u64| {
        evaluations * plan.number_measured_pauli_words() as u64 * shots.unwrap_or(0) as u64
    };
    let mut report = Report {
        ucc_mitigation: if args.affine_ucc { "affine" } else { "none" }.into(),
        ucc_measurement_mode: if args.exact_ucc {
            "exact"
        } else {
            "configured"
        }
        .into(),
        task_name: config.task_name,
        method: config.method.label,
        status: "running".into(),
        error: None,
        failed_macro_iteration: None,
        unique_measured_pauli_words: plan.number_measured_pauli_words(),
        shots_per_pauli_word: shots,
        diagnostic_rdm_evaluations: 1,
        rdm_evaluations: 1,
        modeled_shots: count_shots(1),
        iterations: Vec::new(),
    };
    let started = Instant::now();
    let mut rotation = Array2::<f64>::eye(context.n_orb);
    let (one, two) = optimizer
        .quantum_engine
        .get_rdm(optimizer.vqe_params.as_slice().unwrap());
    let measured = optimizer.classical_engine.evaluate_energy(&one, &two);
    let mut exact = exact_rdms(&optimizer, &plan);
    let exact_energy = optimizer
        .classical_engine
        .evaluate_energy(&exact.0, &exact.1);
    report.iterations.push(Iteration {
        macro_iteration: 0,
        measured_energy_ha: measured,
        exact_energy_ha: exact_energy,
        rdm_evaluations: 1,
        modeled_shots: count_shots(1),
        elapsed_seconds: started.elapsed().as_secs_f64(),
        theta: optimizer.vqe_params.to_vec(),
        orbital_rotation: matrix_rows(&rotation),
        orthogonality_error: 0.0,
        oo: None,
    });
    save(&args.output, &report)?;
    for macro_iteration in 1..=optimizer.max_macro_iterations {
        let result: Result<Iteration> = (|| {
            let samples = optimizer
                .quantum_engine
                .get_pauli_samples(optimizer.vqe_params.as_slice().unwrap());
            report.rdm_evaluations += 1;
            let before_exact = optimizer
                .classical_engine
                .evaluate_energy(&exact.0, &exact.1);
            let (update, statistics) =
                optimizer.orbital_update_from_samples(&samples, macro_iteration)?;
            let proposed_rotation =
                Array2::from_shape_fn((context.n_orb, context.n_orb), |(i, j)| {
                    statistics.proposed_rotation[i][j]
                });
            let (proposed_h1, proposed_h2) = optimizer
                .classical_engine
                .rotated_integrals(&proposed_rotation);
            let proposed_exact_energy_after_ha = optimizer.classical_engine.nuclear_repulsion
                + (&proposed_h1 * &exact.0).sum()
                + (&proposed_h2 * &exact.1).sum();
            let step_norm = match &update.rotation {
                OrbitalRotation::Givens(angles) => {
                    for (&(i, j), &angle) in optimizer.oo_params_pairs.iter().zip(angles) {
                        givens_rotate_2d(&mut rotation, i, j, 0, angle);
                    }
                    angles.iter().map(|v| v * v).sum::<f64>().sqrt()
                }
                OrbitalRotation::Matrix(matrix) => {
                    rotation = matrix.dot(&rotation);
                    update.newton.as_ref().map_or(0.0, |r| r.step_norm)
                }
                OrbitalRotation::Unchanged => 0.0,
            };
            update.apply(&mut optimizer.classical_engine, &optimizer.oo_params_pairs);
            let identity_error = rotation.dot(&rotation.t()) - Array2::<f64>::eye(context.n_orb);
            let orthogonality_error = identity_error.iter().map(|v| v.abs()).fold(0.0, f64::max);
            ensure!(
                orthogonality_error < 1e-8,
                "Orbital rotation lost orthogonality: {orthogonality_error}"
            );
            let after_exact = optimizer
                .classical_engine
                .evaluate_energy(&exact.0, &exact.1);
            let oo = OoRecord {
                proposed_exact_energy_after_ha,
                statistics,
                measured_energy_before_ha: update.energy_before,
                measured_energy_after_ha: update.energy_after,
                exact_energy_before_ha: before_exact,
                exact_energy_after_ha: after_exact,
                gradient_norm: update.newton.as_ref().map(|r| r.gradient_norm),
                hessian_shift: update.newton.as_ref().map(|r| r.hessian_shift),
                step_norm,
                line_search_scale: update.newton.as_ref().map(|r| r.line_search_scale),
                status: update
                    .newton
                    .as_ref()
                    .map_or_else(|| "GivensSweep".into(), |r| format!("{:?}", r.status)),
            };
            let quantum = &optimizer.quantum_engine;
            let classical = &optimizer.classical_engine;
            let evaluations = &mut report.rdm_evaluations;
            let constraints = &optimizer.affine_constraints;
            let mut correction_error = None;
            let mut objective = |theta: &[f64]| {
                let (g1, g2) = if args.exact_ucc {
                    plan.evaluate_exact(&quantum.get_state_vector(theta))
                } else if args.affine_ucc {
                    let batch = quantum.get_pauli_samples(theta);
                    let means = match constraints.correct(&batch, &plan) {
                        Ok(correction) => correction.means,
                        Err(error) => {
                            // Finish the solver callback with a finite value,
                            // then fail this trajectory explicitly below.
                            correction_error = Some(error);
                            batch.means
                        }
                    };
                    plan.reconstruct(&means)
                } else {
                    quantum.get_rdm(theta)
                };
                *evaluations += 1;
                classical.evaluate_energy(&g1, &g2)
            };
            optimizer
                .ucc_opt
                .optimize(&mut optimizer.vqe_params, &mut objective);
            if let Some(error) = correction_error {
                return Err(error).context("UCC affine reconstruction failed");
            }
            ensure!(
                optimizer.vqe_params.iter().all(|v| v.is_finite()),
                "Non-finite UCC parameters"
            );
            let (one, two) = optimizer
                .quantum_engine
                .get_rdm(optimizer.vqe_params.as_slice().unwrap());
            report.rdm_evaluations += 1;
            let measured = optimizer.classical_engine.evaluate_energy(&one, &two);
            exact = exact_rdms(&optimizer, &plan);
            report.diagnostic_rdm_evaluations += 1;
            let exact_energy = optimizer
                .classical_engine
                .evaluate_energy(&exact.0, &exact.1);
            ensure!(
                measured.is_finite() && exact_energy.is_finite(),
                "Non-finite endpoint energy"
            );
            Ok(Iteration {
                macro_iteration,
                measured_energy_ha: measured,
                exact_energy_ha: exact_energy,
                rdm_evaluations: report.rdm_evaluations,
                modeled_shots: count_shots(optimizer.quantum_engine.sampled_batch_count()),
                elapsed_seconds: started.elapsed().as_secs_f64(),
                theta: optimizer.vqe_params.to_vec(),
                orbital_rotation: matrix_rows(&rotation),
                orthogonality_error,
                oo: Some(oo),
            })
        })();
        match result {
            Ok(iteration) => {
                eprintln!(
                    "macro={macro_iteration} measured={:.12} exact={:.12} rdm_evaluations={}",
                    iteration.measured_energy_ha,
                    iteration.exact_energy_ha,
                    iteration.rdm_evaluations
                );
                report.iterations.push(iteration);
            }
            Err(error) => {
                report.status = "failed".into();
                report.error = Some(format!("{error:#}"));
                report.failed_macro_iteration = Some(macro_iteration);
                report.modeled_shots = count_shots(optimizer.quantum_engine.sampled_batch_count());
                save(&args.output, &report).context("Saving a failed trajectory")?;
                return Ok(());
            }
        }
        report.modeled_shots = report
            .iterations
            .last()
            .map_or(0, |iteration| iteration.modeled_shots);
        save(&args.output, &report)?;
    }
    report.status = "completed".into();
    save(&args.output, &report)
}
