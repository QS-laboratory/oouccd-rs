//! One fixed-budget noisy OO trajectory, with exact diagnostics excluded from optimization.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use ndarray::{Array2, Array4};
use oouccd_rs::chemistry::pyscf_driver::PyScfDriver;
use oouccd_rs::config::{RdmEvaluationConfig, TaskConfig};
use oouccd_rs::optimizer::orbital::{OrbitalProblem, OrbitalRotation};
use oouccd_rs::optimizer::solver::Optimizer;
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
            let (one, two) = optimizer
                .quantum_engine
                .get_rdm(optimizer.vqe_params.as_slice().unwrap());
            report.rdm_evaluations += 1;
            let before_exact = optimizer
                .classical_engine
                .evaluate_energy(&exact.0, &exact.1);
            let problem = OrbitalProblem {
                engine: &optimizer.classical_engine,
                rdm1: &one,
                rdm2: &two,
                pairs: &optimizer.oo_params_pairs,
            };
            let update = optimizer.orb_opt.as_ref().unwrap().optimize(&problem)?;
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
            let mut objective = |theta: &[f64]| {
                let (g1, g2) = quantum.get_rdm(theta);
                *evaluations += 1;
                classical.evaluate_energy(&g1, &g2)
            };
            optimizer
                .ucc_opt
                .optimize(&mut optimizer.vqe_params, &mut objective);
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
                modeled_shots: count_shots(report.rdm_evaluations),
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
                report.modeled_shots = count_shots(report.rdm_evaluations);
                save(&args.output, &report).context("Saving a failed trajectory")?;
                return Ok(());
            }
        }
        report.modeled_shots = count_shots(report.rdm_evaluations);
        save(&args.output, &report)?;
    }
    report.status = "completed".into();
    save(&args.output, &report)
}
