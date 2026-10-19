//! RDM-only timings: state preparation is intentionally excluded.
use anyhow::{ensure, Result};
use clap::Parser;
use num_complex::Complex64;
use oouccd_rs::chemistry::rdm::{compute_1rdm_spatial, compute_2rdm_spatial};
use oouccd_rs::quantum::rdm_measurement::RdmMeasurementPlan;
use serde::Serialize;
use std::{hint::black_box, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 6)]
    spatial: usize,
    #[arg(long, default_value_t = 4)]
    electrons: u32,
    #[arg(long, default_value_t = 11)]
    repetitions: usize,
    #[arg(long, default_value_t = 1000)]
    shots: usize,
}

#[derive(Serialize)]
struct Report {
    spatial_orbitals: usize,
    electrons: u32,
    rayon_threads: usize,
    repetitions: usize,
    shots_per_pauli: usize,
    measured_pauli_words: usize,
    plan_setup_ms: f64,
    direct_median_ms: f64,
    exact_pauli_median_ms: f64,
    finite_shot_median_ms: f64,
    max_exact_rdm_error: f64,
}

fn elapsed_ms<T>(f: impl FnOnce() -> T) -> f64 {
    let start = Instant::now();
    black_box(f());
    start.elapsed().as_secs_f64() * 1000.0
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.spatial > 0 && 2 * args.spatial < usize::BITS as usize,
        "Invalid spatial orbital count"
    );
    ensure!(
        args.electrons as usize <= 2 * args.spatial,
        "Too many electrons"
    );
    ensure!(
        args.repetitions > 0 && args.shots > 0,
        "Repetitions and shots must be positive"
    );
    // Deterministic normalized real state in the fixed-electron sector.
    let mut state: Vec<_> = (0usize..1 << (2 * args.spatial))
        .map(|i| {
            Complex64::new(
                if i.count_ones() == args.electrons {
                    ((i + 1) as f64).sin()
                } else {
                    0.0
                },
                0.0,
            )
        })
        .collect();
    let norm = state.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    state.iter_mut().for_each(|v| *v /= norm);
    let start = Instant::now();
    let plan = RdmMeasurementPlan::new(args.spatial);
    let setup = start.elapsed().as_secs_f64() * 1000.0;
    let direct = || {
        (
            compute_1rdm_spatial(black_box(&state), args.spatial),
            compute_2rdm_spatial(black_box(&state), args.spatial),
        )
    };
    let exact = || plan.evaluate_exact(black_box(&state));
    let finite = || plan.measure(black_box(&state), args.shots, 20261019);
    let reference = direct();
    let actual = exact();
    let error = reference
        .0
        .iter()
        .zip(actual.0.iter())
        .chain(reference.1.iter().zip(actual.1.iter()))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    ensure!(error < 1e-12, "Exact RDM error: {error}");
    for _ in 0..3 {
        black_box(direct());
        black_box(exact());
        black_box(finite());
    }
    let mut direct_times = Vec::new();
    let mut exact_times = Vec::new();
    let mut finite_times = Vec::new();
    for _ in 0..args.repetitions {
        direct_times.push(elapsed_ms(direct));
        exact_times.push(elapsed_ms(exact));
        finite_times.push(elapsed_ms(finite));
    }
    let report = Report {
        spatial_orbitals: args.spatial,
        electrons: args.electrons,
        rayon_threads: rayon::current_num_threads(),
        repetitions: args.repetitions,
        shots_per_pauli: args.shots,
        measured_pauli_words: plan.number_measured_pauli_words(),
        plan_setup_ms: setup,
        direct_median_ms: median(direct_times),
        exact_pauli_median_ms: median(exact_times),
        finite_shot_median_ms: median(finite_times),
        max_exact_rdm_error: error,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
