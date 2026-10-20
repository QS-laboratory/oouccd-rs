//! Joint whole-sweep parametric bootstrap and projected-Beta diagnostics.
//! Synthetic replicates never count as additional quantum measurements.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use ndarray::{Array2, Array4};
use oouccd_rs::chemistry::engine::ClassicalEngine;
use oouccd_rs::config::{OrbitalOptimizerConfig, OrbitalSolverConfig};
use oouccd_rs::optimizer::orbital::OrbitalProblem;
use oouccd_rs::optimizer::risk_proposal::RiskAwareProposal;
use oouccd_rs::optimizer::solver::build_orbital_optimizer;
use oouccd_rs::optimizer::statistical_oo::rotation_matrix;
use oouccd_rs::optimizer::uncertainty::posterior_draw;
use oouccd_rs::quantum::rdm_measurement::RdmMeasurementPlan;
use oouccd_rs::quantum::rdm_statistics::{heldout_upper_bound, AffineConstraints, PauliSamples};
use serde::Deserialize;
use serde_json::json;
use std::{
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long)]
    data: PathBuf,
    #[arg(long, default_value_t = 0)]
    index: usize,
    #[arg(long, default_value_t = 256)]
    draws: usize,
    #[arg(long, default_value_t = 90261020)]
    seed: u64,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Deserialize)]
struct Snapshot {
    n_spatial: usize,
    pairs: Vec<(usize, usize)>,
    plan: RdmMeasurementPlan,
    constraints: AffineConstraints,
    h1: Array2<f64>,
    h2: Array4<f64>,
    nuclear_repulsion: f64,
    exact_means: Vec<f64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.output.exists() && args.draws > 0,
        "Invalid bootstrap request or existing output"
    );
    let snapshot: Snapshot = serde_json::from_slice(&fs::read(&args.snapshot)?)?;
    let row = BufReader::new(fs::File::open(&args.data)?)
        .lines()
        .filter_map(|line| line.ok())
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(&line).ok())
        .find(|row| {
            row["record_type"] == "trial" && row["index"].as_u64() == Some(args.index as u64)
        })
        .context("Missing original measurement batch")?;
    let shots = row["counts"]["shots"]
        .as_u64()
        .context("Missing original shots")?;
    let counts: Vec<u64> = serde_json::from_value(row["counts"]["plus_counts"].clone())?;
    let raw = PauliSamples::from_counts(shots, counts);
    let (training, validation) = raw.split(0.8, args.seed ^ 0xd1b54a32d192ed03)?;
    let plan = &snapshot.plan;
    let engine = ClassicalEngine {
        h1: snapshot.h1,
        h2: snapshot.h2,
        nuclear_repulsion: snapshot.nuclear_repulsion,
    };
    let (exact_one, exact_two) = plan.reconstruct(&snapshot.exact_means);
    let exact_before = engine.evaluate_energy(&exact_one, &exact_two);
    let before_weights = plan.energy_weights(&engine.h1, &engine.h2);
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
    let mut writer = BufWriter::new(fs::File::create(&args.output)?);
    writeln!(
        writer,
        "{}",
        json!({"record_type":"metadata","snapshot":args.snapshot,"data":args.data,
        "original_index":args.index,"draws":args.draws,"input_shots":shots,"training_shots":training.shots,
        "validation_shots":validation.shots,"modeled_additional_quantum_shots":0,"seed":args.seed,
        "interpretation":"Full adaptive sweep replay. Projected Beta draws are diagnostic, not a physically constrained posterior or a coverage guarantee."})
    )?;
    let correction = snapshot.constraints.correct(&training, plan)?;
    let training_shots = training.shots.unwrap();
    let smoothed: Vec<_> = training
        .plus_counts
        .iter()
        .zip(plan.pauli_words())
        .map(|(&k, w)| {
            if w.x_mask | w.y_mask | w.z_mask == 0 {
                1.
            } else {
                2. * (k as f64 + 0.5) / (training_shots as f64 + 1.) - 1.
            }
        })
        .collect();
    for (method, base) in &solvers {
        for affine in [false, true] {
            for proposal_kind in ["nominal", "robust", "posterior_mean"] {
                let robust = proposal_kind == "robust";
                let original_means = if proposal_kind == "posterior_mean" {
                    if affine {
                        correction.apply_to(&smoothed)
                    } else {
                        smoothed.clone()
                    }
                } else if affine {
                    correction.means.clone()
                } else {
                    training.means.clone()
                };
                let (original_one, original_two) = plan.reconstruct(&original_means);
                let problem = OrbitalProblem {
                    engine: &engine,
                    rdm1: &original_one,
                    rdm2: &original_two,
                    pairs: &snapshot.pairs,
                };
                let risk = RiskAwareProposal {
                    base: base.as_ref(),
                    sequential: *method == "seq",
                    plan,
                    variances: training.variances(plan),
                    affine: if affine { Some(&correction) } else { None },
                    beta: 2.,
                };
                let proposer: &dyn oouccd_rs::optimizer::orbital::OrbitalOptimizer =
                    if robust { &risk } else { base.as_ref() };
                let original = proposer.optimize(&problem)?;
                let original_rotation =
                    rotation_matrix(&original, snapshot.n_spatial, &snapshot.pairs);
                let (oh1, oh2) = engine.rotated_integrals(&original_rotation);
                let original_weights = plan.energy_weights(&oh1, &oh2);
                let delta_weights: Vec<_> = original_weights
                    .iter()
                    .zip(&before_weights)
                    .map(|(a, b)| a - b)
                    .collect();
                let (validation_delta, validation_bound) =
                    heldout_upper_bound(plan, &validation, &delta_weights, 0.005)?;
                writeln!(
                    writer,
                    "{}",
                    json!({"record_type":"proposal","method":method,"affine":affine,"robust":robust,"proposal_kind":proposal_kind,
            "training_delta_ha":original.energy_after-original.energy_before,"validation_delta_ha":validation_delta,
            "validation_upper_bound_ha":validation_bound,"heldout_accepted":validation_bound<0.,
            "oracle_delta_ha":engine.nuclear_repulsion+(&oh1*&exact_one).sum()+(&oh2*&exact_two).sum()-exact_before,
            "rotation":original_rotation})
                )?;
                for scheme in ["parametric_bootstrap", "projected_beta"] {
                    for draw in 0..args.draws {
                        let start = Instant::now();
                        let seed = args.seed.wrapping_add(draw as u64 * 1000003);
                        let (synthetic, latent) = if scheme == "parametric_bootstrap" {
                            let batch =
                                plan.sample_expectations(&smoothed, training_shots as usize, seed);
                            let means = batch.means.clone();
                            (Some(batch), means)
                        } else {
                            (None, posterior_draw(&training, plan, seed)?)
                        };
                        let generated_correction = if affine {
                            synthetic
                                .as_ref()
                                .map(|batch| snapshot.constraints.correct(batch, plan))
                                .transpose()?
                        } else {
                            None
                        };
                        let fitted = if affine {
                            Some(generated_correction.as_ref().unwrap_or(&correction))
                        } else {
                            None
                        };
                        let latent = if proposal_kind == "posterior_mean" && synthetic.is_some() {
                            let batch = synthetic.as_ref().unwrap();
                            batch
                                .plus_counts
                                .iter()
                                .zip(plan.pauli_words())
                                .map(|(&k, w)| {
                                    if w.x_mask | w.y_mask | w.z_mask == 0 {
                                        1.0
                                    } else {
                                        2.0 * (k as f64 + 0.5) / (training_shots as f64 + 1.0) - 1.0
                                    }
                                })
                                .collect()
                        } else {
                            latent
                        };
                        let means = if affine {
                            if synthetic.is_some() {
                                if proposal_kind == "posterior_mean" {
                                    fitted.unwrap().apply_to(&latent)
                                } else {
                                    fitted.unwrap().means.clone()
                                }
                            } else {
                                fitted.unwrap().apply_to(&latent)
                            }
                        } else {
                            latent.clone()
                        };
                        let (one, two) = plan.reconstruct(&means);
                        let p = OrbitalProblem {
                            engine: &engine,
                            rdm1: &one,
                            rdm2: &two,
                            pairs: &snapshot.pairs,
                        };
                        let r = RiskAwareProposal {
                            base: base.as_ref(),
                            sequential: *method == "seq",
                            plan,
                            variances: synthetic.as_ref().unwrap_or(&training).variances(plan),
                            affine: fitted,
                            beta: 2.,
                        };
                        let solver: &dyn oouccd_rs::optimizer::orbital::OrbitalOptimizer =
                            if robust { &r } else { base.as_ref() };
                        let update = solver.optimize(&p)?;
                        let rotation =
                            rotation_matrix(&update, snapshot.n_spatial, &snapshot.pairs);
                        let (h1, h2) = engine.rotated_integrals(&rotation);
                        let oracle = engine.nuclear_repulsion
                            + (&h1 * &exact_one).sum()
                            + (&h2 * &exact_two).sum()
                            - exact_before;
                        let fixed_proposal_delta = (&oh1 * &one).sum() + (&oh2 * &two).sum()
                            - (&engine.h1 * &one).sum()
                            - (&engine.h2 * &two).sum();
                        writeln!(
                            writer,
                            "{}",
                            json!({"record_type":"draw","method":method,"affine":affine,"robust":robust,"proposal_kind":proposal_kind,
                    "scheme":scheme,"index":draw,"seed":seed,"rotation":rotation,
                    "replayed_draw_delta_ha":update.energy_after-update.energy_before,
                    "fixed_proposal_draw_delta_ha":fixed_proposal_delta,"oracle_delta_ha":oracle,
                    "seconds":start.elapsed().as_secs_f64()})
                        )?;
                    }
                }
            }
        }
    }
    writer.flush()?;
    fs::write(
        args.output.with_extension("completed.json"),
        serde_json::to_vec(
            &json!({"status":"completed","draws":args.draws,"original_quantum_shots_per_word":shots}),
        )?,
    )?;
    Ok(())
}
