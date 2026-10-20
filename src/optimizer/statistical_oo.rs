//! A common train/validation wrapper for *complete* SOO and Newton proposals.
use crate::chemistry::engine::ClassicalEngine;
use crate::config::{OoAcceptance, OrbitalStatisticsConfig, RdmMitigation};
use crate::optimizer::orbital::{OrbitalOptimizer, OrbitalProblem, OrbitalRotation, OrbitalUpdate};
use crate::optimizer::utils::givens_rotate_2d;
use crate::quantum::rdm_measurement::RdmMeasurementPlan;
use crate::quantum::rdm_statistics::{heldout_upper_bound, AffineConstraints, PauliSamples};
use anyhow::{ensure, Result};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredCounts {
    pub shots: u64,
    pub plus_counts: Vec<u64>,
    pub sampling_seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OoStatisticsRecord {
    pub input_shots: Option<u64>,
    pub training_shots: Option<u64>,
    pub validation_shots: Option<u64>,
    pub split_seed: u64,
    pub counts: Option<StoredCounts>,
    pub alpha: f64,
    pub correction_residual_before: f64,
    pub correction_residual_after: f64,
    pub proposed_rotation: Vec<Vec<f64>>,
    pub proposal_energy_before_ha: f64,
    pub proposal_energy_after_ha: f64,
    pub validation_delta_ha: Option<f64>,
    pub validation_upper_bound_ha: Option<f64>,
    pub accepted: bool,
    pub reason: String,
    pub correction_seconds: f64,
    pub proposal_seconds: f64,
    pub validation_seconds: f64,
    pub total_seconds: f64,
}

pub fn rotation_matrix(update: &OrbitalUpdate, n: usize, pairs: &[(usize, usize)]) -> Array2<f64> {
    match &update.rotation {
        OrbitalRotation::Matrix(rotation) => rotation.clone(),
        OrbitalRotation::Givens(angles) => {
            let mut rotation = Array2::eye(n);
            for (&(i, j), &angle) in pairs.iter().zip(angles) {
                givens_rotate_2d(&mut rotation, i, j, 0, angle);
            }
            rotation
        }
        OrbitalRotation::Unchanged => Array2::eye(n),
    }
}

pub fn optimize(
    optimizer: &dyn OrbitalOptimizer,
    engine: &ClassicalEngine,
    pairs: &[(usize, usize)],
    plan: &RdmMeasurementPlan,
    samples: &PauliSamples,
    constraints: &AffineConstraints,
    config: &OrbitalStatisticsConfig,
    macro_iterations: usize,
    macro_iteration: usize,
) -> Result<(OrbitalUpdate, OoStatisticsRecord)> {
    config.validate()?;
    ensure!(
        macro_iterations > 0 && macro_iteration > 0 && macro_iteration <= macro_iterations,
        "Invalid macro-iteration budget for statistical OO"
    );
    let started = Instant::now();
    let split_seed = config.split_seed
        ^ samples.sampling_seed.unwrap_or(0)
        ^ (macro_iteration as u64).wrapping_mul(0xd1b54a32d192ed03);
    let split = config.acceptance != OoAcceptance::Nominal && samples.shots.is_some();
    let (training, validation) = if split {
        samples.split(config.training_fraction, split_seed)?
    } else {
        (samples.clone(), samples.clone())
    };
    let correction_started = Instant::now();
    let residual_before = constraints.max_residual(&training.means);
    let means = match config.mitigation {
        RdmMitigation::None => training.means.clone(),
        RdmMitigation::Affine => constraints.correct(&training, plan)?.means,
    };
    let residual_after = constraints.max_residual(&means);
    let (rdm1, rdm2) = plan.reconstruct(&means);
    let correction_seconds = correction_started.elapsed().as_secs_f64();
    let problem = OrbitalProblem {
        engine,
        rdm1: &rdm1,
        rdm2: &rdm2,
        pairs,
    };
    let proposal_started = Instant::now();
    let mut update = optimizer.optimize(&problem)?;
    let proposal_seconds = proposal_started.elapsed().as_secs_f64();
    let rotation = rotation_matrix(&update, plan.n_spatial(), pairs);
    let validation_started = Instant::now();
    let mut accepted = true;
    let mut delta = None;
    let mut upper = None;
    let mut reason = if split { "SplitNominal" } else { "Nominal" }.to_string();
    let alpha = config.trajectory_alpha / macro_iterations as f64;
    if config.acceptance == OoAcceptance::Heldout {
        // Preserve the proposal's original arithmetic and order of Givens rotations.
        let (h1, h2) = match &update.rotation {
            OrbitalRotation::Givens(angles) => {
                engine.integrals_with_givens(pairs, angles.as_slice().unwrap())
            }
            _ => engine.rotated_integrals(&rotation),
        };
        let before_weights = plan.energy_weights(&engine.h1, &engine.h2);
        let after_weights = plan.energy_weights(&h1, &h2);
        let difference: Vec<_> = after_weights
            .iter()
            .zip(&before_weights)
            .map(|(a, b)| a - b)
            .collect();
        let (observed, bound) = heldout_upper_bound(plan, &validation, &difference, alpha)?;
        ensure!(
            observed.is_finite() && bound.is_finite(),
            "Non-finite held-out OO bound"
        );
        accepted = bound < -config.min_improvement_ha;
        delta = Some(observed);
        upper = Some(bound);
        reason = if accepted {
            "HeldoutAccepted"
        } else {
            "HeldoutRejected"
        }
        .into();
    }
    let record = OoStatisticsRecord {
        input_shots: samples.shots,
        training_shots: training.shots,
        validation_shots: if split { validation.shots } else { None },
        split_seed,
        counts: if config.record_counts {
            samples.shots.map(|shots| StoredCounts {
                shots,
                plus_counts: samples.plus_counts.clone(),
                sampling_seed: samples.sampling_seed,
            })
        } else {
            None
        },
        alpha,
        correction_residual_before: residual_before,
        correction_residual_after: residual_after,
        proposed_rotation: rotation
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        proposal_energy_before_ha: update.energy_before,
        proposal_energy_after_ha: update.energy_after,
        validation_delta_ha: delta,
        validation_upper_bound_ha: upper,
        accepted,
        reason,
        correction_seconds,
        proposal_seconds,
        validation_seconds: validation_started.elapsed().as_secs_f64(),
        total_seconds: started.elapsed().as_secs_f64(),
    };
    if !accepted {
        update.rotation = OrbitalRotation::Unchanged;
        update.energy_after = update.energy_before;
    }
    Ok((update, record))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array4};

    struct FixedProposal {
        angle: f64,
    }
    impl OrbitalOptimizer for FixedProposal {
        fn optimize(&self, p: &OrbitalProblem<'_>) -> Result<OrbitalUpdate> {
            let angles = Array1::from(vec![self.angle]);
            Ok(OrbitalUpdate {
                energy_before: p.energy(),
                energy_after: p.energy_with_givens(angles.as_slice().unwrap()),
                rotation: OrbitalRotation::Givens(angles),
                newton: None,
            })
        }
    }

    #[test]
    fn exact_guard_rejects_uphill_and_accepts_downhill_without_hardware_calls() {
        let plan = RdmMeasurementPlan::new(2);
        let constraints = AffineConstraints::new(&plan, 2);
        let mut state = vec![num_complex::Complex64::new(0.0, 0.0); 16];
        state[3] = num_complex::Complex64::new(1.0, 0.0);
        let samples = PauliSamples::exact(plan.exact_expectations(&state));
        let config = OrbitalStatisticsConfig {
            acceptance: OoAcceptance::Heldout,
            ..Default::default()
        };
        for (diagonal, accepted) in [
            (array![[-1.0, 0.0], [0.0, 1.0]], false),
            (array![[1.0, 0.0], [0.0, -1.0]], true),
        ] {
            let engine = ClassicalEngine {
                h1: diagonal,
                h2: Array4::zeros((2, 2, 2, 2)),
                nuclear_repulsion: 0.0,
            };
            let (update, record) = optimize(
                &FixedProposal { angle: 0.2 },
                &engine,
                &[(0, 1)],
                &plan,
                &samples,
                &constraints,
                &config,
                10,
                1,
            )
            .unwrap();
            assert_eq!(record.accepted, accepted);
            assert_eq!(
                matches!(update.rotation, OrbitalRotation::Unchanged),
                !accepted
            );
            assert_eq!(record.validation_delta_ha, record.validation_upper_bound_ha);
        }
    }
}
