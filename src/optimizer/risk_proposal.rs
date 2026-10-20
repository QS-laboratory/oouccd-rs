//! Secondary research proposals. Risk scores are training-data heuristics;
//! the independent whole-sweep validation wrapper supplies the safety bound.
use crate::optimizer::orbital::{OrbitalOptimizer, OrbitalProblem, OrbitalRotation, OrbitalUpdate};
use crate::optimizer::uncertainty::FourierProfile;
use crate::quantum::rdm_measurement::RdmMeasurementPlan;
use crate::quantum::rdm_statistics::AffineCorrection;
use anyhow::{ensure, Result};
use nalgebra::DMatrix;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

pub struct RiskAwareProposal<'a> {
    pub base: &'a dyn OrbitalOptimizer,
    pub sequential: bool,
    pub plan: &'a RdmMeasurementPlan,
    pub variances: Vec<f64>,
    pub affine: Option<&'a AffineCorrection>,
    pub beta: f64,
}

impl RiskAwareProposal<'_> {
    fn weights(&self, h1: &Array2<f64>, h2: &ndarray::Array4<f64>) -> Vec<f64> {
        let weights = self.plan.energy_weights(h1, h2);
        self.affine
            .map_or_else(|| weights.clone(), |a| a.effective_weights(&weights))
    }
}

impl OrbitalOptimizer for RiskAwareProposal<'_> {
    fn optimize(&self, problem: &OrbitalProblem<'_>) -> Result<OrbitalUpdate> {
        ensure!(
            self.beta.is_finite() && self.beta >= 0.,
            "Invalid risk beta"
        );
        ensure!(
            self.variances.len() == self.plan.pauli_words().len()
                && self.variances.iter().all(|x| x.is_finite() && *x >= 0.0),
            "Invalid risk covariance inputs"
        );
        let before = problem.energy();
        if self.sequential {
            let mut kappa = vec![0.; problem.pairs.len()];
            for coordinate in 0..kappa.len() {
                let mut energies = Vec::new();
                let mut weights = Vec::new();
                for j in 0..9 {
                    let mut trial = kappa.clone();
                    trial[coordinate] += 2. * PI * j as f64 / 9.;
                    let (h1, h2) = problem.engine.integrals_with_givens(problem.pairs, &trial);
                    energies.push(
                        problem.engine.nuclear_repulsion
                            + (&h1 * problem.rdm1).sum()
                            + (&h2 * problem.rdm2).sum(),
                    );
                    weights.push(self.weights(&h1, &h2));
                }
                let profile =
                    FourierProfile::from_measurements(&energies, &weights, &self.variances);
                kappa[coordinate] =
                    (kappa[coordinate] + profile.robust_minimum(self.beta, 128)?.0 + PI)
                        .rem_euclid(2. * PI)
                        - PI;
            }
            Ok(OrbitalUpdate {
                energy_before: before,
                energy_after: problem.energy_with_givens(&kappa),
                rotation: OrbitalRotation::Givens(Array1::from(kappa)),
                newton: None,
            })
        } else {
            let nominal = self.base.optimize(problem)?;
            let OrbitalRotation::Matrix(rotation) = &nominal.rotation else {
                return Ok(nominal);
            };
            let n = rotation.nrows();
            let a = DMatrix::from_fn(n, n, |i, j| rotation[[i, j]] - if i == j { 1. } else { 0. });
            ensure!(
                a.norm() < 0.8,
                "Risk-Newton logarithm outside the verified convergence range"
            );
            let mut power = a.clone();
            let mut generator = DMatrix::zeros(n, n);
            for k in 1..=64 {
                let sign = if k % 2 == 1 { 1. } else { -1. };
                generator += &power * (sign / k as f64);
                power = &power * &a;
            }
            generator = 0.5 * (&generator - generator.transpose());
            let before_weights = self.weights(&problem.engine.h1, &problem.engine.h2);
            let mut best_score = 0.;
            let mut best_rotation = None;
            let mut best_energy = before;
            // Same risk penalty as SOO; search the stabilized Newton ray rather
            // than assigning Newton an unguarded point-estimate comparator.
            for j in 1..=32 {
                let matrix = (&generator * (j as f64 / 32.)).exp();
                let trial = Array2::from_shape_fn((n, n), |(i, k)| matrix[(i, k)]);
                let (h1, h2) = problem.engine.rotated_integrals(&trial);
                let energy = problem.engine.nuclear_repulsion
                    + (&h1 * problem.rdm1).sum()
                    + (&h2 * problem.rdm2).sum();
                let weights = self.weights(&h1, &h2);
                let variance: f64 = weights
                    .iter()
                    .zip(&before_weights)
                    .zip(&self.variances)
                    .map(|((a, b), v)| (a - b).powi(2) * v)
                    .sum();
                let score = energy - before + self.beta * variance.max(0.).sqrt();
                if score < best_score {
                    best_score = score;
                    best_rotation = Some(trial);
                    best_energy = energy;
                }
            }
            Ok(OrbitalUpdate {
                energy_before: before,
                energy_after: best_energy,
                rotation: best_rotation.map_or(OrbitalRotation::Unchanged, OrbitalRotation::Matrix),
                newton: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chemistry::engine::ClassicalEngine;
    use crate::optimizer::newton::NewtonOrbitalOptimizer;
    use crate::quantum::rdm_statistics::PauliSamples;
    use ndarray::{array, Array4};

    #[test]
    fn zero_uncertainty_risk_proposals_preserve_descent_and_rotation() {
        let plan = RdmMeasurementPlan::new(2);
        let mut state = vec![num_complex::Complex64::new(0., 0.); 16];
        state[3] = num_complex::Complex64::new(1., 0.);
        let samples = PauliSamples::exact(plan.exact_expectations(&state));
        let (one, two) = plan.reconstruct(&samples.means);
        let engine = ClassicalEngine {
            h1: array![[-1., 0.2], [0.2, 1.]],
            h2: Array4::zeros((2, 2, 2, 2)),
            nuclear_repulsion: 0.,
        };
        let problem = OrbitalProblem {
            engine: &engine,
            rdm1: &one,
            rdm2: &two,
            pairs: &[(0, 1)],
        };
        let base = NewtonOrbitalOptimizer {
            options: Default::default(),
        };
        let nominal = base.optimize(&problem).unwrap();
        for sequential in [false, true] {
            let risk = RiskAwareProposal {
                base: &base,
                sequential,
                plan: &plan,
                variances: samples.variances(&plan),
                affine: None,
                beta: 2.,
            };
            let update = risk.optimize(&problem).unwrap();
            assert!(update.energy_after <= nominal.energy_after + 1e-10);
            assert!(update.energy_after < update.energy_before);
            let r = crate::optimizer::statistical_oo::rotation_matrix(&update, 2, problem.pairs);
            assert!((&r.dot(&r.t()) - &Array2::<f64>::eye(2))
                .iter()
                .all(|x| x.abs() < 1e-12));
        }
    }
}
