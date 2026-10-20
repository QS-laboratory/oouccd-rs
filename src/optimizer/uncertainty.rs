//! Linear Fourier uncertainty propagation; this module never treats the angles
//! as independent measurements and never advances the quantum RNG stream.
use crate::optimizer::trigonometric::minimize_trigonometric_func;
use crate::quantum::rdm_measurement::RdmMeasurementPlan;
use crate::quantum::rdm_statistics::{covariance, PauliSamples};
use anyhow::{ensure, Result};
use ndarray::Array1;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Beta, Distribution};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourierProfile {
    pub coefficients: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
}

pub fn transform(samples: usize) -> Vec<Vec<f64>> {
    assert!(samples >= 3 && samples % 2 == 1);
    (0..samples)
        .map(|coefficient| {
            (0..samples)
                .map(|j| {
                    if coefficient == 0 {
                        return 1.0 / samples as f64;
                    }
                    let harmonic = (coefficient + 1) / 2;
                    let phase = 2.0 * PI * j as f64 * harmonic as f64 / samples as f64;
                    2.0 / samples as f64
                        * if coefficient % 2 == 1 {
                            phase.cos()
                        } else {
                            phase.sin()
                        }
                })
                .collect()
        })
        .collect()
}

pub fn features(angle: f64, degree: usize) -> Vec<f64> {
    let mut f = vec![1.0];
    for k in 1..=degree {
        f.push((k as f64 * angle).cos());
        f.push((k as f64 * angle).sin());
    }
    f
}

impl FourierProfile {
    pub fn from_measurements(
        energies: &[f64],
        energy_weights: &[Vec<f64>],
        variances: &[f64],
    ) -> Self {
        let t = transform(energies.len());
        assert_eq!(energy_weights.len(), energies.len());
        let coefficients = t
            .iter()
            .map(|row| row.iter().zip(energies).map(|(a, b)| a * b).sum())
            .collect();
        let coefficient_weights: Vec<Vec<f64>> = t
            .iter()
            .map(|row| {
                (0..variances.len())
                    .map(|l| row.iter().zip(energy_weights).map(|(a, w)| a * w[l]).sum())
                    .collect()
            })
            .collect();
        Self {
            coefficients,
            covariance: covariance(&coefficient_weights, variances),
        }
    }

    pub fn evaluate(&self, angle: f64) -> f64 {
        features(angle, (self.coefficients.len() - 1) / 2)
            .iter()
            .zip(&self.coefficients)
            .map(|(a, b)| a * b)
            .sum()
    }

    pub fn difference_variance(&self, angle: f64, baseline: f64) -> f64 {
        let degree = (self.coefficients.len() - 1) / 2;
        let f: Vec<_> = features(angle, degree)
            .iter()
            .zip(features(baseline, degree))
            .map(|(a, b)| a - b)
            .collect();
        let v: f64 = f
            .iter()
            .enumerate()
            .map(|(i, a)| {
                f.iter()
                    .enumerate()
                    .map(|(j, b)| a * self.covariance[i][j] * b)
                    .sum::<f64>()
            })
            .sum();
        v.max(0.0)
    }

    pub fn nominal_minimum(&self) -> (f64, f64) {
        let n = self.coefficients.len();
        let samples =
            Array1::from_iter((0..n).map(|j| self.evaluate(2.0 * PI * j as f64 / n as f64)));
        minimize_trigonometric_func(&samples)
    }

    /// Risk score is generally not a degree-four polynomial. Search a dense
    /// grid and refine all grid-local minima. This is a numerical search,
    /// not a certified global minimum or a frequentist confidence guarantee.
    pub fn robust_minimum(&self, beta: f64, grid: usize) -> Result<(f64, f64)> {
        ensure!(
            beta.is_finite() && beta >= 0.0 && grid >= 32,
            "Invalid robust Fourier search settings"
        );
        let score = |a: f64| {
            self.evaluate(a) - self.evaluate(0.0) + beta * self.difference_variance(a, 0.0).sqrt()
        };
        let step = 2.0 * PI / grid as f64;
        let mut best = (0.0, 0.0);
        let nominal = self.nominal_minimum().0;
        if score(nominal) < best.1 {
            best = (nominal, score(nominal));
        }
        for i in 0..grid {
            let theta = -PI + i as f64 * step;
            if score(theta) <= score(theta - step) && score(theta) <= score(theta + step) {
                let (mut left, mut right) = (theta - step, theta + step);
                for _ in 0..64 {
                    let a = right - (right - left) * 0.6180339887498949;
                    let b = left + (right - left) * 0.6180339887498949;
                    if score(a) < score(b) {
                        right = b;
                    } else {
                        left = a;
                    }
                }
                let angle = 0.5 * (left + right);
                let value = score(angle);
                if value < best.1 {
                    best = (angle, value);
                }
            }
        }
        best.0 = (best.0 + PI).rem_euclid(2.0 * PI) - PI;
        Ok(best)
    }
}

/// Independent Beta posteriors are a diagnostic model for Pauli probabilities,
/// not a claim that their joint draws are N-representable. Reconstruct/project
/// the *whole* draw, then replay the full sweep for adaptive bootstrap studies.
pub fn posterior_draw(
    samples: &PauliSamples,
    plan: &RdmMeasurementPlan,
    seed: u64,
) -> Result<Vec<f64>> {
    let Some(shots) = samples.shots else {
        return Ok(samples.means.clone());
    };
    let mut rng = StdRng::seed_from_u64(seed);
    samples
        .plus_counts
        .iter()
        .zip(plan.pauli_words())
        .map(|(&k, w)| {
            if w.x_mask | w.y_mask | w.z_mask == 0 {
                return Ok(1.0);
            }
            let probability = Beta::new(k as f64 + 0.5, (shots - k) as f64 + 0.5)?.sample(&mut rng);
            Ok(2.0 * probability - 1.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::rdm_statistics::covariance;

    #[test]
    fn shared_offset_does_not_create_angle_uncertainty_or_extra_harmonics() {
        let n = 9;
        let weights: Vec<Vec<f64>> = (0..n)
            .map(|j| {
                let a = 2.0 * PI * j as f64 / n as f64;
                vec![1.0, a.cos(), a.sin()]
            })
            .collect();
        let energies: Vec<_> = weights.iter().map(|w| 10.0 + w[1] + 0.2 * w[2]).collect();
        let profile = FourierProfile::from_measurements(&energies, &weights, &[0.4, 0.0, 0.0]);
        assert!((profile.covariance[0][0] - 0.4).abs() < 1e-12);
        assert!(profile.difference_variance(1.3, 0.0) < 1e-20);
        for &a in &[0.1, 0.7, 2.8] {
            assert!((profile.evaluate(a) - (10.0 + a.cos() + 0.2 * a.sin())).abs() < 1e-12);
        }
        let sigma = covariance(&weights, &[0.4, 0.03, 0.02]);
        assert!((sigma[0][1] - 0.4 - 0.03 * (2.0 * PI / 9.0).cos()).abs() < 1e-12);
        let (a, _) = profile.nominal_minimum();
        assert!((profile.evaluate(a) - (10.0 - 1.04f64.sqrt())).abs() < 1e-10);
    }

    #[test]
    fn more_classical_angles_do_not_reduce_shared_data_uncertainty() {
        let means = [-0.7, 0.3, -0.2, 0.1, 0.04, -0.03, 0.02, 0.01, -0.01];
        let variances = [0.02, 0.01, 0.03, 0.01, 0.04, 0.02, 0.01, 0.03, 0.01];
        let make = |n: usize| {
            let weights: Vec<_> = (0..n)
                .map(|j| features(2.0 * PI * j as f64 / n as f64, 4))
                .collect();
            let energies: Vec<f64> = weights
                .iter()
                .map(|row| row.iter().zip(means).map(|(a, b)| a * b).sum())
                .collect();
            FourierProfile::from_measurements(&energies, &weights, &variances)
        };
        let nine = make(9);
        let more = make(21);
        for i in 0..9 {
            assert!((nine.coefficients[i] - more.coefficients[i]).abs() < 1e-12);
            for j in 0..9 {
                assert!((nine.covariance[i][j] - more.covariance[i][j]).abs() < 1e-12);
            }
        }
        assert!(more.coefficients[9..].iter().all(|x| x.abs() < 1e-12));
    }
}
