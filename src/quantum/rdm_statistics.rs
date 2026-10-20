//! Sufficient statistics and linear propagation for independent Pauli measurements.
//! Identity terms are exact. RDM entries and trial energies are never resampled independently.
use crate::quantum::rdm_measurement::RdmMeasurementPlan;
use anyhow::{ensure, Result};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Hypergeometric};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PauliSamples {
    #[serde(default)]
    pub sampling_seed: Option<u64>,
    pub shots: Option<u64>,
    pub plus_counts: Vec<u64>,
    pub means: Vec<f64>,
}

impl PauliSamples {
    pub fn validate(&self, plan: &RdmMeasurementPlan) -> Result<()> {
        ensure!(
            self.means.len() == plan.pauli_words().len(),
            "Pauli sample/catalog dimensions differ"
        );
        ensure!(
            self.means
                .iter()
                .all(|x| x.is_finite() && *x >= -1.0 && *x <= 1.0),
            "Invalid raw Pauli means"
        );
        match self.shots {
            Some(shots) => {
                ensure!(
                    shots > 0 && self.plus_counts.len() == self.means.len(),
                    "Missing or invalid Pauli counts"
                );
                for ((&count, &mean), word) in self
                    .plus_counts
                    .iter()
                    .zip(&self.means)
                    .zip(plan.pauli_words())
                {
                    ensure!(count <= shots, "Pauli count exceeds its shot budget");
                    ensure!((mean-(2.0*count as f64/shots as f64-1.0)).abs()<1e-12,
                        "Raw counts and means differ; corrected means cannot be used as held-out observations");
                    if word.x_mask | word.y_mask | word.z_mask == 0 {
                        ensure!(count == shots, "Identity counts must be exact");
                    }
                }
            }
            None => {
                ensure!(
                    self.plus_counts.is_empty(),
                    "Exact samples cannot contain finite-shot counts"
                );
                for (&mean, word) in self.means.iter().zip(plan.pauli_words()) {
                    if word.x_mask | word.y_mask | word.z_mask == 0 {
                        ensure!(mean == 1.0, "Identity expectation must be exact");
                    }
                }
            }
        }
        Ok(())
    }
    pub fn from_counts(shots: u64, plus_counts: Vec<u64>) -> Self {
        assert!(shots > 0 && plus_counts.iter().all(|&k| k <= shots));
        let means = plus_counts
            .iter()
            .map(|&k| 2.0 * k as f64 / shots as f64 - 1.0)
            .collect();
        Self {
            sampling_seed: None,
            shots: Some(shots),
            plus_counts,
            means,
        }
    }

    pub fn exact(means: Vec<f64>) -> Self {
        Self {
            sampling_seed: None,
            shots: None,
            plus_counts: Vec::new(),
            means,
        }
    }

    /// Randomly partition the *existing* outcomes. Conditional on the unknown means,
    /// the two partitions have the law of independent binomial experiments.
    pub fn split(&self, training_fraction: f64, seed: u64) -> Result<(Self, Self)> {
        ensure!(
            training_fraction.is_finite() && training_fraction > 0.0 && training_fraction < 1.0,
            "training_fraction must lie strictly between zero and one"
        );
        let Some(shots) = self.shots else {
            return Ok((self.clone(), self.clone()));
        };
        ensure!(
            shots >= 2,
            "held-out validation requires at least two shots per word"
        );
        ensure!(
            self.plus_counts.len() == self.means.len(),
            "missing Pauli counts"
        );
        let training_shots =
            ((shots as f64 * training_fraction).floor() as u64).clamp(1, shots - 1);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut train = Vec::with_capacity(self.means.len());
        let mut validation = Vec::with_capacity(self.means.len());
        for &count in &self.plus_counts {
            let k = Hypergeometric::new(shots, count, training_shots)?.sample(&mut rng);
            train.push(k);
            validation.push(count - k);
        }
        Ok((
            Self::from_counts(training_shots, train),
            Self::from_counts(shots - training_shots, validation),
        ))
    }

    /// Jeffreys posterior variances regularize all-equal finite samples. They are
    /// uncertainty diagnostics, not the basis of the rigorous held-out bound.
    pub fn variances(&self, plan: &RdmMeasurementPlan) -> Vec<f64> {
        let Some(shots) = self.shots else {
            return vec![0.0; self.means.len()];
        };
        self.plus_counts
            .iter()
            .zip(plan.pauli_words())
            .map(|(&k, word)| {
                if word.x_mask | word.y_mask | word.z_mask == 0 {
                    return 0.0;
                }
                let a = k as f64 + 0.5;
                let b = (shots - k) as f64 + 0.5;
                4.0 * a * b / ((a + b).powi(2) * (a + b + 1.0))
            })
            .collect()
    }

    pub fn dot(&self, weights: &[f64]) -> f64 {
        assert_eq!(weights.len(), self.means.len());
        weights.iter().zip(&self.means).map(|(a, b)| a * b).sum()
    }
}

pub fn covariance(weights: &[Vec<f64>], variances: &[f64]) -> Vec<Vec<f64>> {
    weights
        .iter()
        .map(|left| {
            weights
                .iter()
                .map(|right| {
                    assert_eq!(left.len(), variances.len());
                    assert_eq!(right.len(), variances.len());
                    left.iter()
                        .zip(right)
                        .zip(variances)
                        .map(|((a, b), v)| a * b * v)
                        .sum()
                })
                .collect()
        })
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffineConstraints {
    pub rows: Vec<Vec<f64>>,
    pub targets: Vec<f64>,
}

impl AffineConstraints {
    /// Repository convention: gamma2 contains one half of the spin sum.
    pub fn new(plan: &RdmMeasurementPlan, n_elec: usize) -> Self {
        let n = plan.n_spatial();
        let width = plan.pauli_words().len();
        let mut rows = Vec::new();
        let mut targets = Vec::new();
        let mut number = vec![0.0; width];
        for p in 0..n {
            for &(i, c) in &plan.one_body()[p * n + p].terms {
                number[i] += c;
            }
        }
        rows.push(number);
        targets.push(n_elec as f64);
        let mut pairs = vec![0.0; width];
        for p in 0..n {
            for q in 0..n {
                for &(i, c) in &plan.two_body()[((p * n + q) * n + p) * n + q].terms {
                    pairs[i] += c;
                }
            }
        }
        rows.push(pairs);
        targets.push((n_elec * n_elec.saturating_sub(1)) as f64 / 2.0);
        for p in 0..n {
            for r in p..n {
                let mut row = vec![0.0; width];
                for q in 0..n {
                    for &(i, c) in &plan.two_body()[((p * n + q) * n + r) * n + q].terms {
                        row[i] += c;
                    }
                }
                for &(i, c) in &plan.one_body()[p * n + r].terms {
                    row[i] -= n_elec.saturating_sub(1) as f64 * 0.5 * c;
                }
                rows.push(row);
                targets.push(0.0);
            }
        }
        Self { rows, targets }
    }

    pub fn max_residual(&self, means: &[f64]) -> f64 {
        self.rows
            .iter()
            .zip(&self.targets)
            .map(|(row, target)| {
                (row.iter().zip(means).map(|(a, b)| a * b).sum::<f64>() - target).abs()
            })
            .fold(0.0, f64::max)
    }

    pub fn correct(
        &self,
        samples: &PauliSamples,
        plan: &RdmMeasurementPlan,
    ) -> Result<AffineCorrection> {
        samples.validate(plan)?;
        let variances = samples.variances(plan);
        let m = self.rows.len();
        let gram: DMatrix<f64> = DMatrix::from_fn(m, m, |i, j| {
            self.rows[i]
                .iter()
                .zip(&self.rows[j])
                .zip(&variances)
                .map(|((a, b), v)| a * b * v)
                .sum()
        });
        let eigen = SymmetricEigen::try_new(gram, f64::EPSILON, 10_000)
            .ok_or_else(|| anyhow::anyhow!("Affine constraint eigendecomposition failed"))?;
        let scale = eigen.eigenvalues.iter().copied().fold(0.0, f64::max);
        let inverse_values = eigen.eigenvalues.map(|v| {
            if v > 1e-12 * scale.max(1e-30) {
                1.0 / v
            } else {
                0.0
            }
        });
        let inverse = &eigen.eigenvectors
            * DMatrix::from_diagonal(&inverse_values)
            * eigen.eigenvectors.transpose();
        let residual: DVector<f64> = DVector::from_iterator(
            m,
            self.rows.iter().zip(&self.targets).map(|(row, target)| {
                row.iter()
                    .zip(&samples.means)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    - target
            }),
        );
        let multipliers: DVector<f64> = &inverse * residual;
        let means: Vec<f64> = samples
            .means
            .iter()
            .enumerate()
            .map(|(j, &mean)| {
                mean - variances[j]
                    * (0..m)
                        .map(|i| self.rows[i][j] * multipliers[i])
                        .sum::<f64>()
            })
            .collect();
        let after = self.max_residual(&means);
        ensure!(
            means.iter().all(|v| v.is_finite()) && after < 1e-8,
            "Affine correction failed: maximum constraint residual {after:e}"
        );
        Ok(AffineCorrection {
            raw_means: samples.means.clone(),
            means,
            rows: self.rows.clone(),
            inverse,
            variances,
            residual_before: self.max_residual(&samples.means),
            residual_after: after,
        })
    }
}

pub struct AffineCorrection {
    raw_means: Vec<f64>,
    pub means: Vec<f64>,
    rows: Vec<Vec<f64>>,
    inverse: DMatrix<f64>,
    variances: Vec<f64>,
    pub residual_before: f64,
    pub residual_after: f64,
}

impl AffineCorrection {
    /// Apply the fitted linear map to a latent diagnostic draw. This is a
    /// projected diagnostic distribution, not a constrained physical posterior.
    pub fn apply_to(&self, candidate: &[f64]) -> Vec<f64> {
        assert_eq!(candidate.len(), self.raw_means.len());
        let delta: Vec<_> = candidate
            .iter()
            .zip(&self.raw_means)
            .map(|(a, b)| a - b)
            .collect();
        let residual = DVector::from_iterator(
            self.rows.len(),
            self.rows
                .iter()
                .map(|row| row.iter().zip(&delta).map(|(a, b)| a * b).sum::<f64>()),
        );
        let multipliers = &self.inverse * residual;
        delta
            .iter()
            .enumerate()
            .map(|(j, &d)| {
                self.means[j] + d
                    - self.variances[j]
                        * self
                            .rows
                            .iter()
                            .enumerate()
                            .map(|(i, row)| row[j] * multipliers[i])
                            .sum::<f64>()
            })
            .collect()
    }

    /// Transpose of the estimated affine correction map, for plug-in covariance.
    /// Estimated weights are treated as fixed; this is not a coverage guarantee.
    pub fn effective_weights(&self, weights: &[f64]) -> Vec<f64> {
        let product = DVector::from_iterator(
            self.rows.len(),
            self.rows.iter().map(|row| {
                row.iter()
                    .zip(weights)
                    .zip(&self.variances)
                    .map(|((a, b), v)| a * b * v)
                    .sum()
            }),
        );
        let multipliers = &self.inverse * product;
        weights
            .iter()
            .enumerate()
            .map(|(j, &w)| {
                w - self
                    .rows
                    .iter()
                    .enumerate()
                    .map(|(i, row)| row[j] * multipliers[i])
                    .sum::<f64>()
            })
            .collect()
    }
}

/// One-sided Hoeffding upper bound, conditional on a proposal made using other
/// data. The same holdout estimates both energies, so only difference weights
/// enter. No plug-in variance, Gaussian approximation, or extra shots are used.
pub fn heldout_upper_bound(
    plan: &RdmMeasurementPlan,
    samples: &PauliSamples,
    difference_weights: &[f64],
    alpha: f64,
) -> Result<(f64, f64)> {
    samples.validate(plan)?;
    ensure!(
        difference_weights.len() == samples.means.len()
            && difference_weights.iter().all(|x| x.is_finite()),
        "Invalid held-out energy-difference weights"
    );
    ensure!(
        alpha.is_finite() && alpha > 0.0 && alpha < 1.0,
        "alpha must lie between zero and one"
    );
    let delta = samples.dot(difference_weights);
    let Some(shots) = samples.shots else {
        return Ok((delta, delta));
    };
    let proxy: f64 = difference_weights
        .iter()
        .zip(plan.pauli_words())
        .filter(|(_, word)| word.x_mask | word.y_mask | word.z_mask != 0)
        .map(|(a, _)| a * a / shots as f64)
        .sum();
    Ok((delta, delta + (2.0 * (1.0 / alpha).ln() * proxy).sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn split_conserves_counts_and_replays_without_new_measurements() {
        let s = PauliSamples::from_counts(100, vec![0, 23, 100]);
        let (a, b) = s.split(0.8, 42).unwrap();
        assert_eq!(a.shots, Some(80));
        assert_eq!(b.shots, Some(20));
        assert_eq!(a.plus_counts, s.split(0.8, 42).unwrap().0.plus_counts);
        for i in 0..3 {
            assert_eq!(a.plus_counts[i] + b.plus_counts[i], s.plus_counts[i]);
        }
        assert!(PauliSamples::from_counts(1, vec![1]).split(0.8, 1).is_err());
    }

    #[test]
    fn affine_restores_joint_number_and_half_scaled_contractions() {
        let plan = RdmMeasurementPlan::new(2);
        let mut state = vec![Complex64::new(0.0, 0.0); 16];
        state[3] = Complex64::new(0.8, 0.0);
        state[12] = Complex64::new(0.6, 0.0);
        let exact = plan.exact_expectations(&state);
        let constraints = AffineConstraints::new(&plan, 2);
        assert!(constraints.max_residual(&exact) < 1e-12);
        let measured = plan.sample_expectations(&exact, 100, 81);
        let corrected = constraints.correct(&measured, &plan).unwrap();
        assert!(corrected.residual_after < 1e-10);
        let candidate: Vec<_> = measured
            .means
            .iter()
            .enumerate()
            .map(|(i, &mu)| {
                if measured.variances(&plan)[i] == 0.0 {
                    mu
                } else {
                    mu + 0.01 * (i as f64).sin()
                }
            })
            .collect();
        let mapped = corrected.apply_to(&candidate);
        assert!(constraints.max_residual(&mapped) < 1e-10);
        let weights: Vec<_> = (0..candidate.len()).map(|i| (i as f64).cos()).collect();
        let effective = corrected.effective_weights(&weights);
        let direct: f64 = weights
            .iter()
            .zip(mapped.iter().zip(&corrected.means))
            .map(|(w, (a, b))| w * (a - b))
            .sum();
        let propagated: f64 = effective
            .iter()
            .zip(candidate.iter().zip(&measured.means))
            .map(|(w, (a, b))| w * (a - b))
            .sum();
        assert!((direct - propagated).abs() < 1e-12);
        let (one, two) = plan.reconstruct(&corrected.means);
        assert!((one.diag().sum() - 2.0).abs() < 1e-10);
        let trace: f64 = (0..2)
            .flat_map(|p| (0..2).map(move |q| (p, q)))
            .map(|(p, q)| two[[p, q, p, q]])
            .sum();
        assert!((trace - 1.0).abs() < 1e-10);
        let exact_samples = PauliSamples::exact(exact.clone());
        assert_eq!(
            constraints.correct(&exact_samples, &plan).unwrap().means,
            exact
        );
    }

    #[test]
    fn energy_difference_cancels_shared_offset_and_identity_uncertainty() {
        let plan = RdmMeasurementPlan::new(2);
        let mut weights = vec![0.0; plan.pauli_words().len()];
        let identity = plan
            .pauli_words()
            .iter()
            .position(|w| w.x_mask | w.y_mask | w.z_mask == 0)
            .unwrap();
        weights[identity] = -0.01;
        let samples = PauliSamples::from_counts(100, vec![100; weights.len()]);
        let (delta, upper) = heldout_upper_bound(&plan, &samples, &weights, 0.005).unwrap();
        assert_eq!(delta, upper);
        assert_eq!(delta, -0.01);
        let zero = vec![0.0; weights.len()];
        assert_eq!(
            heldout_upper_bound(&plan, &samples, &zero, 0.005).unwrap(),
            (0.0, 0.0)
        );
        assert_eq!(samples.variances(&plan)[identity], 0.0);
        assert!(samples.variances(&plan).iter().any(|&v| v > 0.0));
    }

    #[test]
    fn heldout_rejects_corrected_or_mismatched_observations() {
        let plan = RdmMeasurementPlan::new(2);
        let mut samples = PauliSamples::from_counts(100, vec![100; plan.pauli_words().len()]);
        let weights = vec![0.; samples.means.len()];
        let index = plan
            .pauli_words()
            .iter()
            .position(|w| w.x_mask | w.y_mask | w.z_mask != 0)
            .unwrap();
        samples.means[index] = 0.9;
        assert!(heldout_upper_bound(&plan, &samples, &weights, 0.005).is_err());
        samples.means[index] = 1.0;
        samples.plus_counts.pop();
        assert!(heldout_upper_bound(&plan, &samples, &weights, 0.005).is_err());
    }
}
