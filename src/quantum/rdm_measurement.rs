use crate::quantum::jordan_wigner::{transform_hermitian_part, FermionAction, Pauli, PauliWord};
use ndarray::{Array2, Array4};
use num_complex::Complex64;
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::Binomial;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CompiledPauliWord {
    pub x_mask: usize,
    pub y_mask: usize,
    pub z_mask: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservablePlan {
    pub terms: Vec<(usize, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdmMeasurementPlan {
    n_spatial: usize,
    pauli_words: Vec<CompiledPauliWord>,
    one_body: Vec<ObservablePlan>,
    two_body: Vec<ObservablePlan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpinRdmPlans {
    pub n_spin: usize,
    pub pairs: Vec<(usize, usize)>,
    pub one_body: Vec<ObservablePlan>,
    pub pair_body: Vec<ObservablePlan>,
}

impl RdmMeasurementPlan {
    pub fn new(n_spatial: usize) -> Self {
        let mut word_indices = BTreeMap::<CompiledPauliWord, usize>::new();
        let mut pauli_words = Vec::new();
        let mut one_body = Vec::with_capacity(n_spatial * n_spatial);
        let mut two_body = Vec::with_capacity(n_spatial.pow(4));

        for p in 0..n_spatial {
            for q in 0..n_spatial {
                let mut terms = BTreeMap::<usize, f64>::new();
                for spin in 0..2 {
                    append_observable(
                        &[
                            (2 * p + spin, FermionAction::Create),
                            (2 * q + spin, FermionAction::Annihilate),
                        ],
                        1.0,
                        &mut terms,
                        &mut word_indices,
                        &mut pauli_words,
                    );
                }
                one_body.push(ObservablePlan {
                    terms: terms.into_iter().collect(),
                });
            }
        }

        for p in 0..n_spatial {
            for q in 0..n_spatial {
                for r in 0..n_spatial {
                    for s in 0..n_spatial {
                        let mut terms = BTreeMap::<usize, f64>::new();
                        for spin1 in 0..2 {
                            for spin2 in 0..2 {
                                append_observable(
                                    &[
                                        (2 * p + spin1, FermionAction::Create),
                                        (2 * q + spin2, FermionAction::Create),
                                        (2 * s + spin2, FermionAction::Annihilate),
                                        (2 * r + spin1, FermionAction::Annihilate),
                                    ],
                                    0.5,
                                    &mut terms,
                                    &mut word_indices,
                                    &mut pauli_words,
                                );
                            }
                        }
                        two_body.push(ObservablePlan {
                            terms: terms.into_iter().collect(),
                        });
                    }
                }
            }
        }

        Self {
            n_spatial,
            pauli_words,
            one_body,
            two_body,
        }
    }

    /// Exact expectation values using the same observables as finite-shot sampling.
    pub fn evaluate_exact(&self, state: &[Complex64]) -> (Array2<f64>, Array4<f64>) {
        self.reconstruct(&self.exact_expectations(state))
    }

    pub fn exact_expectations(&self, state: &[Complex64]) -> Vec<f64> {
        assert_eq!(state.len(), 1usize << (2 * self.n_spatial));
        self.pauli_words
            .par_iter()
            .map(|word| exact_pauli_expectation(state, *word).clamp(-1.0, 1.0))
            .collect()
    }

    pub fn measure(
        &self,
        state: &[Complex64],
        shots: usize,
        seed: u64,
    ) -> (Array2<f64>, Array4<f64>) {
        let samples = self.sample_expectations(&self.exact_expectations(state), shots, seed);
        self.reconstruct(&samples.means)
    }

    /// Retain sufficient statistics without changing the legacy RNG order.
    pub fn sample_expectations(
        &self,
        expectations: &[f64],
        shots: usize,
        seed: u64,
    ) -> crate::quantum::rdm_statistics::PauliSamples {
        assert!(shots > 0, "finite-shot RDM evaluation requires shots > 0");
        assert_eq!(expectations.len(), self.pauli_words.len());
        let mut rng = StdRng::seed_from_u64(seed);
        let plus_counts: Vec<u64> = expectations
            .iter()
            .map(|&expectation| {
                Binomial::new(shots as u64, 0.5 * (1.0 + expectation))
                    .expect("invalid binomial parameters")
                    .sample(&mut rng)
            })
            .collect();
        let mut samples =
            crate::quantum::rdm_statistics::PauliSamples::from_counts(shots as u64, plus_counts);
        samples.sampling_seed = Some(seed);
        samples
    }

    pub fn reconstruct(&self, expectations: &[f64]) -> (Array2<f64>, Array4<f64>) {
        assert_eq!(expectations.len(), self.pauli_words.len());
        let mut gamma1 = Array2::<f64>::zeros((self.n_spatial, self.n_spatial));
        for p in 0..self.n_spatial {
            for q in 0..self.n_spatial {
                gamma1[[p, q]] =
                    evaluate_plan(&self.one_body[p * self.n_spatial + q], expectations);
            }
        }

        let mut gamma2 = Array4::<f64>::zeros((
            self.n_spatial,
            self.n_spatial,
            self.n_spatial,
            self.n_spatial,
        ));
        for p in 0..self.n_spatial {
            for q in 0..self.n_spatial {
                for r in 0..self.n_spatial {
                    for s in 0..self.n_spatial {
                        let index =
                            (((p * self.n_spatial + q) * self.n_spatial + r) * self.n_spatial) + s;
                        gamma2[[p, q, r, s]] = evaluate_plan(&self.two_body[index], expectations);
                    }
                }
            }
        }

        (gamma1, gamma2)
    }

    #[cfg(test)]
    fn number_pauli_words(&self) -> usize {
        self.pauli_words.len()
    }

    /// Unique observables that require shots; the identity has no measurement cost.
    pub fn number_measured_pauli_words(&self) -> usize {
        self.pauli_words
            .iter()
            .filter(|word| word.x_mask | word.y_mask | word.z_mask != 0)
            .count()
    }

    pub fn n_spatial(&self) -> usize {
        self.n_spatial
    }
    pub fn pauli_words(&self) -> &[CompiledPauliWord] {
        &self.pauli_words
    }
    pub fn one_body(&self) -> &[ObservablePlan] {
        &self.one_body
    }
    pub fn two_body(&self) -> &[ObservablePlan] {
        &self.two_body
    }

    /// Real, Sz-conserving spin-orbital marginals from the existing catalog.
    /// A missing word is an error; this API never silently requests new shots.
    pub fn spin_plans(&self) -> anyhow::Result<SpinRdmPlans> {
        let n = 2 * self.n_spatial;
        let pairs: Vec<_> = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .collect();
        let lookup: BTreeMap<_, _> = self
            .pauli_words
            .iter()
            .copied()
            .enumerate()
            .map(|(i, w)| (w, i))
            .collect();
        let compile = |ops: &[(usize, FermionAction)]| -> anyhow::Result<ObservablePlan> {
            let mut terms = Vec::new();
            for (word, c) in transform_hermitian_part(ops) {
                if c.re.abs() < 1e-12 {
                    continue;
                }
                let index = lookup.get(&compile_word(&word)).ok_or_else(|| {
                    anyhow::anyhow!("Spin-RDM constraint requires an unmeasured Pauli word")
                })?;
                terms.push((*index, c.re));
            }
            terms.sort_by_key(|&(i, _)| i);
            Ok(ObservablePlan { terms })
        };
        let mut one_body = Vec::new();
        for i in 0..n {
            for j in 0..n {
                one_body.push(if i % 2 == j % 2 {
                    compile(&[(i, FermionAction::Create), (j, FermionAction::Annihilate)])?
                } else {
                    ObservablePlan { terms: Vec::new() }
                });
            }
        }
        let mut pair_body = Vec::new();
        for &(i, j) in &pairs {
            for &(k, l) in &pairs {
                pair_body.push(if i % 2 + j % 2 == k % 2 + l % 2 {
                    compile(&[
                        (i, FermionAction::Create),
                        (j, FermionAction::Create),
                        (l, FermionAction::Annihilate),
                        (k, FermionAction::Annihilate),
                    ])?
                } else {
                    ObservablePlan { terms: Vec::new() }
                });
            }
        }
        Ok(SpinRdmPlans {
            n_spin: n,
            pairs,
            one_body,
            pair_body,
        })
    }

    /// Coefficients in E = E_nuc + weights · Pauli means, including the identity.
    pub fn energy_weights(&self, h1: &Array2<f64>, h2: &Array4<f64>) -> Vec<f64> {
        assert_eq!(h1.dim(), (self.n_spatial, self.n_spatial));
        assert_eq!(
            h2.dim(),
            (
                self.n_spatial,
                self.n_spatial,
                self.n_spatial,
                self.n_spatial
            )
        );
        let mut weights = vec![0.0; self.pauli_words.len()];
        for (observable, &coefficient) in self
            .one_body
            .iter()
            .zip(h1.iter())
            .chain(self.two_body.iter().zip(h2.iter()))
        {
            for &(index, scale) in &observable.terms {
                weights[index] += coefficient * scale;
            }
        }
        weights
    }
}

fn append_observable(
    operators: &[(usize, FermionAction)],
    scale: f64,
    terms: &mut BTreeMap<usize, f64>,
    word_indices: &mut BTreeMap<CompiledPauliWord, usize>,
    pauli_words: &mut Vec<CompiledPauliWord>,
) {
    for (word, coefficient) in transform_hermitian_part(operators) {
        let compiled = compile_word(&word);
        let index = *word_indices.entry(compiled).or_insert_with(|| {
            let index = pauli_words.len();
            pauli_words.push(compiled);
            index
        });
        *terms.entry(index).or_insert(0.0) += scale * coefficient.re;
    }
    terms.retain(|_, coefficient| coefficient.abs() > 1e-12);
}

fn compile_word(word: &PauliWord) -> CompiledPauliWord {
    let mut compiled = CompiledPauliWord {
        x_mask: 0,
        y_mask: 0,
        z_mask: 0,
    };
    for (&qubit, &pauli) in word {
        match pauli {
            Pauli::X => compiled.x_mask |= 1usize << qubit,
            Pauli::Y => compiled.y_mask |= 1usize << qubit,
            Pauli::Z => compiled.z_mask |= 1usize << qubit,
        }
    }
    compiled
}

fn evaluate_plan(plan: &ObservablePlan, sampled_expectations: &[f64]) -> f64 {
    plan.terms
        .iter()
        .map(|(index, coefficient)| coefficient * sampled_expectations[*index])
        .sum()
}

#[cfg(test)]
fn sample_expectation(expectation: f64, shots: usize, rng: &mut StdRng) -> f64 {
    let probability_plus = 0.5 * (1.0 + expectation);
    let plus_count = Binomial::new(shots as u64, probability_plus)
        .expect("invalid binomial parameters")
        .sample(rng);
    2.0 * plus_count as f64 / shots as f64 - 1.0
}

fn exact_pauli_expectation(state: &[Complex64], word: CompiledPauliWord) -> f64 {
    if word.x_mask == 0 && word.y_mask == 0 && word.z_mask == 0 {
        return 1.0;
    }
    let flip_mask = word.x_mask | word.y_mask;
    let phase_mask = word.y_mask | word.z_mask;
    let y_count = word.y_mask.count_ones();
    let initial_phase = match y_count % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    };

    let mut expectation = Complex64::new(0.0, 0.0);
    for (basis, amplitude) in state.iter().enumerate() {
        let target = basis ^ flip_mask;
        let parity = (basis & phase_mask).count_ones() % 2;
        let phase = if parity == 0 {
            initial_phase
        } else {
            -initial_phase
        };
        expectation += amplitude.conj() * phase * state[target];
    }
    expectation.re
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chemistry::rdm::{compute_1rdm_spatial, compute_2rdm_spatial};

    fn test_state() -> Vec<Complex64> {
        let mut state = vec![Complex64::new(0.0, 0.0); 16];
        state[0b0011] = Complex64::new(0.6, 0.0);
        state[0b0110] = Complex64::new(0.3, 0.0);
        state[0b1001] = Complex64::new(-0.4, 0.0);
        state[0b1100] = Complex64::new(-0.6244997998398398, 0.0);
        state
    }

    #[test]
    fn exact_pauli_rdms_match_direct_rdms_at_h2_and_lih_sizes() {
        for (n, electrons) in [(2, 2), (6, 4)] {
            let mut state: Vec<_> = (0usize..1 << (2 * n))
                .map(|i| {
                    Complex64::new(
                        if i.count_ones() == electrons {
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
            let plan = RdmMeasurementPlan::new(n);
            let (one, two) = plan.evaluate_exact(&state);
            let reference_one = compute_1rdm_spatial(&state, n);
            let reference_two = compute_2rdm_spatial(&state, n);
            let error = one
                .iter()
                .zip(reference_one.iter())
                .chain(two.iter().zip(reference_two.iter()))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(error < 1e-12, "n={n}: RDM error={error}");
            if n == 6 {
                assert_eq!(plan.number_measured_pauli_words(), 2778);
            }
        }
    }

    #[test]
    fn exact_pauli_rdms_preserve_real_part_convention_for_complex_states() {
        let mut state = test_state();
        state[0b0110] *= Complex64::new(0.0, 1.0);
        let plan = RdmMeasurementPlan::new(2);
        let (one, two) = plan.evaluate_exact(&state);
        assert!(one
            .iter()
            .zip(compute_1rdm_spatial(&state, 2).iter())
            .all(|(a, b)| (a - b).abs() < 1e-12));
        assert!(two
            .iter()
            .zip(compute_2rdm_spatial(&state, 2).iter())
            .all(|(a, b)| (a - b).abs() < 1e-12));
    }

    #[test]
    fn identity_and_pauli_eigenstates_have_no_sampling_noise() {
        let mut rng = StdRng::seed_from_u64(17);
        assert_eq!(sample_expectation(1.0, 1000, &mut rng), 1.0);
        assert_eq!(sample_expectation(-1.0, 1000, &mut rng), -1.0);
        let identity = CompiledPauliWord {
            x_mask: 0,
            y_mask: 0,
            z_mask: 0,
        };
        assert_eq!(exact_pauli_expectation(&test_state(), identity), 1.0);
    }

    #[test]
    fn compiled_plan_converges_to_statevector_rdms() {
        let state = test_state();
        let exact1 = compute_1rdm_spatial(&state, 2);
        let exact2 = compute_2rdm_spatial(&state, 2);
        let plan = RdmMeasurementPlan::new(2);
        let (sampled1, sampled2) = plan.measure(&state, 100_000, 1234);
        assert!((&sampled1 - &exact1).iter().all(|error| error.abs() < 0.02));
        assert!((&sampled2 - &exact2).iter().all(|error| error.abs() < 0.02));
    }

    #[test]
    fn compiled_plan_is_seed_reproducible() {
        let state = test_state();
        let plan = RdmMeasurementPlan::new(2);
        assert_eq!(
            plan.measure(&state, 1000, 42),
            plan.measure(&state, 1000, 42)
        );
    }

    #[test]
    fn plan_deduplicates_pauli_words() {
        let plan = RdmMeasurementPlan::new(2);
        assert!(plan.number_pauli_words() < 2 * 2 + 2usize.pow(4) * 4 * 8);
    }

    #[test]
    fn binomial_sample_is_a_valid_finite_shot_value() {
        let shots = 37;
        let mut rng = StdRng::seed_from_u64(99);
        let estimate = sample_expectation(0.0, shots, &mut rng);
        let plus_count = 0.5 * shots as f64 * (estimate + 1.0);
        assert!((-1.0..=1.0).contains(&estimate));
        assert!((plus_count - plus_count.round()).abs() < 1e-12);
    }
}
