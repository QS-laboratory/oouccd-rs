use crate::quantum::jordan_wigner::{transform_hermitian_part, FermionAction, Pauli, PauliWord};
use ndarray::{Array2, Array4};
use num_complex::Complex64;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::SeedableRng;

pub fn measure_spatial_rdms(
    state: &[Complex64],
    n_spatial: usize,
    shots: usize,
    seed: u64,
) -> (Array2<f64>, Array4<f64>) {
    assert!(shots > 0, "finite-shot RDM evaluation requires shots > 0");
    let mut rng = StdRng::seed_from_u64(seed);
    let mut gamma1 = Array2::<f64>::zeros((n_spatial, n_spatial));
    let mut gamma2 = Array4::<f64>::zeros((n_spatial, n_spatial, n_spatial, n_spatial));

    for p in 0..n_spatial {
        for q in p..n_spatial {
            let mut value = 0.0;
            for spin in 0..2 {
                value += measure_fermion_observable(
                    state,
                    &[
                        (2 * p + spin, FermionAction::Create),
                        (2 * q + spin, FermionAction::Annihilate),
                    ],
                    shots,
                    &mut rng,
                );
            }
            gamma1[[p, q]] = value;
            gamma1[[q, p]] = value;
        }
    }

    for p in 0..n_spatial {
        for q in 0..n_spatial {
            for r in 0..n_spatial {
                for s in 0..n_spatial {
                    let mut value = 0.0;
                    for spin1 in 0..2 {
                        for spin2 in 0..2 {
                            value += 0.5
                                * measure_fermion_observable(
                                    state,
                                    &[
                                        (2 * p + spin1, FermionAction::Create),
                                        (2 * q + spin2, FermionAction::Create),
                                        (2 * s + spin2, FermionAction::Annihilate),
                                        (2 * r + spin1, FermionAction::Annihilate),
                                    ],
                                    shots,
                                    &mut rng,
                                );
                        }
                    }
                    gamma2[[p, q, r, s]] = value;
                }
            }
        }
    }

    (gamma1, gamma2)
}

fn measure_fermion_observable(
    state: &[Complex64],
    operators: &[(usize, FermionAction)],
    shots: usize,
    rng: &mut StdRng,
) -> f64 {
    transform_hermitian_part(operators)
        .iter()
        .map(|(word, coefficient)| coefficient.re * measure_pauli_word(state, word, shots, rng))
        .sum()
}

fn measure_pauli_word(
    state: &[Complex64],
    word: &PauliWord,
    shots: usize,
    rng: &mut StdRng,
) -> f64 {
    if word.is_empty() {
        return 1.0;
    }

    let mut rotated = state.to_vec();
    for (&qubit, &pauli) in word {
        match pauli {
            Pauli::X => apply_single_qubit_gate(&mut rotated, qubit, hadamard()),
            Pauli::Y => apply_single_qubit_gate(&mut rotated, qubit, s_dagger_hadamard()),
            Pauli::Z => {}
        }
    }

    let probabilities: Vec<f64> = rotated
        .iter()
        .map(|amplitude| amplitude.norm_sqr())
        .collect();
    let distribution =
        WeightedIndex::new(&probabilities).expect("invalid state-vector probabilities");
    let mut sum = 0.0;
    for _ in 0..shots {
        let outcome = distribution.sample(rng);
        let odd_parity = word
            .keys()
            .filter(|&&qubit| outcome & (1usize << qubit) != 0)
            .count()
            % 2
            == 1;
        sum += if odd_parity { -1.0 } else { 1.0 };
    }
    sum / shots as f64
}

fn apply_single_qubit_gate(state: &mut [Complex64], qubit: usize, gate: [[Complex64; 2]; 2]) {
    let step = 1usize << qubit;
    let block = step << 1;
    for base in (0..state.len()).step_by(block) {
        for offset in 0..step {
            let zero = base + offset;
            let one = zero + step;
            let amplitude_zero = state[zero];
            let amplitude_one = state[one];
            state[zero] = gate[0][0] * amplitude_zero + gate[0][1] * amplitude_one;
            state[one] = gate[1][0] * amplitude_zero + gate[1][1] * amplitude_one;
        }
    }
}

fn hadamard() -> [[Complex64; 2]; 2] {
    let scale = 1.0 / 2.0_f64.sqrt();
    [
        [Complex64::new(scale, 0.0), Complex64::new(scale, 0.0)],
        [Complex64::new(scale, 0.0), Complex64::new(-scale, 0.0)],
    ]
}

fn s_dagger_hadamard() -> [[Complex64; 2]; 2] {
    let scale = 1.0 / 2.0_f64.sqrt();
    [
        [Complex64::new(scale, 0.0), Complex64::new(0.0, -scale)],
        [Complex64::new(scale, 0.0), Complex64::new(0.0, scale)],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chemistry::rdm::{compute_1rdm_spatial, compute_2rdm_spatial};

    fn exact_pauli_expectation(state: &[Complex64], word: &PauliWord) -> f64 {
        let mut transformed = state.to_vec();
        for (&qubit, &pauli) in word {
            let gate = match pauli {
                Pauli::X => [
                    [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                ],
                Pauli::Y => [
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
                    [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
                ],
                Pauli::Z => [
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
                ],
            };
            apply_single_qubit_gate(&mut transformed, qubit, gate);
        }
        state
            .iter()
            .zip(transformed)
            .map(|(left, right)| (left.conj() * right).re)
            .sum()
    }

    fn exact_fermion_observable(state: &[Complex64], operators: &[(usize, FermionAction)]) -> f64 {
        transform_hermitian_part(operators)
            .iter()
            .map(|(word, coefficient)| coefficient.re * exact_pauli_expectation(state, word))
            .sum()
    }

    #[test]
    fn finite_shot_rdms_converge_to_statevector_rdms() {
        let mut state = vec![Complex64::new(0.0, 0.0); 16];
        state[0b0011] = Complex64::new(0.6, 0.0);
        state[0b0110] = Complex64::new(0.3, 0.0);
        state[0b1001] = Complex64::new(-0.4, 0.0);
        state[0b1100] = Complex64::new(-0.6244997998398398, 0.0);

        let exact1 = compute_1rdm_spatial(&state, 2);
        let exact2 = compute_2rdm_spatial(&state, 2);
        let (sampled1, sampled2) = measure_spatial_rdms(&state, 2, 100_000, 1234);

        assert!((&sampled1 - &exact1).iter().all(|error| error.abs() < 0.02));
        let diff = &sampled2 - &exact2;
        assert!(diff.iter().all(|error| error.abs() < 0.02));
    }

    #[test]
    fn hermitian_pauli_observables_match_real_spatial_rdms() {
        let mut state = vec![Complex64::new(0.0, 0.0); 16];
        state[0b0011] = Complex64::new(0.6, 0.0);
        state[0b0110] = Complex64::new(0.3, 0.0);
        state[0b1001] = Complex64::new(-0.4, 0.0);
        state[0b1100] = Complex64::new(-0.6244997998398398, 0.0);
        let exact2 = compute_2rdm_spatial(&state, 2);

        for p in 0..2 {
            for q in 0..2 {
                for r in 0..2 {
                    for s in 0..2 {
                        let mut value = 0.0;
                        for spin1 in 0..2 {
                            for spin2 in 0..2 {
                                value += 0.5
                                    * exact_fermion_observable(
                                        &state,
                                        &[
                                            (2 * p + spin1, FermionAction::Create),
                                            (2 * q + spin2, FermionAction::Create),
                                            (2 * s + spin2, FermionAction::Annihilate),
                                            (2 * r + spin1, FermionAction::Annihilate),
                                        ],
                                    );
                            }
                        }
                        assert!((value - exact2[[p, q, r, s]]).abs() < 1e-12);
                    }
                }
            }
        }
    }

    #[test]
    fn finite_shot_rdms_are_seed_reproducible() {
        let scale = 1.0 / 2.0_f64.sqrt();
        let state = vec![Complex64::new(scale, 0.0), Complex64::new(scale, 0.0)];
        let first = measure_spatial_rdms(&state, 1, 1000, 42);
        let second = measure_spatial_rdms(&state, 1, 1000, 42);
        assert_eq!(first, second);
    }
}
