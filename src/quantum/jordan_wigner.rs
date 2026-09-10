use num_complex::Complex64;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FermionAction {
    Create,
    Annihilate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Pauli {
    X,
    Y,
    Z,
}

pub type PauliWord = BTreeMap<usize, Pauli>;
pub type PauliPolynomial = BTreeMap<PauliWord, Complex64>;

/// Jordan-Wigner transforms an ordered fermionic product such as
/// a†_p a†_q a_s a_r into a Pauli polynomial.
pub fn transform(operators: &[(usize, FermionAction)]) -> PauliPolynomial {
    let mut polynomial = PauliPolynomial::from([(PauliWord::new(), Complex64::new(1.0, 0.0))]);
    for &(index, action) in operators {
        polynomial = multiply_polynomials(&polynomial, &transform_single(index, action));
    }
    polynomial.retain(|_, coefficient| coefficient.norm() > 1e-12);
    polynomial
}

/// Returns the Hermitian part (O + O†) / 2 of a fermionic product.
/// This is the observable required for the real-valued RDM convention used here.
pub fn transform_hermitian_part(operators: &[(usize, FermionAction)]) -> PauliPolynomial {
    let polynomial = transform(operators);
    polynomial
        .into_iter()
        .map(|(word, coefficient)| (word, Complex64::new(coefficient.re, 0.0)))
        .filter(|(_, coefficient)| coefficient.norm() > 1e-12)
        .collect()
}

fn transform_single(index: usize, action: FermionAction) -> PauliPolynomial {
    let z_prefix: PauliWord = (0..index).map(|qubit| (qubit, Pauli::Z)).collect();
    let mut x_word = z_prefix.clone();
    x_word.insert(index, Pauli::X);
    let mut y_word = z_prefix;
    y_word.insert(index, Pauli::Y);

    let y_coefficient = match action {
        FermionAction::Create => Complex64::new(0.0, -0.5),
        FermionAction::Annihilate => Complex64::new(0.0, 0.5),
    };
    PauliPolynomial::from([(x_word, Complex64::new(0.5, 0.0)), (y_word, y_coefficient)])
}

fn multiply_polynomials(left: &PauliPolynomial, right: &PauliPolynomial) -> PauliPolynomial {
    let mut result = PauliPolynomial::new();
    for (left_word, left_coefficient) in left {
        for (right_word, right_coefficient) in right {
            let (word, phase) = multiply_words(left_word, right_word);
            *result.entry(word).or_insert(Complex64::new(0.0, 0.0)) +=
                left_coefficient * right_coefficient * phase;
        }
    }
    result
}

fn multiply_words(left: &PauliWord, right: &PauliWord) -> (PauliWord, Complex64) {
    let mut result = left.clone();
    let mut phase = Complex64::new(1.0, 0.0);
    for (&qubit, &right_pauli) in right {
        match result.get(&qubit).copied() {
            None => {
                result.insert(qubit, right_pauli);
            }
            Some(left_pauli) if left_pauli == right_pauli => {
                result.remove(&qubit);
            }
            Some(left_pauli) => {
                let (product, local_phase) = multiply_paulis(left_pauli, right_pauli);
                result.insert(qubit, product);
                phase *= local_phase;
            }
        }
    }
    (result, phase)
}

fn multiply_paulis(left: Pauli, right: Pauli) -> (Pauli, Complex64) {
    match (left, right) {
        (Pauli::X, Pauli::Y) => (Pauli::Z, Complex64::new(0.0, 1.0)),
        (Pauli::Y, Pauli::X) => (Pauli::Z, Complex64::new(0.0, -1.0)),
        (Pauli::Y, Pauli::Z) => (Pauli::X, Complex64::new(0.0, 1.0)),
        (Pauli::Z, Pauli::Y) => (Pauli::X, Complex64::new(0.0, -1.0)),
        (Pauli::Z, Pauli::X) => (Pauli::Y, Complex64::new(0.0, 1.0)),
        (Pauli::X, Pauli::Z) => (Pauli::Y, Complex64::new(0.0, -1.0)),
        _ => unreachable!("equal Pauli operators are handled before multiplication"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chemistry::rdm::{expect_1body, expect_2body};

    fn word(entries: &[(usize, Pauli)]) -> PauliWord {
        entries.iter().copied().collect()
    }

    fn apply_pauli_word(state: &[Complex64], word: &PauliWord) -> Vec<Complex64> {
        let mut result = vec![Complex64::new(0.0, 0.0); state.len()];
        for (basis, amplitude) in state.iter().enumerate() {
            let mut target = basis;
            let mut phase = Complex64::new(1.0, 0.0);
            for (&qubit, &pauli) in word {
                let occupied = basis & (1usize << qubit) != 0;
                match pauli {
                    Pauli::X => target ^= 1usize << qubit,
                    Pauli::Y => {
                        target ^= 1usize << qubit;
                        phase *= if occupied {
                            Complex64::new(0.0, -1.0)
                        } else {
                            Complex64::new(0.0, 1.0)
                        };
                    }
                    Pauli::Z => {
                        if occupied {
                            phase = -phase;
                        }
                    }
                }
            }
            result[target] += phase * amplitude;
        }
        result
    }

    fn expectation(state: &[Complex64], polynomial: &PauliPolynomial) -> Complex64 {
        polynomial
            .iter()
            .map(|(word, coefficient)| {
                let transformed = apply_pauli_word(state, word);
                coefficient
                    * state
                        .iter()
                        .zip(transformed)
                        .map(|(left, right)| left.conj() * right)
                        .sum::<Complex64>()
            })
            .sum()
    }

    #[test]
    fn one_body_matches_openfermion_reference() {
        let transformed = transform(&[(0, FermionAction::Create), (2, FermionAction::Annihilate)]);
        let expected = PauliPolynomial::from([
            (
                word(&[(0, Pauli::X), (1, Pauli::Z), (2, Pauli::X)]),
                Complex64::new(0.25, 0.0),
            ),
            (
                word(&[(0, Pauli::X), (1, Pauli::Z), (2, Pauli::Y)]),
                Complex64::new(0.0, 0.25),
            ),
            (
                word(&[(0, Pauli::Y), (1, Pauli::Z), (2, Pauli::X)]),
                Complex64::new(0.0, -0.25),
            ),
            (
                word(&[(0, Pauli::Y), (1, Pauli::Z), (2, Pauli::Y)]),
                Complex64::new(0.25, 0.0),
            ),
        ]);
        assert_eq!(transformed, expected);
    }

    #[test]
    fn number_operator_matches_openfermion_reference() {
        let transformed = transform(&[(0, FermionAction::Create), (0, FermionAction::Annihilate)]);
        let expected = PauliPolynomial::from([
            (PauliWord::new(), Complex64::new(0.5, 0.0)),
            (word(&[(0, Pauli::Z)]), Complex64::new(-0.5, 0.0)),
        ]);
        assert_eq!(transformed, expected);
    }

    #[test]
    fn hermitian_part_has_only_real_coefficients() {
        let transformed =
            transform_hermitian_part(&[(0, FermionAction::Create), (2, FermionAction::Annihilate)]);
        assert!(transformed
            .values()
            .all(|coefficient| coefficient.im == 0.0));
        assert_eq!(transformed.len(), 2);
    }

    #[test]
    fn transformed_rdm_operators_match_existing_statevector_implementation() {
        let mut state = vec![Complex64::new(0.0, 0.0); 16];
        state[0b0011] = Complex64::new(0.6, 0.0);
        state[0b0110] = Complex64::new(0.0, 0.3);
        state[0b1001] = Complex64::new(-0.4, 0.0);
        state[0b1100] = Complex64::new(0.0, -0.6244997998398398);

        for p in 0..4 {
            for q in 0..4 {
                let transformed =
                    transform(&[(p, FermionAction::Create), (q, FermionAction::Annihilate)]);
                let jw_value = expectation(&state, &transformed).re;
                assert!((jw_value - expect_1body(&state, p, q)).abs() < 1e-12);

                for r in 0..4 {
                    for s in 0..4 {
                        let transformed = transform(&[
                            (p, FermionAction::Create),
                            (q, FermionAction::Create),
                            (r, FermionAction::Annihilate),
                            (s, FermionAction::Annihilate),
                        ]);
                        let jw_value = expectation(&state, &transformed).re;
                        assert!((jw_value - expect_2body(&state, p, q, r, s)).abs() < 1e-12);
                    }
                }
            }
        }
    }
}
