use roqoqo::operations::PauliX;
use roqoqo::Circuit;

use super::real_excitation_gates::{
    naive_real_double_excitation, naive_real_single_excitation, yordanov_real_double_excitation,
    yordanov_real_single_excitation,
};

#[derive(Debug, Clone)]
pub enum Excitation {
    Single(usize, usize),                       // (i, a) spin-orbital indices
    Double(usize, usize, usize, usize),          // (i, j, a, b) spin-orbital indices
}

/// Builds a parameterized Unitary Coupled Cluster circuit.
/// Returns the circuit, the excitation metadata (one entry per parameter), and the parameter count.
pub fn build_ucc_ansatz(
    n_active_electrons: usize,
    n_active_orbitals: usize,
    use_yordanov: bool,
    pair: bool,
    include_singles: bool,
) -> (Circuit, Vec<Excitation>, usize) {
    let mut circuit = Circuit::new();
    let nbits = 2 * n_active_orbitals;

    // 1. Prepare Initial HF State
    for i in 0..n_active_electrons {
        circuit += PauliX::new(i);
    }

    let mut siter = Vec::new();
    let mut diter = Vec::new();

    if pair {
        for a in (n_active_electrons / 2)..n_active_orbitals {
            for i in 0..(n_active_electrons / 2) {
                diter.push((2 * i, 2 * i + 1, 2 * a, 2 * a + 1));
            }
        }
    } else {
        if include_singles {
            for i in 0..n_active_electrons {
                for k in n_active_electrons..nbits {
                    if (k + i) % 2 == 0 {
                        siter.push((i, k));
                    }
                }
            }
        }
        for i in 0..n_active_electrons {
            for j in (i + 1)..n_active_electrons {
                for k in n_active_electrons..nbits {
                    for l in (k + 1)..nbits {
                        if (i & 1) + (j & 1) == (k & 1) + (l & 1) {
                            diter.push((i, j, k, l));
                        }
                    }
                }
            }
        }
    }

    let mut excitations = Vec::new();
    let mut param_count = 0;

    for (i, j, k, l) in &diter {
        let mut qubits = Vec::new();
        qubits.extend(*i..=*j);
        qubits.extend(*k..=*l);
        let j_minus_i = j - i;
        let symbol = qoqo_calculator::CalculatorFloat::from(format!("theta_{}", param_count));

        if use_yordanov {
            circuit += yordanov_real_double_excitation(&qubits, j_minus_i, symbol);
        } else {
            circuit += naive_real_double_excitation(&qubits, j_minus_i, symbol);
        }
        excitations.push(Excitation::Double(*i, *j, *k, *l));
        param_count += 1;
    }

    for (i, k) in &siter {
        let mut qubits = Vec::new();
        qubits.extend(*i..=*k);
        let symbol = qoqo_calculator::CalculatorFloat::from(format!("theta_{}", param_count));

        if use_yordanov {
            circuit += yordanov_real_single_excitation(&qubits, symbol);
        } else {
            circuit += naive_real_single_excitation(&qubits, symbol);
        }
        excitations.push(Excitation::Single(*i, *k));
        param_count += 1;
    }

    (circuit, excitations, param_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_ucc_ansatz_pair() {
        // H2 in sto-3g corresponds to 2 electrons in 2 spatial orbitals
        let (circuit, excitations, param_count) = build_ucc_ansatz(2, 2, false, true, false);

        // pair=true generates EXACTLY 1 double excitation term for H2 sto-3g
        assert_eq!(param_count, 1);
        assert_eq!(excitations.len(), 1);
        assert!(matches!(excitations[0], Excitation::Double(_, _, _, _)));

        // Assert circuit produces HF state (X ops) + UCC string
        assert!(circuit.iter().count() > 50);
    }
}
