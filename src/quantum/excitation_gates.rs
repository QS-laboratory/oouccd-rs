use roqoqo::operations::CNOT;
use roqoqo::Circuit;

/// Constructs a leftward-propogating CNOT ladder, commonly used in Jordan-Wigner mappings.
/// Example: CX(3,2) -> CX(2,1) -> CX(1,0)
pub fn naive_left_cx_ladder(qubits: &[usize]) -> Circuit {
    let mut circuit = Circuit::new();
    if qubits.len() < 2 {
        return circuit;
    }
    for idx in (1..qubits.len()).rev() {
        circuit += CNOT::new(qubits[idx], qubits[idx - 1]);
    }
    circuit
}

/// Constructs a rightward-propogating CNOT ladder, reversing the left-ladder.
/// Example: CX(1,0) -> CX(2,1) -> CX(3,2)
pub fn naive_right_cx_ladder(qubits: &[usize]) -> Circuit {
    let mut circuit = Circuit::new();
    if qubits.len() < 2 {
        return circuit;
    }
    for idx in 1..qubits.len() {
        circuit += CNOT::new(qubits[idx], qubits[idx - 1]);
    }
    circuit
}

/// Yordanov single-excitation left ladder. Drops the outer boundary qubits before mapping.
pub fn yordanov_single_left_cx_ladder(qubits: &[usize]) -> Circuit {
    if qubits.len() < 4 {
        return Circuit::new();
    }
    naive_left_cx_ladder(&qubits[1..qubits.len() - 1])
}

/// Yordanov single-excitation right ladder. Drops the outer boundary qubits before mapping.
pub fn yordanov_single_right_cx_ladder(qubits: &[usize]) -> Circuit {
    if qubits.len() < 4 {
        return Circuit::new();
    }
    naive_right_cx_ladder(&qubits[1..qubits.len() - 1])
}

/// Yordanov double-excitation left ladder. Extracts the internal Jordan-Wigner parity strings,
/// discarding the interacting bounds (i, j, k, l).
pub fn yordanov_double_left_cx_ladder(qubits: &[usize], j_minus_i: usize) -> Circuit {
    let mut pbits = Vec::new();

    // Extract internal string between i and j (exclusive)
    if j_minus_i > 1 {
        pbits.extend_from_slice(&qubits[1..j_minus_i]);
    }

    // Extract internal string between k and l (exclusive)
    let k_index = j_minus_i + 1;
    if qubits.len() > k_index + 1 {
        pbits.extend_from_slice(&qubits[(k_index + 1)..(qubits.len() - 1)]);
    }

    naive_left_cx_ladder(&pbits)
}

/// Yordanov double-excitation right ladder. Extracts the internal Jordan-Wigner parity strings.
pub fn yordanov_double_right_cx_ladder(qubits: &[usize], j_minus_i: usize) -> Circuit {
    let mut pbits = Vec::new();

    if j_minus_i > 1 {
        pbits.extend_from_slice(&qubits[1..j_minus_i]);
    }

    let k_index = j_minus_i + 1;
    if qubits.len() > k_index + 1 {
        pbits.extend_from_slice(&qubits[(k_index + 1)..(qubits.len() - 1)]);
    }

    naive_right_cx_ladder(&pbits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_naive_cx_ladders() {
        let qubits = vec![0, 1, 2, 3];

        let left_circ = naive_left_cx_ladder(&qubits);
        // Should produce CX(3,2), CX(2,1), CX(1,0) = 3 ops
        assert_eq!(left_circ.iter().count(), 3);

        let right_circ = naive_right_cx_ladder(&qubits);
        // Should produce CX(1,0), CX(2,1), CX(3,2) = 3 ops
        assert_eq!(right_circ.iter().count(), 3);
    }

    #[test]
    fn test_yordanov_ladders() {
        let qubits = vec![0, 1, 2, 3, 4, 5];

        // Single Yordanov drops outer bounds
        let left_single = yordanov_single_left_cx_ladder(&qubits);
        // Using indices [1, 2, 3, 4] -> length 4, ladder has 3 CNOTs
        assert_eq!(left_single.iter().count(), 3);

        // Double Yordanov gap testing
        // For q=[0, 1, 2, 3, 4, 5], j_minus_i = 2 -> i=0, j=2, k=3, l=5
        // internal strings: (0 to 2) -> 1 element (1). (3 to 5) -> 1 element (4).
        // Array of 2 elements total.
        let left_double = yordanov_double_left_cx_ladder(&qubits, 2);
        assert_eq!(left_double.iter().count(), 1); // Only 1 CX for 2 elements
    }
}
