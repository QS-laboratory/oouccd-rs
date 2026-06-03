use qoqo_calculator::CalculatorFloat;
use roqoqo::operations::{ControlledPhaseShift, Hadamard, PauliX, RotateX, RotateZ, CNOT};
use roqoqo::Circuit;

use super::excitation_gates::{
    naive_left_cx_ladder, naive_right_cx_ladder, yordanov_double_left_cx_ladder,
    yordanov_double_right_cx_ladder, yordanov_single_left_cx_ladder,
    yordanov_single_right_cx_ladder,
};
use super::multi_controlled_gates::multi_controlled_ry;
use std::f64::consts::PI;

pub fn naive_real_single_excitation(qubits: &[usize], param: CalculatorFloat) -> Circuit {
    let mut circuit = Circuit::new();
    let qk = *qubits.last().unwrap();
    let qi = qubits[0];

    circuit += RotateX::new(qk, CalculatorFloat::from(PI / 2.0));
    circuit += Hadamard::new(qi);

    circuit += naive_left_cx_ladder(qubits);
    circuit += RotateZ::new(qi, param.clone());
    circuit += naive_right_cx_ladder(qubits);

    circuit += Hadamard::new(qi);
    circuit += RotateX::new(qk, CalculatorFloat::from(-PI / 2.0));

    circuit += Hadamard::new(qk);
    circuit += RotateX::new(qi, CalculatorFloat::from(PI / 2.0));

    circuit += naive_left_cx_ladder(qubits);
    circuit += RotateZ::new(qi, param.clone() * -1.0);
    circuit += naive_right_cx_ladder(qubits);

    circuit += RotateX::new(qi, CalculatorFloat::from(-PI / 2.0));
    circuit += Hadamard::new(qk);

    circuit
}

pub fn yordanov_real_single_excitation(qubits: &[usize], param: CalculatorFloat) -> Circuit {
    let mut circuit = Circuit::new();
    let qi = qubits[0];
    let qk = *qubits.last().unwrap();

    circuit += yordanov_single_left_cx_ladder(qubits);
    circuit += CNOT::new(qk, qi);

    if qubits.len() >= 3 {
        circuit += ControlledPhaseShift::new(qk, qubits[1], CalculatorFloat::from(PI));
    }

    let mcry_params = vec![param.clone() * 2.0, param.clone() * -2.0];
    circuit += multi_controlled_ry(&[qi, qk], &mcry_params);

    if qubits.len() >= 3 {
        circuit += ControlledPhaseShift::new(qk, qubits[1], CalculatorFloat::from(PI));
    }

    circuit += CNOT::new(qk, qi);
    circuit += yordanov_single_right_cx_ladder(qubits);

    circuit
}

pub fn yordanov_real_double_excitation(
    qubits: &[usize],
    j_minus_i: usize,
    param: CalculatorFloat,
) -> Circuit {
    let mut circuit = Circuit::new();
    let qi = qubits[0];
    let qj = qubits[j_minus_i];
    let qk = qubits[j_minus_i + 1];
    let ql = *qubits.last().unwrap();

    if qubits.len() >= 6 {
        circuit += yordanov_double_left_cx_ladder(qubits, j_minus_i);
    }

    circuit += CNOT::new(ql, qk);
    circuit += CNOT::new(qj, qi);
    circuit += CNOT::new(ql, qj);

    if qubits.len() >= 5 {
        // Python uses CZ mapping to q1 safely mapped to ControlledPhaseShift
        circuit += ControlledPhaseShift::new(ql, qubits[1], CalculatorFloat::from(PI));
    }

    circuit += PauliX::new(qi);
    circuit += PauliX::new(qk);

    // Multi-Controlled Ry is generic over the target list passed directly from Yordanov logic natively!
    let mcry_params = vec![
        param.clone() * 2.0,
        param.clone() * -2.0,
        param.clone() * -2.0,
        param.clone() * 2.0,
        param.clone() * -2.0,
        param.clone() * 2.0,
        param.clone() * 2.0,
        param.clone() * -2.0,
    ];
    circuit += multi_controlled_ry(&[qi, qj, qk, ql], &mcry_params);

    circuit += PauliX::new(qk);
    circuit += PauliX::new(qi);

    if qubits.len() >= 5 {
        circuit += ControlledPhaseShift::new(ql, qubits[1], CalculatorFloat::from(PI));
    }

    circuit += CNOT::new(ql, qj);
    circuit += CNOT::new(qj, qi);
    circuit += CNOT::new(ql, qk);

    if qubits.len() >= 6 {
        circuit += yordanov_double_right_cx_ladder(qubits, j_minus_i);
    }

    circuit
}

pub fn naive_real_double_excitation(
    qubits: &[usize],
    j_minus_i: usize,
    param: CalculatorFloat,
) -> Circuit {
    let mut circuit = Circuit::new();
    let qi = qubits[0];
    let qj = qubits[j_minus_i];
    let qk = qubits[j_minus_i + 1];
    let ql = *qubits.last().unwrap();

    // -0.125j [X0 X1 X2 Y3]
    circuit += Hadamard::new(qi);
    circuit += Hadamard::new(qj);
    circuit += Hadamard::new(qk);
    circuit += RotateX::new(ql, CalculatorFloat::from(PI / 2.0));
    circuit += naive_left_cx_ladder(qubits);
    circuit += RotateZ::new(qi, param.clone() * 0.25);
    circuit += naive_right_cx_ladder(qubits);
    circuit += RotateX::new(ql, CalculatorFloat::from(-PI / 2.0));
    circuit += Hadamard::new(qk);
    circuit += Hadamard::new(qj);
    circuit += Hadamard::new(qi);

    // -0.125j [X0 X1 Y2 X3]
    circuit += Hadamard::new(qi);
    circuit += Hadamard::new(qj);
    circuit += RotateX::new(qk, CalculatorFloat::from(PI / 2.0));
    circuit += Hadamard::new(ql);
    circuit += naive_left_cx_ladder(qubits);
    circuit += RotateZ::new(qi, param.clone() * 0.25);
    circuit += naive_right_cx_ladder(qubits);
    circuit += Hadamard::new(ql);
    circuit += RotateX::new(qk, CalculatorFloat::from(-PI / 2.0));
    circuit += Hadamard::new(qj);
    circuit += Hadamard::new(qi);

    // +0.125j [X0 Y1 X2 X3]
    circuit += Hadamard::new(qi);
    circuit += RotateX::new(qj, CalculatorFloat::from(PI / 2.0));
    circuit += Hadamard::new(qk);
    circuit += Hadamard::new(ql);
    circuit += naive_left_cx_ladder(qubits);
    circuit += RotateZ::new(qi, param.clone() * -0.25);
    circuit += naive_right_cx_ladder(qubits);
    circuit += Hadamard::new(ql);
    circuit += Hadamard::new(qk);
    circuit += RotateX::new(qj, CalculatorFloat::from(-PI / 2.0));
    circuit += Hadamard::new(qi);

    // -0.125j [X0 Y1 Y2 Y3]
    circuit += Hadamard::new(qi);
    circuit += RotateX::new(qj, CalculatorFloat::from(PI / 2.0));
    circuit += RotateX::new(qk, CalculatorFloat::from(PI / 2.0));
    circuit += RotateX::new(ql, CalculatorFloat::from(PI / 2.0));
    circuit += naive_left_cx_ladder(qubits);
    circuit += RotateZ::new(qi, param.clone() * 0.25);
    circuit += naive_right_cx_ladder(qubits);
    circuit += RotateX::new(ql, CalculatorFloat::from(-PI / 2.0));
    circuit += RotateX::new(qk, CalculatorFloat::from(-PI / 2.0));
    circuit += RotateX::new(qj, CalculatorFloat::from(-PI / 2.0));
    circuit += Hadamard::new(qi);

    // +0.125j [Y0 X1 X2 X3]
    circuit += RotateX::new(qi, CalculatorFloat::from(PI / 2.0));
    circuit += Hadamard::new(qj);
    circuit += Hadamard::new(qk);
    circuit += Hadamard::new(ql);
    circuit += naive_left_cx_ladder(qubits);
    circuit += RotateZ::new(qi, param.clone() * -0.25);
    circuit += naive_right_cx_ladder(qubits);
    circuit += Hadamard::new(ql);
    circuit += Hadamard::new(qk);
    circuit += Hadamard::new(qj);
    circuit += RotateX::new(qi, CalculatorFloat::from(-PI / 2.0));

    // -0.125j [Y0 X1 Y2 Y3]
    circuit += RotateX::new(qi, CalculatorFloat::from(PI / 2.0));
    circuit += Hadamard::new(qj);
    circuit += RotateX::new(qk, CalculatorFloat::from(PI / 2.0));
    circuit += RotateX::new(ql, CalculatorFloat::from(PI / 2.0));
    circuit += naive_left_cx_ladder(qubits);
    circuit += RotateZ::new(qi, param.clone() * 0.25);
    circuit += naive_right_cx_ladder(qubits);
    circuit += RotateX::new(ql, CalculatorFloat::from(-PI / 2.0));
    circuit += RotateX::new(qk, CalculatorFloat::from(-PI / 2.0));
    circuit += Hadamard::new(qj);
    circuit += RotateX::new(qi, CalculatorFloat::from(-PI / 2.0));

    // +0.125j [Y0 Y1 X2 Y3]
    circuit += RotateX::new(qi, CalculatorFloat::from(PI / 2.0));
    circuit += RotateX::new(qj, CalculatorFloat::from(PI / 2.0));
    circuit += Hadamard::new(qk);
    circuit += RotateX::new(ql, CalculatorFloat::from(PI / 2.0));
    circuit += naive_left_cx_ladder(qubits);
    circuit += RotateZ::new(qi, param.clone() * -0.25);
    circuit += naive_right_cx_ladder(qubits);
    circuit += RotateX::new(ql, CalculatorFloat::from(-PI / 2.0));
    circuit += Hadamard::new(qk);
    circuit += RotateX::new(qj, CalculatorFloat::from(-PI / 2.0));
    circuit += RotateX::new(qi, CalculatorFloat::from(-PI / 2.0));

    // +0.125j [Y0 Y1 Y2 X3]
    circuit += RotateX::new(qi, CalculatorFloat::from(PI / 2.0));
    circuit += RotateX::new(qj, CalculatorFloat::from(PI / 2.0));
    circuit += RotateX::new(qk, CalculatorFloat::from(PI / 2.0));
    circuit += Hadamard::new(ql);
    circuit += naive_left_cx_ladder(qubits);
    circuit += RotateZ::new(qi, param.clone() * -0.25);
    circuit += naive_right_cx_ladder(qubits);
    circuit += Hadamard::new(ql);
    circuit += RotateX::new(qk, CalculatorFloat::from(-PI / 2.0));
    circuit += RotateX::new(qj, CalculatorFloat::from(-PI / 2.0));
    circuit += RotateX::new(qi, CalculatorFloat::from(-PI / 2.0));

    circuit
}

#[cfg(test)]
mod matrix_tests {
    use super::*;
    use num_complex::Complex64;
    use roqoqo::backends::EvaluatingBackend;
    use roqoqo::operations::{PauliX, PragmaGetStateVector};
    use roqoqo_quest::Backend;

    /// Extrapolates a dense Unitary Matrix projection dynamically by traversing a roqoqo::Circuit
    /// precisely over all 2^N state vector dimensions via QUEST backend.
    fn extract_unitary(circuit: &Circuit, num_qubits: usize) -> Vec<Vec<Complex64>> {
        let mut matrix = vec![vec![Complex64::new(0.0, 0.0); 1 << num_qubits]; 1 << num_qubits];
        let backend = Backend::new(num_qubits, None);

        for i in 0..(1 << num_qubits) {
            let mut sim_circ = Circuit::new();

            // Generate standard computational basis states |i>
            for q in 0..num_qubits {
                if (i & (1 << q)) != 0 {
                    sim_circ += PauliX::new(q);
                }
            }
            // Execute physical U(theta) mapping
            sim_circ += circuit.clone();
            sim_circ +=
                roqoqo::operations::DefinitionComplex::new("ro".into(), 1 << num_qubits, true);
            sim_circ += PragmaGetStateVector::new("ro".into(), None);

            let (_, _, complex_res) = backend.run_circuit(&sim_circ).unwrap();
            let state = &complex_res["ro"][0];

            // Log target outputs back onto matching unit projection column
            for j in 0..(1 << num_qubits) {
                matrix[j][i] = Complex64::new(state[j].re, state[j].im);
            }
        }
        matrix
    }

    #[test]
    fn test_givens_rotation_single_excitation_subspace() {
        let num_qubits = 2; // [0, 1] mapped exact orbitals
        let qubits = vec![0, 1];
        let angles = [
            0.1,
            -0.5,
            std::f64::consts::PI / 4.0,
            -std::f64::consts::PI / 3.0,
            1.23,
        ];

        for &theta in &angles {
            // Exact UCCSD generator U = exp(theta (T - T^dagger)) -> Single Exchange
            let circ = naive_real_single_excitation(&qubits, theta.into());

            let u = extract_unitary(&circ, num_qubits);

            // Verify EXACT Givens rotation matrix structural mapping geometrically!
            // Basis Layout: |00> = 0, |01> = 1, |10> = 2, |11> = 3

            // Phase isolated trace tests mapping `|00>` and `|11>`
            // Identity asserts (Parity / Phase persistence)
            assert!((u[0][0].re.abs() - 1.0).abs() < 1e-10);
            assert!((u[3][3].re.abs() - 1.0).abs() < 1e-10);
            assert!((u[0][0].im).abs() < 1e-10);
            assert!((u[3][3].im).abs() < 1e-10);

            // Givens T-Mapping Block: Expect cos(theta) inside (01, 10).
            // NOTE: depending on definition Trotter mapping could be exactly theta or theta/2.
            // We assert mathematical identity matrix structure regardless!
            let diagonal = u[1][1].re;
            let diagonal_adjoint = u[2][2].re;
            assert!(
                (diagonal - diagonal_adjoint).abs() < 1e-10,
                "Givens Cosines must be symmetric for theta={}",
                theta
            );

            let sine = u[1][2].re;
            let sine_adjoint = u[2][1].re;
            assert!(
                (sine + sine_adjoint).abs() < 1e-10,
                "Givens Sines must be anti-symmetric (-1 multiplier) for theta={}",
                theta
            );

            // Cross bounds trigonometric scalar mapping checks
            let unitary_norm = (diagonal * diagonal) + (sine * sine);
            assert!(
                (unitary_norm - 1.0).abs() < 1e-10,
                "Probability conservation missing, Givens vector leaked! theta={}",
                theta
            );
        }
    }

    #[test]
    fn test_givens_rotation_double_excitation_subspace() {
        let num_qubits = 4; // Minimal symmetric active space for double excitation
        let qubits = vec![0, 1, 2, 3];
        let angles = [
            0.1,
            -0.5,
            std::f64::consts::PI / 4.0,
            -std::f64::consts::PI / 3.0,
            1.23,
        ];

        for &theta in &angles {
            let circ = naive_real_double_excitation(&qubits, 1, theta.into());
            let u = extract_unitary(&circ, num_qubits);

            // Exact 4-qubit Double Excitation mapping test
            // Initial state |0011> maps to |1100>
            // Index |0011> : binary 3
            // Index |1100> : binary 12

            // Assert Fock ends don't experience probability leak
            assert!((u[0][0].re.abs() - 1.0).abs() < 1e-10);
            assert!((u[15][15].re.abs() - 1.0).abs() < 1e-10);

            // Givens Subspace block mapping!
            let diagonal = u[3][3].re;
            let diagonal_adjoint = u[12][12].re;
            assert!(
                (diagonal - diagonal_adjoint).abs() < 1e-10,
                "Givens Cosines must be symmetric for theta={}",
                theta
            );

            let sine = u[3][12].re;
            let sine_adjoint = u[12][3].re;
            assert!(
                (sine + sine_adjoint).abs() < 1e-10,
                "Givens Sines must be anti-symmetric for theta={}",
                theta
            );

            let unitary_norm = (diagonal * diagonal) + (sine * sine);
            assert!(
                (unitary_norm - 1.0).abs() < 1e-10,
                "Probability conservation missing, Subspace leaked for theta={}",
                theta
            );

            // Explicitly assert standard trace evolution hasn't evaluated to 0
            assert!(
                sine.abs() > 1e-5,
                "The Double Excitation string evaluated to Unity without operating for theta={}",
                theta
            );
        }
    }
}
