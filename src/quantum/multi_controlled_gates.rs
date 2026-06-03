use qoqo_calculator::CalculatorFloat;
use roqoqo::operations::{PhaseShift, RotateY, CNOT};
use roqoqo::Circuit;

/// Internal recursive synthesizer for multi-controlled Ry gates
fn sub_unit_ry(tbit: usize, cbits: &[usize], params: &[CalculatorFloat]) -> Circuit {
    let mut circuit = Circuit::new();

    if params.len() == 1 {
        circuit += RotateY::new(tbit, params[0].clone());
    } else {
        let mid = params.len() / 2;
        let cbits_head = &cbits[..cbits.len() - 1];
        let c_last = cbits[cbits.len() - 1];

        // First half: 0.5 * params[..mid]
        let mut first_half = Vec::with_capacity(mid);
        for i in 0..mid {
            first_half.push(params[i].clone() * 0.5);
        }
        circuit += sub_unit_ry(tbit, cbits_head, &first_half);

        circuit += CNOT::new(c_last, tbit);

        // Second half: -0.5 * params[mid..]
        let mut second_half = Vec::with_capacity(mid);
        for i in 0..mid {
            second_half.push(params[mid + i].clone() * -0.5);
        }
        circuit += sub_unit_ry(tbit, cbits_head, &second_half);

        circuit += CNOT::new(c_last, tbit);
    }
    circuit
}

/// Helper gate generation for a dynamically sized multi-controlled Ry rotation string
pub fn multi_controlled_ry(qubits: &[usize], params: &[CalculatorFloat]) -> Circuit {
    let tbit = qubits
        .last()
        .copied()
        .expect("Qubits slice cannot be empty");
    let cbits = &qubits[..qubits.len() - 1];

    // Total parameters must be exactly 2^(controls)
    assert_eq!(
        params.len(),
        1 << cbits.len(),
        "Incorrect number of parameters for multi-controlled decomposition"
    );

    sub_unit_ry(tbit, cbits, params)
}

/// Helper gate generation for a dynamically sized multi-controlled Rx rotation string
pub fn multi_controlled_rx(qubits: &[usize], params: &[CalculatorFloat]) -> Circuit {
    let tbit = qubits
        .last()
        .copied()
        .expect("Qubits slice cannot be empty");
    let mut circuit = Circuit::new();

    // S gate
    circuit += PhaseShift::new(tbit, CalculatorFloat::from(std::f64::consts::PI / 2.0));

    // MC Ry
    circuit += multi_controlled_ry(qubits, params);

    // S_dag gate
    circuit += PhaseShift::new(tbit, CalculatorFloat::from(-std::f64::consts::PI / 2.0));

    circuit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_controlled_ry() {
        let qubits = vec![0, 1, 2]; // 2 controls, 1 target
        let params: Vec<CalculatorFloat> = vec![0.1.into(), 0.2.into(), 0.3.into(), 0.4.into()]; // 2^2 = 4 params required

        let circ = multi_controlled_ry(&qubits, &params);
        assert!(circ.iter().count() > 0);
    }

    #[test]
    fn test_multi_controlled_rx() {
        let qubits = vec![0, 1, 2, 3]; // 3 controls, 1 target
        let params: Vec<CalculatorFloat> = vec![
            0.1.into(),
            0.2.into(),
            0.3.into(),
            0.4.into(),
            0.5.into(),
            0.6.into(),
            0.7.into(),
            0.8.into(),
        ]; // 2^3 = 8 params

        let circ = multi_controlled_rx(&qubits, &params);
        assert!(circ.iter().count() > 0);
    }
}
