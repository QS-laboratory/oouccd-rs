use crate::chemistry::rdm::{compute_1rdm_spatial, compute_2rdm_spatial};
use crate::quantum::unitary_coupled_cluster_gates::{build_ucc_ansatz, Excitation};
use ndarray::{Array2, Array4};
use qoqo_calculator::Calculator;
use roqoqo::backends::EvaluatingBackend;
use roqoqo::operations::*;
use roqoqo::Circuit;
use roqoqo_quest::Backend;



pub struct QuantumEngine {
    pub ansatz_circuit: Circuit,
    backend: Backend,
    pub n_qubits: usize,
    pub n_params: usize,
    pub excitations: Vec<Excitation>,
}

impl QuantumEngine {
    pub fn new(n_orb: usize, n_elec: usize, include_singles: bool) -> Self {
        let n_qubits = n_orb * 2;
        let (ansatz_circuit, excitations, n_params) =
            build_ucc_ansatz(n_elec, n_orb, false, false, include_singles);
        let backend = Backend::new(n_qubits, None);

        Self {
            ansatz_circuit,
            backend,
            n_qubits,
            n_params,
            excitations,
        }
    }

    pub fn get_rdm(&self, theta: &[f64]) -> (Array2<f64>, Array4<f64>) {
        let mut calc = Calculator::new();
        for (i, &t) in theta.iter().enumerate() {
            calc.set_variable(&format!("theta_{}", i), t);
        }

        let mut eval_circuit = self.ansatz_circuit.substitute_parameters(&calc).unwrap();
        eval_circuit += DefinitionComplex::new("ro".into(), 1, true);
        eval_circuit += PragmaGetStateVector::new("ro".into(), None);

        let (_, _, res_complex) = self.backend.run_circuit(&eval_circuit).unwrap();
        let state_vector = res_complex.get("ro").unwrap().get(0).unwrap();

        let n_spatial = self.n_qubits / 2;
        let gamma1 = compute_1rdm_spatial(state_vector, n_spatial);
        let gamma2 = compute_2rdm_spatial(state_vector, n_spatial);

        (gamma1, gamma2)
    }
}
