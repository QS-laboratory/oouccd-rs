use crate::chemistry::rdm::{compute_1rdm_spatial, compute_2rdm_spatial};
use crate::config::RdmEvaluationConfig;
use crate::quantum::rdm_measurement::RdmMeasurementPlan;
use crate::quantum::unitary_coupled_cluster_gates::{build_ucc_ansatz, Excitation};
use ndarray::{Array2, Array4};
use num_complex::Complex64;
use qoqo_calculator::Calculator;
use roqoqo::backends::EvaluatingBackend;
use roqoqo::operations::*;
use roqoqo::Circuit;
use roqoqo_quest::Backend;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct QuantumEngine {
    pub ansatz_circuit: Circuit,
    backend: Backend,
    pub n_qubits: usize,
    pub n_params: usize,
    pub excitations: Vec<Excitation>,
    rdm_evaluation: RdmEvaluationConfig,
    rdm_measurement_plan: Option<RdmMeasurementPlan>,
    evaluation_count: AtomicU64,
}

impl QuantumEngine {
    pub fn new(
        n_orb: usize,
        n_elec: usize,
        include_singles: bool,
        rdm_evaluation: RdmEvaluationConfig,
    ) -> Self {
        let n_qubits = n_orb * 2;
        let (ansatz_circuit, excitations, n_params) =
            build_ucc_ansatz(n_elec, n_orb, false, false, include_singles);
        let backend = Backend::new(n_qubits, None);
        let rdm_measurement_plan = match rdm_evaluation {
            RdmEvaluationConfig::Statevector {} => None,
            RdmEvaluationConfig::FiniteShot { .. } => Some(RdmMeasurementPlan::new(n_orb)),
        };

        Self {
            ansatz_circuit,
            backend,
            n_qubits,
            n_params,
            excitations,
            rdm_evaluation,
            rdm_measurement_plan,
            evaluation_count: AtomicU64::new(0),
        }
    }

    pub fn get_rdm(&self, theta: &[f64]) -> (Array2<f64>, Array4<f64>) {
        let state_vector = self.get_state_vector(theta);
        let n_spatial = self.n_qubits / 2;

        match self.rdm_evaluation {
            RdmEvaluationConfig::Statevector {} => (
                compute_1rdm_spatial(&state_vector, n_spatial),
                compute_2rdm_spatial(&state_vector, n_spatial),
            ),
            RdmEvaluationConfig::FiniteShot { shots, seed } => {
                let evaluation = self.evaluation_count.fetch_add(1, Ordering::Relaxed);
                let base_seed = seed.unwrap_or_else(rand::random);
                let measurement_seed = base_seed.wrapping_add(evaluation);
                self.rdm_measurement_plan
                    .as_ref()
                    .expect("finite-shot mode requires an RDM measurement plan")
                    .measure(&state_vector, shots, measurement_seed)
            }
        }
    }

    pub fn get_state_vector(&self, theta: &[f64]) -> Vec<Complex64> {
        let mut calc = Calculator::new();
        for (i, &t) in theta.iter().enumerate() {
            calc.set_variable(&format!("theta_{}", i), t);
        }

        let mut eval_circuit = self.ansatz_circuit.substitute_parameters(&calc).unwrap();
        eval_circuit += DefinitionComplex::new("ro".into(), 1, true);
        eval_circuit += PragmaGetStateVector::new("ro".into(), None);

        let (_, _, res_complex) = self.backend.run_circuit(&eval_circuit).unwrap();
        res_complex.get("ro").unwrap().get(0).unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statevector_and_finite_shot_modes_are_switchable() {
        let exact_engine = QuantumEngine::new(2, 2, false, RdmEvaluationConfig::Statevector {});
        let sampled_engine = QuantumEngine::new(
            2,
            2,
            false,
            RdmEvaluationConfig::FiniteShot {
                shots: 100_000,
                seed: Some(1234),
            },
        );
        let theta = vec![0.0; exact_engine.n_params];

        let (exact1, exact2) = exact_engine.get_rdm(&theta);
        let (sampled1, sampled2) = sampled_engine.get_rdm(&theta);

        assert!((&sampled1 - &exact1).iter().all(|error| error.abs() < 0.02));
        assert!((&sampled2 - &exact2).iter().all(|error| error.abs() < 0.02));
    }
}
