use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Deserialize, Debug, Clone)]
pub struct ChemistryConfig {
    pub geometry: String,
    pub basis: String,
    #[serde(default)]
    pub charge: i32,
    #[serde(default)]
    pub spin: i32,
    #[serde(default)]
    pub freeze_core: usize,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SolverConfig {
    Sequential {},
    Lbfgs { max_iter: usize },
}

#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum AnsatzConfig {
    #[serde(rename = "UCCD")]
    Uccd,
    #[serde(rename = "UCCSD")]
    Uccsd,
}

#[derive(Deserialize, Debug, Clone)]
pub struct OrbitalOptimizerConfig {
    pub solver_config: SolverConfig,
}

#[derive(Deserialize, Debug, Clone)]
pub struct UccOptimizerConfig {
    pub solver_config: SolverConfig,
    pub ansatz_config: AnsatzConfig,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RdmEvaluationConfig {
    Statevector {},
    FiniteShot { shots: usize, seed: Option<u64> },
}

impl Default for RdmEvaluationConfig {
    fn default() -> Self {
        Self::Statevector {}
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct MethodConfig {
    pub label: String,
    pub max_macro_iterations: Option<usize>,
    pub energy_tolerance: Option<f64>,
    pub ucc_optimizer: UccOptimizerConfig,
    pub orbital_optimizer: Option<OrbitalOptimizerConfig>,
    #[serde(default)]
    pub rdm_evaluation: RdmEvaluationConfig,
}

#[derive(Deserialize, Debug, Clone)]
pub struct TaskConfig {
    pub task_name: String,
    pub chemistry: ChemistryConfig,
    pub method: MethodConfig,
}

impl TaskConfig {
    pub fn generate_conditions(&self) -> BTreeMap<String, Value> {
        let mut conditions = BTreeMap::new();
        conditions.insert(
            "task_name".to_string(),
            serde_json::json!(self.task_name.clone()),
        );
        conditions.insert(
            "geometry".to_string(),
            serde_json::json!(self.chemistry.geometry.clone()),
        );
        conditions.insert(
            "basis".to_string(),
            serde_json::json!(self.chemistry.basis.clone()),
        );
        conditions.insert(
            "freeze_core".to_string(),
            serde_json::json!(self.chemistry.freeze_core),
        );
        conditions.insert(
            "method".to_string(),
            serde_json::json!(self.method.label.clone()),
        );
        conditions.insert(
            "rdm_evaluation".to_string(),
            match &self.method.rdm_evaluation {
                RdmEvaluationConfig::Statevector {} => serde_json::json!({
                    "type": "statevector",
                }),
                RdmEvaluationConfig::FiniteShot { shots, seed } => serde_json::json!({
                    "type": "finite_shot",
                    "shots": shots,
                    "seed": seed,
                }),
            },
        );

        conditions
    }
}
