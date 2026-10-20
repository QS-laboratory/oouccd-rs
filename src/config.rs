use serde::{Deserialize, Serialize};
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

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SolverConfig {
    Sequential {},
    Lbfgs { max_iter: usize },
}

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum AnsatzConfig {
    #[serde(rename = "UCCD")]
    Uccd,
    #[serde(rename = "UCCSD")]
    Uccsd,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(default)]
pub struct NewtonOptions {
    pub stabilized: bool,
    pub hessian_floor: f64,
    pub max_step_norm: f64,
    pub gradient_tolerance: f64,
    pub max_backtracks: usize,
}

impl Default for NewtonOptions {
    fn default() -> Self {
        Self {
            stabilized: true,
            hessian_floor: 1e-4,
            max_step_norm: 0.2,
            gradient_tolerance: 1e-6,
            max_backtracks: 20,
        }
    }
}

impl NewtonOptions {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.gradient_tolerance.is_finite() && self.gradient_tolerance >= 0.0,
            "Newton gradient_tolerance must be finite and nonnegative"
        );
        if self.stabilized {
            anyhow::ensure!(
                self.hessian_floor.is_finite() && self.hessian_floor > 0.0,
                "Newton hessian_floor must be finite and positive"
            );
            anyhow::ensure!(
                self.max_step_norm.is_finite() && self.max_step_norm > 0.0,
                "Newton max_step_norm must be finite and positive"
            );
        }
        Ok(())
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OrbitalSolverConfig {
    Sequential {},
    Lbfgs {
        max_iter: usize,
    },
    NewtonRaphson {
        #[serde(flatten)]
        options: NewtonOptions,
    },
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OrbitalOptimizerConfig {
    pub solver_config: OrbitalSolverConfig,
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RdmMitigation {
    #[default]
    None,
    Affine,
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OoAcceptance {
    #[default]
    Nominal,
    SplitNominal,
    Heldout,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(default)]
pub struct OrbitalStatisticsConfig {
    pub mitigation: RdmMitigation,
    pub acceptance: OoAcceptance,
    pub training_fraction: f64,
    pub trajectory_alpha: f64,
    pub min_improvement_ha: f64,
    pub split_seed: u64,
    pub record_counts: bool,
}

impl Default for OrbitalStatisticsConfig {
    fn default() -> Self {
        Self {
            mitigation: RdmMitigation::None,
            acceptance: OoAcceptance::Nominal,
            training_fraction: 0.8,
            trajectory_alpha: 0.05,
            min_improvement_ha: 0.0,
            split_seed: 0x9e3779b97f4a7c15,
            record_counts: false,
        }
    }
}

impl OrbitalStatisticsConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.training_fraction.is_finite()
                && self.training_fraction > 0.0
                && self.training_fraction < 1.0,
            "OO training_fraction must lie strictly between zero and one"
        );
        anyhow::ensure!(
            self.trajectory_alpha.is_finite()
                && self.trajectory_alpha > 0.0
                && self.trajectory_alpha < 1.0,
            "OO trajectory_alpha must lie strictly between zero and one"
        );
        anyhow::ensure!(
            self.min_improvement_ha.is_finite() && self.min_improvement_ha >= 0.0,
            "OO min_improvement_ha must be finite and nonnegative"
        );
        Ok(())
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
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
    #[serde(default)]
    pub orbital_statistics: OrbitalStatisticsConfig,
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
        conditions.insert(
            "ucc_optimizer".to_string(),
            serde_json::json!(self.method.ucc_optimizer),
        );
        conditions.insert(
            "orbital_optimizer".to_string(),
            serde_json::json!(self.method.orbital_optimizer),
        );
        conditions.insert(
            "orbital_statistics".to_string(),
            serde_json::json!(self.method.orbital_statistics),
        );
        conditions.insert(
            "max_macro_iterations".to_string(),
            serde_json::json!(self.method.max_macro_iterations.unwrap_or(50)),
        );
        conditions.insert(
            "energy_tolerance".to_string(),
            serde_json::json!(self.method.energy_tolerance.unwrap_or(1e-6)),
        );

        conditions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbital_solver_json_is_compatible_and_newton_defaults_are_recorded() {
        for json in [
            r#"{"type":"sequential"}"#,
            r#"{"type":"lbfgs","max_iter":50}"#,
        ] {
            let parsed: OrbitalSolverConfig = serde_json::from_str(json).unwrap();
            assert_eq!(
                serde_json::to_value(parsed).unwrap(),
                serde_json::from_str::<Value>(json).unwrap()
            );
        }
        let parsed: OrbitalSolverConfig =
            serde_json::from_str(r#"{"type":"newton_raphson"}"#).unwrap();
        let OrbitalSolverConfig::NewtonRaphson { options } = &parsed else {
            panic!("wrong solver")
        };
        assert!(options.stabilized);
        assert_eq!(options.hessian_floor, 1e-4);
        assert_eq!(options.max_step_norm, 0.2);
        assert_eq!(options.gradient_tolerance, 1e-6);
        assert_eq!(options.max_backtracks, 20);
        assert!(options.validate().is_ok());
        let serialized = serde_json::to_value(parsed).unwrap();
        assert_eq!(serialized["stabilized"], true);
        assert_eq!(serialized["max_backtracks"], 20);
        assert!(serde_json::from_str::<SolverConfig>(r#"{"type":"newton_raphson"}"#).is_err());
    }

    #[test]
    fn run_conditions_distinguish_newton_settings_even_with_identical_labels() {
        let mut task: TaskConfig = serde_json::from_str(r#"{
            "task_name":"test", "chemistry":{"geometry":"H 0 0 0; H 0 0 0.735","basis":"sto-3g"},
            "method":{"label":"same", "ucc_optimizer":{"ansatz_config":{"type":"UCCD"},"solver_config":{"type":"sequential"}},
            "orbital_optimizer":{"solver_config":{"type":"newton_raphson"}}}
        }"#).unwrap();
        let stable = task.generate_conditions();
        let OrbitalSolverConfig::NewtonRaphson { options } = &mut task
            .method
            .orbital_optimizer
            .as_mut()
            .unwrap()
            .solver_config
        else {
            unreachable!()
        };
        options.stabilized = false;
        assert_ne!(stable, task.generate_conditions());
        assert_eq!(
            stable["orbital_optimizer"]["solver_config"]["hessian_floor"],
            1e-4
        );
    }

    #[test]
    fn invalid_newton_settings_fail_validation() {
        let mut options = NewtonOptions::default();
        options.hessian_floor = 0.0;
        assert!(options.validate().is_err());
        options = NewtonOptions::default();
        options.max_step_norm = f64::NAN;
        assert!(options.validate().is_err());
        options = NewtonOptions::default();
        options.gradient_tolerance = -1.0;
        assert!(options.validate().is_err());
    }
}
