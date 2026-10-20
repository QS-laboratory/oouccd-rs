use crate::checkpoint::manager::Logger;
use crate::chemistry::engine::ClassicalEngine;
use crate::chemistry::molecular_context::MolecularContext;
use crate::config::OrbitalStatisticsConfig;
use crate::config::{
    AnsatzConfig, MethodConfig, OrbitalOptimizerConfig, OrbitalSolverConfig, RdmEvaluationConfig,
    SolverConfig, UccOptimizerConfig,
};
use crate::optimizer::lbfgs::finite_diff_gradient;
use crate::optimizer::newton::NewtonOrbitalOptimizer;
pub use crate::optimizer::orbital::{
    OrbitalOptimizer, OrbitalProblem, OrbitalRotation, OrbitalUpdate,
};
use crate::optimizer::statistical_oo::{self, OoStatisticsRecord};
use crate::optimizer::trigonometric::minimize_trigonometric_func;
use crate::quantum::engine::QuantumEngine;
use crate::quantum::rdm_statistics::{AffineConstraints, PauliSamples};
use crate::quantum::unitary_coupled_cluster_gates::Excitation;
use ndarray::Array1;
use std::f64::consts::PI;

use argmin::core::{CostFunction, Error, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;

pub struct Optimizer {
    pub orbital_statistics: OrbitalStatisticsConfig,
    pub affine_constraints: AffineConstraints,
    pub classical_engine: ClassicalEngine,
    pub quantum_engine: QuantumEngine,
    pub vqe_params: Array1<f64>,
    pub vqe_params_pairs: Vec<Excitation>,
    pub oo_params: Array1<f64>,
    pub oo_params_pairs: Vec<(usize, usize)>,
    pub max_macro_iterations: usize,
    pub energy_tolerance: f64,
    pub ucc_opt: Box<dyn UccOptimizer>,
    pub orb_opt: Option<Box<dyn OrbitalOptimizer>>,
}

impl Optimizer {
    pub fn build(ctx: &MolecularContext, method: &MethodConfig) -> anyhow::Result<Self> {
        method.orbital_statistics.validate()?;
        if let RdmEvaluationConfig::FiniteShot { shots, .. } = method.rdm_evaluation {
            anyhow::ensure!(shots > 0, "finite-shot RDM evaluation requires shots > 0");
        }
        let orb_opt = method
            .orbital_optimizer
            .as_ref()
            .map(build_orbital_optimizer)
            .transpose()?;
        let is_singles = method.ucc_optimizer.ansatz_config == AnsatzConfig::Uccsd;
        let quantum_engine = QuantumEngine::new(
            ctx.n_orb,
            ctx.n_elec,
            is_singles,
            method.rdm_evaluation.clone(),
        );
        let classical_engine = ClassicalEngine::from_context(ctx);

        let n_occ = ctx.n_elec / 2;
        let n_orb = ctx.n_orb;

        let vqe_params_pairs = quantum_engine.excitations.clone();
        let n_vqe = quantum_engine.n_params;

        let oo_params_pairs: Vec<(usize, usize)> = (0..n_occ)
            .flat_map(|i| (n_occ..n_orb).map(move |k| (i, k)))
            .collect();
        let n_oo = oo_params_pairs.len();

        let ucc_opt = build_ucc_optimizer(&method.ucc_optimizer);
        let affine_constraints =
            AffineConstraints::new(quantum_engine.measurement_plan(), ctx.n_elec);
        Ok(Self {
            orbital_statistics: method.orbital_statistics.clone(),
            affine_constraints,
            classical_engine,
            quantum_engine,
            vqe_params: Array1::zeros(n_vqe),
            vqe_params_pairs,
            oo_params: Array1::zeros(n_oo),
            oo_params_pairs,
            max_macro_iterations: method.max_macro_iterations.unwrap_or(50),
            energy_tolerance: method.energy_tolerance.unwrap_or(1e-6),
            ucc_opt,
            orb_opt,
        })
    }

    pub fn orbital_update_from_samples(
        &self,
        samples: &PauliSamples,
        macro_iteration: usize,
    ) -> anyhow::Result<(OrbitalUpdate, OoStatisticsRecord)> {
        statistical_oo::optimize(
            self.orb_opt
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("No orbital optimizer"))?
                .as_ref(),
            &self.classical_engine,
            &self.oo_params_pairs,
            self.quantum_engine.measurement_plan(),
            samples,
            &self.affine_constraints,
            &self.orbital_statistics,
            self.max_macro_iterations,
            macro_iteration,
        )
    }

    pub fn run(&mut self, logger: &mut Logger) -> anyhow::Result<()> {
        let ucc_names = (0..self.vqe_params.len())
            .map(|i| format!("theta_{}", i))
            .collect();
        let oo_names = self
            .oo_params_pairs
            .iter()
            .map(|(i, k)| format!("kappa_{}_{}", i, k))
            .collect();
        logger.register_parameters(ucc_names, oo_names);

        let (rdm1, rdm2) = self
            .quantum_engine
            .get_rdm(self.vqe_params.as_slice().unwrap());
        let mut prev_energy = self.classical_engine.evaluate_energy(&rdm1, &rdm2);

        for sweep in 1..=self.max_macro_iterations {
            let t_sweep = std::time::Instant::now();
            // --- Orbital Optimization ---
            if self.orb_opt.is_some() {
                eprintln!(
                    "[solver] sweep={sweep}/{} | OO start",
                    self.max_macro_iterations
                );
                let t_oo = std::time::Instant::now();
                let samples = self
                    .quantum_engine
                    .get_pauli_samples(self.vqe_params.as_slice().unwrap());
                let (update, statistics) = self.orbital_update_from_samples(&samples, sweep)?;
                eprintln!("[oo_statistics] sweep={sweep} accepted={} reason={} validation_delta={:?} upper={:?}",
                    statistics.accepted,statistics.reason,statistics.validation_delta_ha,statistics.validation_upper_bound_ha);
                if let Some(report) = &update.newton {
                    eprintln!(
                        "[newton] sweep={sweep} energy_before={:.12} energy_after={:.12} gradient_norm={:.6e} hessian_shift={:.6e} step_norm={:.6e} line_search_scale={:.6e} status={:?}",
                        update.energy_before, update.energy_after, report.gradient_norm,
                        report.hessian_shift, report.step_norm, report.line_search_scale, report.status,
                    );
                }
                update.apply(&mut self.classical_engine, &self.oo_params_pairs);
                self.oo_params.fill(0.0);
                eprintln!(
                    "[solver] sweep={sweep}/{} | OO done ({:.1}s)",
                    self.max_macro_iterations,
                    t_oo.elapsed().as_secs_f64()
                );
            }

            // --- UCC Optimization ---
            eprintln!(
                "[solver] sweep={sweep}/{} | UCC start",
                self.max_macro_iterations
            );
            let t_ucc = std::time::Instant::now();
            let quantum = &self.quantum_engine;
            let classical = &self.classical_engine;
            let mut eval_vqe = |theta: &[f64]| -> f64 {
                let (g1, g2) = quantum.get_rdm(theta);
                classical.evaluate_energy(&g1, &g2)
            };
            self.ucc_opt.optimize(&mut self.vqe_params, &mut eval_vqe);
            eprintln!(
                "[solver] sweep={sweep}/{} | UCC done ({:.1}s)",
                self.max_macro_iterations,
                t_ucc.elapsed().as_secs_f64()
            );

            let (rdm1, rdm2) = self
                .quantum_engine
                .get_rdm(self.vqe_params.as_slice().unwrap());
            let e = self.classical_engine.evaluate_energy(&rdm1, &rdm2);
            let delta = (prev_energy - e).abs();
            let converged = delta < self.energy_tolerance;
            eprintln!(
                "[solver] sweep={sweep}/{} | energy={e:.8} delta={delta:.2e} converged={converged} ({:.1}s total)",
                self.max_macro_iterations, t_sweep.elapsed().as_secs_f64(),
            );
            logger.save_iteration(sweep, 0, e, converged, &self.vqe_params, converged);

            if converged {
                return Ok(());
            }
            prev_energy = e;
        }

        let (rdm1, rdm2) = self
            .quantum_engine
            .get_rdm(self.vqe_params.as_slice().unwrap());
        let final_e = self.classical_engine.evaluate_energy(&rdm1, &rdm2);
        logger.save_iteration(
            self.max_macro_iterations,
            0,
            final_e,
            false,
            &self.vqe_params,
            true,
        );

        Ok(())
    }
}

pub trait UccOptimizer {
    fn optimize(&self, theta: &mut Array1<f64>, objective: &mut dyn FnMut(&[f64]) -> f64) -> f64;
}

fn build_ucc_optimizer(config: &UccOptimizerConfig) -> Box<dyn UccOptimizer> {
    match config.solver_config {
        SolverConfig::Sequential {} => Box::new(SequentialUccOptimizer {}),
        SolverConfig::Lbfgs { max_iter } => Box::new(LbfgsUccOptimizer { max_iter }),
    }
}

pub fn build_orbital_optimizer(
    config: &OrbitalOptimizerConfig,
) -> anyhow::Result<Box<dyn OrbitalOptimizer>> {
    Ok(match &config.solver_config {
        OrbitalSolverConfig::Sequential {} => Box::new(SequentialOrbitalOptimizer {}),
        OrbitalSolverConfig::Lbfgs { max_iter } => Box::new(LbfgsOrbitalOptimizer {
            max_iter: *max_iter,
        }),
        OrbitalSolverConfig::NewtonRaphson { options } => {
            options.validate()?;
            Box::new(NewtonOrbitalOptimizer {
                options: options.clone(),
            })
        }
    })
}

struct SequentialUccOptimizer;
impl UccOptimizer for SequentialUccOptimizer {
    fn optimize(&self, theta: &mut Array1<f64>, objective: &mut dyn FnMut(&[f64]) -> f64) -> f64 {
        let mut current_energy = objective(theta.as_slice().unwrap());

        for p in 0..theta.len() {
            let mut samples = ndarray::Array1::zeros(5);
            for j in 0..5 {
                let mut t = theta.to_vec();
                t[p] += (PI * 2.0 / 5.0) * (j as f64);
                samples[j] = objective(&t);
            }
            let (opt_shift, opt_energy) = minimize_trigonometric_func(&samples);
            theta[p] = (theta[p] + opt_shift + PI).rem_euclid(2.0 * PI) - PI;
            current_energy = opt_energy;
        }

        current_energy
    }
}

struct LbfgsUccOptimizer {
    max_iter: usize,
}
impl UccOptimizer for LbfgsUccOptimizer {
    fn optimize(&self, theta: &mut Array1<f64>, objective: &mut dyn FnMut(&[f64]) -> f64) -> f64 {
        let problem = ArgminProblem::new(objective, 1e-4);
        let init_param = theta.to_vec();

        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();
        let solver = LBFGS::new(linesearch, 10)
            .with_tolerance_grad(1e-6)
            .unwrap();

        let res = Executor::new(problem, solver)
            .configure(|state| state.param(init_param).max_iters(self.max_iter as u64))
            .run()
            .unwrap();

        let x_opt = res.state().get_best_param().unwrap().clone();
        let e_opt = res.state().get_best_cost();

        *theta = Array1::from_vec(x_opt);
        e_opt
    }
}

struct LbfgsOrbitalOptimizer {
    max_iter: usize,
}
impl OrbitalOptimizer for LbfgsOrbitalOptimizer {
    fn optimize(&self, orbital: &OrbitalProblem<'_>) -> anyhow::Result<OrbitalUpdate> {
        let mut objective = |kappa: &[f64]| orbital.energy_with_givens(kappa);
        let problem = ArgminProblem::new(&mut objective, 1e-5);
        let init_param = vec![0.0; orbital.pairs.len()];

        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();
        let solver = LBFGS::new(linesearch, 10)
            .with_tolerance_grad(1e-6)
            .unwrap();

        let res = Executor::new(problem, solver)
            .configure(|state| state.param(init_param).max_iters(self.max_iter as u64))
            .run()?;

        let x_opt = res.state().get_best_param().unwrap().clone();
        let e_opt = res.state().get_best_cost();

        Ok(OrbitalUpdate {
            rotation: OrbitalRotation::Givens(Array1::from_vec(x_opt)),
            energy_before: orbital.energy(),
            energy_after: e_opt,
            newton: None,
        })
    }
}

struct SequentialOrbitalOptimizer;
impl OrbitalOptimizer for SequentialOrbitalOptimizer {
    fn optimize(&self, problem: &OrbitalProblem<'_>) -> anyhow::Result<OrbitalUpdate> {
        let mut kappa = Array1::<f64>::zeros(problem.pairs.len());
        let n = kappa.len();
        let energy_before = problem.energy();

        for p in 0..n {
            // E(κ_p) is a degree-4 Fourier polynomial (m=2 spin components → 2m=4 frequencies).
            // Requires 2*4+1 = 9 samples to uniquely determine all coefficients.
            let mut samples = Array1::zeros(9);
            for j in 0..9 {
                let mut kappa_trial = kappa.to_vec();
                kappa_trial[p] = kappa[p] + (PI * 2.0 / 9.0) * (j as f64);
                samples[j] = problem.energy_with_givens(&kappa_trial);
            }
            let (opt_shift, _) = minimize_trigonometric_func(&samples);
            kappa[p] = (kappa[p] + opt_shift + PI).rem_euclid(2.0 * PI) - PI;
        }

        let energy_after = problem.energy_with_givens(kappa.as_slice().unwrap());
        Ok(OrbitalUpdate {
            rotation: OrbitalRotation::Givens(kappa),
            energy_before,
            energy_after,
            newton: None,
        })
    }
}

// Wrapper struct to make FnMut closures work with argmin's CostFunction and Gradient traits
use std::cell::RefCell;

struct ArgminProblem<'a> {
    objective: RefCell<&'a mut dyn FnMut(&[f64]) -> f64>,
    finite_diff_eps: f64,
}

impl<'a> ArgminProblem<'a> {
    fn new(objective: &'a mut dyn FnMut(&[f64]) -> f64, finite_diff_eps: f64) -> Self {
        Self {
            objective: RefCell::new(objective),
            finite_diff_eps,
        }
    }
}

impl<'a> CostFunction for ArgminProblem<'a> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok((self.objective.borrow_mut())(param))
    }
}

impl<'a> Gradient for ArgminProblem<'a> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        let grad = finite_diff_gradient(
            param,
            &mut *self.objective.borrow_mut(),
            self.finite_diff_eps,
        );
        Ok(grad)
    }
}
