use crate::chemistry::orbital_derivatives::orbital_gradient_hessian;
use crate::config::NewtonOptions;
use crate::optimizer::orbital::{
    NewtonDiagnostics, NewtonStatus, OrbitalOptimizer, OrbitalProblem, OrbitalRotation,
    OrbitalUpdate,
};
use crate::optimizer::utils::exponential_orbital_rotation;
use anyhow::{ensure, Context, Result};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use ndarray::{Array1, Array2};

pub struct NewtonOrbitalOptimizer {
    pub options: NewtonOptions,
}

fn solve_direction(
    gradient: &Array1<f64>,
    hessian: &Array2<f64>,
    options: &NewtonOptions,
) -> Result<(Vec<f64>, f64)> {
    ensure!(
        gradient.iter().chain(hessian.iter()).all(|v| v.is_finite()),
        "Newton gradient/Hessian contains non-finite values"
    );
    let n = gradient.len();
    ensure!(
        hessian.dim() == (n, n),
        "Newton Hessian dimensions do not match gradient"
    );
    let mut matrix = DMatrix::from_fn(n, n, |i, j| 0.5 * (hessian[[i, j]] + hessian[[j, i]]));
    let rhs = DVector::from_iterator(n, gradient.iter().map(|g| -g));
    let mut shift = 0.0;
    let mut direction = if options.stabilized {
        let eigen = SymmetricEigen::try_new(matrix.clone(), f64::EPSILON, 10_000)
            .context("Newton Hessian eigendecomposition did not converge")?;
        let minimum = eigen.eigenvalues.min();
        shift = (options.hessian_floor - minimum).max(0.0);
        for i in 0..n {
            matrix[(i, i)] += shift;
        }
        matrix
            .cholesky()
            .context("Shifted Newton Hessian is not positive definite")?
            .solve(&rhs)
    } else {
        matrix
            .full_piv_lu()
            .solve(&rhs)
            .context("Singular Newton Hessian (stabilization is disabled)")?
    };
    ensure!(
        direction.iter().all(|v| v.is_finite()),
        "Non-finite Newton direction"
    );
    let norm = direction.norm();
    ensure!(norm.is_finite(), "Non-finite Newton step norm");
    if options.stabilized && norm > options.max_step_norm {
        direction *= options.max_step_norm / norm;
    }
    Ok((direction.iter().copied().collect(), shift))
}

/// All energies here belong to one frozen-RDM objective, with no fresh samples.
fn backtrack(
    direction: &[f64],
    gradient: &Array1<f64>,
    energy: f64,
    max_backtracks: usize,
    mut evaluate: impl FnMut(&[f64]) -> f64,
) -> Option<(Vec<f64>, f64, f64)> {
    let slope: f64 = gradient.iter().zip(direction).map(|(g, d)| g * d).sum();
    if !slope.is_finite() || slope >= 0.0 {
        return None;
    }
    let mut scale = 1.0;
    for _ in 0..=max_backtracks {
        let step: Vec<_> = direction.iter().map(|d| scale * d).collect();
        let trial = evaluate(&step);
        if trial.is_finite() && trial <= energy + 1e-4 * scale * slope {
            return Some((step, trial, scale));
        }
        scale *= 0.5;
    }
    None
}

impl OrbitalOptimizer for NewtonOrbitalOptimizer {
    fn optimize(&self, problem: &OrbitalProblem<'_>) -> Result<OrbitalUpdate> {
        self.options.validate()?;
        let energy = problem.energy();
        ensure!(energy.is_finite(), "Non-finite energy before Newton OO");
        let mut diagnostics = NewtonDiagnostics {
            gradient_norm: 0.0,
            hessian_shift: 0.0,
            step_norm: 0.0,
            line_search_scale: 0.0,
            status: NewtonStatus::NoPairs,
        };
        let unchanged = |newton| OrbitalUpdate {
            rotation: OrbitalRotation::Unchanged,
            energy_before: energy,
            energy_after: energy,
            newton: Some(newton),
        };
        if problem.pairs.is_empty() {
            return Ok(unchanged(diagnostics));
        }
        let (gradient, hessian) =
            orbital_gradient_hessian(problem.engine, problem.rdm1, problem.rdm2, problem.pairs);
        ensure!(
            gradient.iter().chain(hessian.iter()).all(|v| v.is_finite()),
            "Newton gradient/Hessian contains non-finite values"
        );
        diagnostics.gradient_norm = gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        ensure!(
            diagnostics.gradient_norm.is_finite(),
            "Non-finite Newton gradient norm"
        );
        if diagnostics.gradient_norm <= self.options.gradient_tolerance {
            diagnostics.status = NewtonStatus::GradientConverged;
            return Ok(unchanged(diagnostics));
        }
        let (direction, shift) = solve_direction(&gradient, &hessian, &self.options)?;
        diagnostics.hessian_shift = shift;
        let n_orb = problem.engine.h1.nrows();
        let evaluate = |step: &[f64]| {
            let rotation = exponential_orbital_rotation(n_orb, problem.pairs, step);
            if rotation.iter().all(|v| v.is_finite()) {
                problem.energy_with_rotation(&rotation)
            } else {
                f64::NAN
            }
        };
        let selected = if self.options.stabilized {
            backtrack(
                &direction,
                &gradient,
                energy,
                self.options.max_backtracks,
                evaluate,
            )
        } else {
            let trial = evaluate(&direction);
            ensure!(trial.is_finite(), "Non-finite energy after raw Newton OO");
            Some((direction, trial, 1.0))
        };
        let Some((step, trial, scale)) = selected else {
            diagnostics.status = NewtonStatus::LineSearchRejected;
            return Ok(unchanged(diagnostics));
        };
        diagnostics.step_norm = step.iter().map(|v| v * v).sum::<f64>().sqrt();
        diagnostics.line_search_scale = scale;
        diagnostics.status = NewtonStatus::Accepted;
        Ok(OrbitalUpdate {
            rotation: OrbitalRotation::Matrix(exponential_orbital_rotation(
                n_orb,
                problem.pairs,
                &step,
            )),
            energy_before: energy,
            energy_after: trial,
            newton: Some(diagnostics),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chemistry::engine::ClassicalEngine;
    use ndarray::{array, Array4};

    #[test]
    fn raw_newton_solves_positive_hessian_without_step_limit() {
        let options = NewtonOptions {
            stabilized: false,
            ..Default::default()
        };
        let (d, shift) = solve_direction(
            &array![1.0, -2.0],
            &array![[2.0, 0.0], [0.0, 4.0]],
            &options,
        )
        .unwrap();
        assert_eq!(shift, 0.0);
        assert!((d[0] + 0.5).abs() < 1e-12 && (d[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn indefinite_hessian_is_shifted_and_step_is_limited() {
        let options = NewtonOptions {
            hessian_floor: 0.1,
            ..Default::default()
        };
        let (d, shift) = solve_direction(
            &array![1.0, -2.0],
            &array![[-1.0, 0.0], [0.0, 0.0]],
            &options,
        )
        .unwrap();
        assert!((shift - 1.1).abs() < 1e-12);
        assert!(
            (d.iter().map(|v| v * v).sum::<f64>().sqrt() - options.max_step_norm).abs() < 1e-12
        );
        assert!(d[0] - 2.0 * d[1] < 0.0);
    }

    #[test]
    fn singular_raw_hessian_and_nonfinite_derivatives_are_errors() {
        let raw = NewtonOptions {
            stabilized: false,
            ..Default::default()
        };
        assert!(solve_direction(&array![1.0], &array![[0.0]], &raw).is_err());
        assert!(solve_direction(&array![1.0], &array![[0.0]], &NewtonOptions::default()).is_ok());
        assert!(solve_direction(&array![f64::NAN], &array![[1.0]], &raw).is_err());
        assert!(solve_direction(
            &array![1.0],
            &array![[f64::INFINITY]],
            &NewtonOptions::default()
        )
        .is_err());
    }

    #[test]
    fn line_search_backtracks_or_rejects_without_an_uphill_fallback() {
        let g = array![1.0];
        let (step, e, scale) = backtrack(&[-2.0], &g, 0.0, 20, |x| x[0] + x[0] * x[0]).unwrap();
        assert!(scale < 1.0 && e < 0.0 && step[0] < 0.0);
        assert!(backtrack(&[-1.0], &g, 0.0, 20, |_| 0.0).is_none());
        assert!(backtrack(&[-1.0], &g, 0.0, 20, |_| f64::NAN).is_none());
    }

    #[test]
    fn frozen_rdm_update_decreases_energy_and_handles_zero_gradient_and_no_pairs() {
        let mut engine = ClassicalEngine {
            h1: array![[-1.0, 0.3], [0.3, 0.5]],
            h2: Array4::zeros((2, 2, 2, 2)),
            nuclear_repulsion: 0.0,
        };
        let rdm1 = array![[2.0, 0.0], [0.0, 0.0]];
        let rdm2 = Array4::zeros((2, 2, 2, 2));
        let pairs = [(0, 1)];
        let optimizer = NewtonOrbitalOptimizer {
            options: NewtonOptions::default(),
        };
        let problem = OrbitalProblem {
            engine: &engine,
            rdm1: &rdm1,
            rdm2: &rdm2,
            pairs: &pairs,
        };
        let update = optimizer.optimize(&problem).unwrap();
        assert!(update.energy_after < update.energy_before);
        assert_eq!(
            update.newton.as_ref().unwrap().status,
            NewtonStatus::Accepted
        );
        let reported = update.energy_after;
        update.apply(&mut engine, &pairs);
        assert!((engine.evaluate_energy(&rdm1, &rdm2) - reported).abs() < 1e-12);
        engine.h1 = array![[-1.0, 0.0], [0.0, 0.5]];
        let mut problem = OrbitalProblem {
            engine: &engine,
            rdm1: &rdm1,
            rdm2: &rdm2,
            pairs: &pairs,
        };
        let update = optimizer.optimize(&problem).unwrap();
        assert_eq!(
            update.newton.unwrap().status,
            NewtonStatus::GradientConverged
        );
        problem.pairs = &[];
        assert_eq!(
            optimizer.optimize(&problem).unwrap().newton.unwrap().status,
            NewtonStatus::NoPairs
        );
    }
}
