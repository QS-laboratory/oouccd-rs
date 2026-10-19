//! Frozen-RDM orbital problems. No quantum backend is accessed during OO.
use crate::chemistry::engine::ClassicalEngine;
use ndarray::{Array1, Array2, Array4};

pub struct OrbitalProblem<'a> {
    pub engine: &'a ClassicalEngine,
    pub rdm1: &'a Array2<f64>,
    pub rdm2: &'a Array4<f64>,
    pub pairs: &'a [(usize, usize)],
}

impl OrbitalProblem<'_> {
    pub fn energy(&self) -> f64 {
        self.engine.evaluate_energy(self.rdm1, self.rdm2)
    }

    pub fn energy_with_givens(&self, kappa: &[f64]) -> f64 {
        self.engine
            .evaluate_energy_with_kappa(self.rdm1, self.rdm2, self.pairs, kappa)
    }

    pub fn energy_with_rotation(&self, rotation: &Array2<f64>) -> f64 {
        let (h1, h2) = self.engine.rotated_integrals(rotation);
        self.engine.nuclear_repulsion + (&h1 * self.rdm1).sum() + (&h2 * self.rdm2).sum()
    }
}

pub enum OrbitalRotation {
    /// Preserve the existing order and arithmetic of sequential / L-BFGS updates.
    Givens(Array1<f64>),
    Matrix(Array2<f64>),
    Unchanged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewtonStatus {
    Accepted,
    GradientConverged,
    NoPairs,
    LineSearchRejected,
}

#[derive(Debug, Clone)]
pub struct NewtonDiagnostics {
    pub gradient_norm: f64,
    pub hessian_shift: f64,
    pub step_norm: f64,
    pub line_search_scale: f64,
    pub status: NewtonStatus,
}

pub struct OrbitalUpdate {
    pub rotation: OrbitalRotation,
    pub energy_before: f64,
    pub energy_after: f64,
    pub newton: Option<NewtonDiagnostics>,
}

impl OrbitalUpdate {
    pub fn apply(&self, engine: &mut ClassicalEngine, pairs: &[(usize, usize)]) {
        match &self.rotation {
            OrbitalRotation::Givens(angles) => {
                assert_eq!(angles.len(), pairs.len());
                for (&(i, j), &angle) in pairs.iter().zip(angles) {
                    engine.rotate_electron_integrals(i, j, angle);
                }
            }
            OrbitalRotation::Matrix(rotation) => engine.apply_orbital_rotation(rotation),
            OrbitalRotation::Unchanged => {}
        }
    }
}

pub trait OrbitalOptimizer {
    fn optimize(&self, problem: &OrbitalProblem<'_>) -> anyhow::Result<OrbitalUpdate>;
}
