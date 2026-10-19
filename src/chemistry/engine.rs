use crate::chemistry::molecular_context::MolecularContext;
use crate::optimizer::utils::{rotate_1body_integrals, rotate_2body_integrals, rotate_tensor};
use ndarray::{Array2, Array4};

pub struct ClassicalEngine {
    pub h1: Array2<f64>,
    pub h2: Array4<f64>,
    pub nuclear_repulsion: f64,
}

impl ClassicalEngine {
    pub fn rotated_integrals(&self, rotation: &Array2<f64>) -> (Array2<f64>, Array4<f64>) {
        (
            rotate_tensor(&self.h1, rotation),
            rotate_tensor(&self.h2, rotation),
        )
    }

    pub fn apply_orbital_rotation(&mut self, rotation: &Array2<f64>) {
        (self.h1, self.h2) = self.rotated_integrals(rotation);
    }

    pub fn from_context(ctx: &MolecularContext) -> Self {
        Self {
            h1: ctx.h1.clone(),
            h2: ctx.h2.clone(),
            nuclear_repulsion: ctx.nuclear_repulsion,
        }
    }

    pub fn evaluate_energy(&self, rdm1: &Array2<f64>, rdm2: &Array4<f64>) -> f64 {
        self.nuclear_repulsion + (&self.h1 * rdm1).sum() + (&self.h2 * rdm2).sum()
    }

    /// Non-destructive trial evaluation: applies kappa rotations to a clone of h1/h2
    /// and evaluates energy without modifying the engine's state.
    pub fn evaluate_energy_with_kappa(
        &self,
        rdm1: &Array2<f64>,
        rdm2: &Array4<f64>,
        pairs: &[(usize, usize)],
        kappa: &[f64],
    ) -> f64 {
        let mut h1 = self.h1.clone();
        let mut h2 = self.h2.clone();
        for (p, &angle) in kappa.iter().enumerate() {
            let (i, k) = pairs[p];
            rotate_1body_integrals(&mut h1, i, k, angle);
            rotate_2body_integrals(&mut h2, i, k, angle);
        }
        self.nuclear_repulsion + (&h1 * rdm1).sum() + (&h2 * rdm2).sum()
    }

    pub fn rotate_electron_integrals(&mut self, i: usize, k: usize, angle: f64) {
        rotate_1body_integrals(&mut self.h1, i, k, angle);
        rotate_2body_integrals(&mut self.h2, i, k, angle);
    }
}
