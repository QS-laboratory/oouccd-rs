use ndarray::{Array2, Array4};

/// Pure data container for molecular integrals computed by PySCF.
/// h1 and h2 are in the original (unrotated) MO basis.
pub struct MolecularContext {
    pub n_orb: usize,
    pub n_elec: usize,
    pub nuclear_repulsion: f64,
    pub h1: Array2<f64>,
    pub h2: Array4<f64>,
}
