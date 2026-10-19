use crate::chemistry::molecular_context::MolecularContext;
use crate::config::ChemistryConfig;
use anyhow::{Context, Result};
use ndarray::{Array2, Array4};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

pub struct PyScfDriver;

impl PyScfDriver {
    pub fn compute(config: &ChemistryConfig) -> Result<MolecularContext> {
        Python::with_gil(|py| -> Result<MolecularContext> {
            let code = format!(
                r#"
import numpy as np
from pyscf import gto, scf, mcscf, fci, ao2mo

def compute(atom_str, basis_str, spin, chg, ncore):
    mol = gto.M(atom=atom_str, basis=basis_str, spin=spin, charge=chg, symmetry=False, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()

    if ncore > 0:
        ncas = mol.nao - ncore
        nelec = mol.nelectron - 2 * ncore
        mc = mcscf.CASCI(mf, ncas, nelec)
        try:
            mc.kernel()
        except Exception:
            pass
        h1_cas, core_energy = mc.get_h1eff()
        h2_packed = mc.get_h2eff()
        h2_cas_chemist = ao2mo.restore(1, h2_packed, ncas)
        h2_cas_physicist = np.transpose(h2_cas_chemist, (0, 2, 1, 3))
    else:
        ncas = mol.nao
        nelec = mol.nelectron
        core_energy = mol.energy_nuc()
        mo_coeff = mf.mo_coeff
        h1_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        h1_cas = mo_coeff.T @ h1_ao @ mo_coeff
        eri_ao = mol.intor('int2e')
        eri_mo = ao2mo.incore.full(eri_ao, mo_coeff)
        eri_mo_4d_chemist = ao2mo.restore(1, eri_mo, mol.nao)
        h2_cas_physicist = np.transpose(eri_mo_4d_chemist, (0, 2, 1, 3))

    return {{
        "nao": int(ncas),
        "nelec": int(nelec),
        "enuc": float(core_energy),
        "h1": h1_cas.tolist(),
        "eri": h2_cas_physicist.tolist(),
    }}
"#
            );

            let compute_module = PyModule::from_code(py, &code, "compute.py", "compute")
                .context("Failed to evaluate PySCF python code")?;
            let compute_func = compute_module
                .getattr("compute")
                .context("Failed to extract compute function")?;

            let py_atom_str = PyString::new(py, &config.geometry);
            let py_basis_str = PyString::new(py, &config.basis);

            let result_dict: &PyDict = compute_func
                .call1((
                    py_atom_str,
                    py_basis_str,
                    config.spin,
                    config.charge,
                    config.freeze_core,
                ))
                .context("PySCF computation failed")?
                .downcast()
                .map_err(|e| anyhow::anyhow!("Failed to downcast python result: {:?}", e))?;

            let n_orb: usize = result_dict
                .get_item("nao")?
                .context("Missing nao")?
                .extract()?;
            let n_elec: usize = result_dict
                .get_item("nelec")?
                .context("Missing nelec")?
                .extract()?;
            let nuclear_repulsion: f64 = result_dict
                .get_item("enuc")?
                .context("Missing enuc")?
                .extract()?;

            let h1_raw: Vec<Vec<f64>> = result_dict
                .get_item("h1")?
                .context("Missing h1")?
                .extract()?;
            let eri_raw: Vec<Vec<Vec<Vec<f64>>>> = result_dict
                .get_item("eri")?
                .context("Missing eri")?
                .extract()?;

            let mut h1 = Array2::zeros((n_orb, n_orb));
            for i in 0..n_orb {
                for j in 0..n_orb {
                    h1[[i, j]] = h1_raw[i][j];
                }
            }

            let mut h2 = Array4::zeros((n_orb, n_orb, n_orb, n_orb));
            for i in 0..n_orb {
                for j in 0..n_orb {
                    for k in 0..n_orb {
                        for l in 0..n_orb {
                            h2[[i, j, k, l]] = eri_raw[i][j][k][l];
                        }
                    }
                }
            }

            Ok(MolecularContext {
                n_orb,
                n_elec,
                nuclear_repulsion,
                h1,
                h2,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_h2() {
        pyo3::prepare_freethreaded_python();
        let config = ChemistryConfig {
            geometry: "H 0 0 0; H 0 0 0.735".to_string(),
            basis: "sto-3g".to_string(),
            charge: 0,
            spin: 0,
            freeze_core: 0,
        };
        let ctx = PyScfDriver::compute(&config).unwrap();
        assert_eq!(ctx.n_elec, 2);
        assert_eq!(ctx.n_orb, 2);
    }
}
