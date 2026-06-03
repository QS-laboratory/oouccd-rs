use ndarray::{Array2, Array4};
use num_complex::Complex64;

fn parity(state: usize, k: usize) -> i32 {
    let mask = (1 << k) - 1;
    let ones = (state & mask).count_ones();
    if ones % 2 == 1 {
        -1
    } else {
        1
    }
}

fn apply_annihilate(state: usize, k: usize) -> Option<(usize, i32)> {
    if (state & (1 << k)) == 0 {
        return None;
    }
    let p = parity(state, k);
    Some((state & !(1 << k), p))
}

fn apply_create(state: usize, k: usize) -> Option<(usize, i32)> {
    if (state & (1 << k)) != 0 {
        return None;
    }
    let p = parity(state, k);
    Some((state | (1 << k), p))
}

// Computes <psi | a_p^ a_q | psi>
pub fn expect_1body(state_vector: &[Complex64], p_dag: usize, q_ann: usize) -> f64 {
    let mut exp_val = 0.0;
    for i in 0..state_vector.len() {
        if let Some((i1, p1)) = apply_annihilate(i, q_ann) {
            if let Some((i2, p2)) = apply_create(i1, p_dag) {
                let term = (state_vector[i2].conj() * state_vector[i]).re * (p1 * p2) as f64;
                exp_val += term;
            }
        }
    }
    exp_val
}

// Computes <psi | a_{p_dag}^ a_{q_dag}^ a_{r_ann} a_{s_ann} | psi>
pub fn expect_2body(
    state_vector: &[Complex64],
    p_dag: usize,
    q_dag: usize,
    r_ann: usize,
    s_ann: usize,
) -> f64 {
    let mut exp_val = 0.0;
    for i in 0..state_vector.len() {
        if let Some((i1, p1)) = apply_annihilate(i, s_ann) {
            // acts 1st
            if let Some((i2, p2)) = apply_annihilate(i1, r_ann) {
                // acts 2nd
                if let Some((i3, p3)) = apply_create(i2, q_dag) {
                    // acts 3rd
                    if let Some((i4, p4)) = apply_create(i3, p_dag) {
                        // acts 4th
                        let term = (state_vector[i4].conj() * state_vector[i]).re
                            * (p1 * p2 * p3 * p4) as f64;
                        exp_val += term;
                    }
                }
            }
        }
    }
    exp_val
}

pub fn compute_1rdm_spatial(state: &[Complex64], n_spatial: usize) -> Array2<f64> {
    let mut gamma = Array2::<f64>::zeros((n_spatial, n_spatial));
    use rayon::prelude::*;

    let mut elements = Vec::new();
    for p in 0..n_spatial {
        for q in 0..n_spatial {
            elements.push((p, q));
        }
    }

    let results: Vec<((usize, usize), f64)> = elements
        .into_par_iter()
        .map(|(p, q)| {
            let mut val = 0.0;
            for s in 0..2 {
                val += expect_1body(state, 2 * p + s, 2 * q + s);
            }
            ((p, q), val)
        })
        .collect();

    for ((p, q), val) in results {
        gamma[[p, q]] = val;
    }
    gamma
}

pub fn compute_2rdm_spatial(state: &[Complex64], n_spatial: usize) -> Array4<f64> {
    let mut gamma2 = Array4::<f64>::zeros((n_spatial, n_spatial, n_spatial, n_spatial));
    use rayon::prelude::*;

    let mut elements = Vec::new();
    for p in 0..n_spatial {
        for q in 0..n_spatial {
            for r in 0..n_spatial {
                for s in 0..n_spatial {
                    elements.push((p, q, r, s));
                }
            }
        }
    }

    let results: Vec<((usize, usize, usize, usize), f64)> = elements
        .into_par_iter()
        .map(|(p, q, r, s)| {
            let mut val = 0.0;
            for s1 in 0..2 {
                for s2 in 0..2 {
                    let p_dag = 2 * p + s1;
                    let q_dag = 2 * q + s2;
                    let r_ann = 2 * s + s2;
                    let s_ann = 2 * r + s1;
                    val += 0.5 * expect_2body(state, p_dag, q_dag, r_ann, s_ann);
                }
            }
            ((p, q, r, s), val)
        })
        .collect();

    for ((p, q, r, s), val) in results {
        gamma2[[p, q, r, s]] = val;
    }
    gamma2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parity() {
        assert_eq!(parity(0b0011, 2), 1);
        assert_eq!(parity(0b0111, 2), 1);
    }

    #[test]
    fn test_apply_annihilate_create() {
        assert_eq!(apply_annihilate(0b1010, 1), Some((0b1000, 1))); // bit 1 drops, bit 0 is 0 (parity 1)
        assert_eq!(apply_create(0b1000, 1), Some((0b1010, 1)));
    }
}
