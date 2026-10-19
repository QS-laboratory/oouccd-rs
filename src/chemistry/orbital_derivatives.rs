//! Derivatives of E(exp(K)) at K=0, contracted with a fixed real spin-summed RDM.
use crate::chemistry::engine::ClassicalEngine;
use ndarray::{Array, Array1, Array2, Array4, Axis, RemoveAxis};

/// G_ij has G[i,j]=+1, G[j,i]=-1, matching the existing Givens convention.
fn generator_action<D: RemoveAxis>(h: &Array<f64, D>, i: usize, j: usize) -> Array<f64, D> {
    let mut result = Array::zeros(h.raw_dim());
    for axis in 0..h.ndim() {
        result
            .index_axis_mut(Axis(axis), i)
            .scaled_add(1.0, &h.index_axis(Axis(axis), j));
        result
            .index_axis_mut(Axis(axis), j)
            .scaled_add(-1.0, &h.index_axis(Axis(axis), i));
    }
    result
}

pub fn orbital_gradient_hessian(
    engine: &ClassicalEngine,
    rdm1: &Array2<f64>,
    rdm2: &Array4<f64>,
    pairs: &[(usize, usize)],
) -> (Array1<f64>, Array2<f64>) {
    let first: Vec<_> = pairs
        .iter()
        .map(|&(i, j)| {
            (
                generator_action(&engine.h1, i, j),
                generator_action(&engine.h2, i, j),
            )
        })
        .collect();
    // The project's 2-RDM already includes the 1/2 factor. Do not add another.
    let contract = |h1: &Array2<f64>, h2: &Array4<f64>| (h1 * rdm1).sum() + (h2 * rdm2).sum();
    let gradient = Array1::from_iter(first.iter().map(|(h1, h2)| contract(h1, h2)));
    let mut hessian = Array2::zeros((pairs.len(), pairs.len()));
    for (a, &(i, j)) in pairs.iter().enumerate() {
        for (b, &(k, l)) in pairs.iter().enumerate().take(a + 1) {
            // Symmetrize the two orders: mixed derivatives of an exponential,
            // rather than those of a product of independently ordered Givens rotations.
            let ab = contract(
                &generator_action(&first[b].0, i, j),
                &generator_action(&first[b].1, i, j),
            );
            let ba = contract(
                &generator_action(&first[a].0, k, l),
                &generator_action(&first[a].1, k, l),
            );
            hessian[[a, b]] = 0.5 * (ab + ba);
            hessian[[b, a]] = hessian[[a, b]];
        }
    }
    (gradient, hessian)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chemistry::rdm::{compute_1rdm_spatial, compute_2rdm_spatial};
    use crate::optimizer::orbital::OrbitalProblem;
    use crate::optimizer::utils::exponential_orbital_rotation;
    use num_complex::Complex64;

    #[test]
    fn exponential_derivatives_match_finite_differences_including_noncommuting_pairs() {
        let n = 3;
        let engine = ClassicalEngine {
            h1: Array2::from_shape_fn((n, n), |(p, q)| ((p + q + 1) as f64).cos()),
            h2: Array4::from_shape_fn((n, n, n, n), |(p, q, r, s)| {
                ((p + r + 1) as f64).sin() * ((q + s + 1) as f64).sin() * 0.3
            }),
            nuclear_repulsion: 0.7,
        };
        let mut state: Vec<_> = (0usize..1 << (2 * n))
            .map(|i| {
                let amplitude = if i.count_ones() == 2 {
                    ((i + 1) as f64).sin()
                } else {
                    0.0
                };
                Complex64::new(amplitude, 0.0)
            })
            .collect();
        let norm = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        state.iter_mut().for_each(|c| *c /= norm);
        let rdm1 = compute_1rdm_spatial(&state, n);
        let rdm2 = compute_2rdm_spatial(&state, n);
        let pairs = [(0, 1), (0, 2)];
        let problem = OrbitalProblem {
            engine: &engine,
            rdm1: &rdm1,
            rdm2: &rdm2,
            pairs: &pairs,
        };
        let (g, h) = orbital_gradient_hessian(&engine, &rdm1, &rdm2, &pairs);
        let evaluate = |kappa: &[f64]| {
            problem.energy_with_rotation(&exponential_orbital_rotation(n, &pairs, kappa))
        };
        let zero = vec![0.0; pairs.len()];
        let e0 = evaluate(&zero);
        let eps = 1e-4;
        assert!(g.iter().any(|v| v.abs() > 1e-3));
        for a in 0..pairs.len() {
            let mut plus = zero.clone();
            plus[a] = eps;
            let mut minus = zero.clone();
            minus[a] = -eps;
            let numerical_g = (evaluate(&plus) - evaluate(&minus)) / (2.0 * eps);
            let numerical_h = (evaluate(&plus) - 2.0 * e0 + evaluate(&minus)) / eps.powi(2);
            assert!(
                (g[a] - numerical_g).abs() < 1e-7,
                "gradient {a}: {} vs {numerical_g}",
                g[a]
            );
            assert!(
                (h[[a, a]] - numerical_h).abs() < 2e-6,
                "diagonal Hessian {a}"
            );
            for b in 0..a {
                let mut total = 0.0;
                for sa in [-1.0, 1.0] {
                    for sb in [-1.0, 1.0] {
                        let mut x = zero.clone();
                        x[a] = sa * eps;
                        x[b] = sb * eps;
                        total += sa * sb * evaluate(&x);
                    }
                }
                let mixed = total / (4.0 * eps.powi(2));
                assert!(
                    (h[[a, b]] - mixed).abs() < 2e-6,
                    "mixed Hessian: {} vs {mixed}",
                    h[[a, b]]
                );
                assert_eq!(h[[a, b]], h[[b, a]]);
            }
        }
    }
}
