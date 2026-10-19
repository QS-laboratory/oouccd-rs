use ndarray::{Array, Array2, Array4, Axis, RemoveAxis};

/// exp(sum_a kappa_a G_a), in the same sign convention as givens_rotate_*.
pub fn exponential_orbital_rotation(
    n_orb: usize,
    pairs: &[(usize, usize)],
    kappa: &[f64],
) -> Array2<f64> {
    assert_eq!(pairs.len(), kappa.len());
    let mut generator = nalgebra::DMatrix::<f64>::zeros(n_orb, n_orb);
    for (&(i, j), &angle) in pairs.iter().zip(kappa) {
        generator[(i, j)] += angle;
        generator[(j, i)] -= angle;
    }
    let rotation = generator.exp();
    Array2::from_shape_fn((n_orb, n_orb), |(i, j)| rotation[(i, j)])
}

/// Apply R to every index; never build an n^(2*rank) transformation tensor.
pub fn rotate_tensor<D: RemoveAxis>(h: &Array<f64, D>, rotation: &Array2<f64>) -> Array<f64, D> {
    let n = rotation.nrows();
    assert_eq!(rotation.ncols(), n);
    assert!(h.shape().iter().all(|&length| length == n));
    let mut result = h.clone();
    for axis in 0..h.ndim() {
        let mut next = Array::zeros(h.raw_dim());
        for i in 0..n {
            for j in 0..n {
                next.index_axis_mut(Axis(axis), i)
                    .scaled_add(rotation[[i, j]], &result.index_axis(Axis(axis), j));
            }
        }
        result = next;
    }
    result
}

/// Apply a single Givens rotation to the specified axis of a 2D tensor.
/// The rotation mixes rows/columns `idx_i` and `idx_j` by angle `theta`.
pub fn givens_rotate_2d(
    tensor: &mut Array2<f64>,
    idx_i: usize,
    idx_j: usize,
    axis: usize,
    theta: f64,
) {
    let c = theta.cos();
    let s = theta.sin();

    // Iterate over the orthogonal axis
    let orth_axis = 1 - axis;
    let length = tensor.shape()[orth_axis];

    for k in 0..length {
        let (val_i, val_j) = if axis == 0 {
            (tensor[[idx_i, k]], tensor[[idx_j, k]])
        } else {
            (tensor[[k, idx_i]], tensor[[k, idx_j]])
        };

        // Convention aligned exactly with the python implementation
        let new_i = c * val_i + s * val_j;
        let new_j = -s * val_i + c * val_j;

        if axis == 0 {
            tensor[[idx_i, k]] = new_i;
            tensor[[idx_j, k]] = new_j;
        } else {
            tensor[[k, idx_i]] = new_i;
            tensor[[k, idx_j]] = new_j;
        };
    }
}

/// Apply orbital rotation U = exp(kappa) to a 2-index tensor h_pq.
pub fn rotate_1body_integrals(h: &mut Array2<f64>, idx_i: usize, idx_j: usize, theta: f64) {
    for axis in 0..2 {
        givens_rotate_2d(h, idx_i, idx_j, axis, theta);
    }
}

/// Apply a single Givens rotation to the specified axis of a 4D tensor.
pub fn givens_rotate_4d(
    tensor: &mut Array4<f64>,
    idx_i: usize,
    idx_j: usize,
    axis: usize,
    theta: f64,
) {
    let c = theta.cos();
    let s = theta.sin();

    let shape = tensor.shape().to_vec();
    let mut indices = vec![0, 0, 0, 0];

    // Build loops dynamically based on shape sizes, skipping the selected axis
    let mut iter_axes = vec![0, 1, 2, 3];
    iter_axes.retain(|&x| x != axis);

    for d0 in 0..shape[iter_axes[0]] {
        indices[iter_axes[0]] = d0;
        for d1 in 0..shape[iter_axes[1]] {
            indices[iter_axes[1]] = d1;
            for d2 in 0..shape[iter_axes[2]] {
                indices[iter_axes[2]] = d2;

                indices[axis] = idx_i;
                let val_i = tensor[[indices[0], indices[1], indices[2], indices[3]]];

                indices[axis] = idx_j;
                let val_j = tensor[[indices[0], indices[1], indices[2], indices[3]]];

                let new_i = c * val_i + s * val_j;
                let new_j = -s * val_i + c * val_j;

                indices[axis] = idx_i;
                tensor[[indices[0], indices[1], indices[2], indices[3]]] = new_i;

                indices[axis] = idx_j;
                tensor[[indices[0], indices[1], indices[2], indices[3]]] = new_j;
            }
        }
    }
}

/// Apply orbital rotation U = exp(kappa) to a 4-index tensor h_pqrs.
pub fn rotate_2body_integrals(h: &mut Array4<f64>, idx_i: usize, idx_j: usize, theta: f64) {
    for axis in 0..4 {
        givens_rotate_4d(h, idx_i, idx_j, axis, theta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn exponential_rotation_matches_single_givens_and_is_orthogonal() {
        let pairs = [(0, 2), (1, 2)];
        let h1 = Array2::from_shape_fn((3, 3), |(i, j)| (i * 3 + j) as f64);
        let h2 = Array4::from_shape_fn((3, 3, 3, 3), |(i, j, k, l)| {
            (i + 2 * j + 3 * k + 4 * l) as f64
        });
        let rotation = exponential_orbital_rotation(3, &pairs[..1], &[0.37]);
        let mut reference1 = h1.clone();
        let mut reference2 = h2.clone();
        rotate_1body_integrals(&mut reference1, 0, 2, 0.37);
        rotate_2body_integrals(&mut reference2, 0, 2, 0.37);
        assert!(rotate_tensor(&h1, &rotation)
            .iter()
            .zip(reference1.iter())
            .all(|(a, b)| (a - b).abs() < 1e-12));
        assert!(rotate_tensor(&h2, &rotation)
            .iter()
            .zip(reference2.iter())
            .all(|(a, b)| (a - b).abs() < 1e-12));
        let multiple = exponential_orbital_rotation(3, &pairs, &[0.37, -0.23]);
        let identity = multiple.dot(&multiple.t());
        for ((i, j), v) in identity.indexed_iter() {
            assert!((v - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
        }
    }

    #[test]
    fn test_givens_rotate_2d() {
        // Simple 2x2 identity matrix rotated by 90 degrees (pi/2)
        let mut mat = array![[1.0, 0.0], [0.0, 1.0]];
        let theta = std::f64::consts::PI / 2.0;

        // Apply a single Givens rotation to mix rows 0 and 1 along axis 0
        givens_rotate_2d(&mut mat, 0, 1, 0, theta);

        // Expected manually verified Python outcome:
        // [[0, 1], [-1, 0]]
        assert!((mat[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((mat[[1, 0]] - (-1.0)).abs() < 1e-10);
    }
}
