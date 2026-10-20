
/// Numerical gradient via central finite differences.
/// Used for OO where evaluating the objective is cheap (no quantum circuit).
pub fn finite_diff_gradient(
    x: &[f64],
    objective: &mut dyn FnMut(&[f64]) -> f64,
    eps: f64,
) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0; n];
    for j in 0..n {
        let mut xp = x.to_vec();
        xp[j] += eps;
        let mut xm = x.to_vec();
        xm[j] -= eps;
        grad[j] = (objective(&xp) - objective(&xm)) / (2.0 * eps);
    }
    grad
}
