
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

/// L-BFGS two-loop recursion (Nocedal & Wright, Algorithm 7.4).
/// Returns the search direction d = -H_k * g_k.
/// s_history[i] = x_{k-m+i+1} - x_{k-m+i}, y_history[i] = g_{k-m+i+1} - g_{k-m+i}
/// (index 0 = oldest, last index = most recent)
pub fn lbfgs_direction(
    g: &[f64],
    s_history: &[Vec<f64>],
    y_history: &[Vec<f64>],
) -> Vec<f64> {
    let m = s_history.len();
    let n = g.len();
    let mut q = g.to_vec();
    let mut alphas = vec![0.0f64; m];
    let mut rhos = vec![0.0f64; m];

    // Forward pass: newest to oldest
    for i in (0..m).rev() {
        let sy: f64 = dot(&s_history[i], &y_history[i]);
        if sy.abs() < 1e-15 {
            continue;
        }
        rhos[i] = 1.0 / sy;
        let sq: f64 = dot(&s_history[i], &q);
        alphas[i] = rhos[i] * sq;
        for k in 0..n {
            q[k] -= alphas[i] * y_history[i][k];
        }
    }

    // Scale with H_0 = gamma * I (Byrd et al. scaling)
    let gamma = if m > 0 {
        let sy: f64 = dot(&s_history[m - 1], &y_history[m - 1]);
        let yy: f64 = dot(&y_history[m - 1], &y_history[m - 1]);
        if yy.abs() < 1e-15 { 1.0 } else { sy / yy }
    } else {
        1.0
    };
    let mut r: Vec<f64> = q.iter().map(|qi| gamma * qi).collect();

    // Backward pass: oldest to newest
    for i in 0..m {
        let yr: f64 = dot(&y_history[i], &r);
        let beta = rhos[i] * yr;
        for k in 0..n {
            r[k] += s_history[i][k] * (alphas[i] - beta);
        }
    }

    // Direction is -H_k * g_k
    r.iter().map(|ri| -ri).collect()
}

/// Armijo (backtracking) line search.
/// Returns (x_new, f_new) satisfying the sufficient decrease condition.
pub fn armijo_line_search(
    x: &[f64],
    d: &[f64],
    f0: f64,
    g0: &[f64],
    objective: &mut dyn FnMut(&[f64]) -> f64,
) -> (Vec<f64>, f64) {
    let dg0: f64 = dot(d, g0);
    let c1 = 1e-4;
    let mut alpha = 1.0_f64;

    for _ in 0..50 {
        let x_new: Vec<f64> = x.iter().zip(d.iter()).map(|(xi, di)| xi + alpha * di).collect();
        let f_new = objective(&x_new);
        if f_new <= f0 + c1 * alpha * dg0 {
            return (x_new, f_new);
        }
        alpha *= 0.5;
        if alpha < 1e-14 {
            break;
        }
    }

    // Line search failed: return a tiny step
    let x_new: Vec<f64> = x.iter().zip(d.iter()).map(|(xi, di)| xi + 1e-8 * di).collect();
    let f_new = objective(&x_new);
    (x_new, f_new)
}

pub enum GradientType {
    FiniteDiff(f64),
}

/// Runs L-BFGS optimization.
///
/// - `objective`: the function to minimize (called for value and gradient)
/// - `gradient_type`: how to compute gradients
/// - Returns (x_optimal, f_optimal)
pub fn run_lbfgs(
    x0: Vec<f64>,
    max_iter: usize,
    objective: &mut dyn FnMut(&[f64]) -> f64,
    gradient_type: GradientType,
) -> (Vec<f64>, f64) {
    let memory = 10;
    let grad_tol = 1e-6;

    let mut x = x0;
    let mut f = objective(&x);
    let mut g = compute_gradient(&x, objective, &gradient_type);

    let mut s_history: Vec<Vec<f64>> = Vec::new();
    let mut y_history: Vec<Vec<f64>> = Vec::new();

    for _ in 0..max_iter {
        let gnorm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
        if gnorm < grad_tol {
            break;
        }

        let d = lbfgs_direction(&g, &s_history, &y_history);

        // Verify descent direction
        let dg: f64 = dot(&d, &g);
        if dg >= 0.0 {
            // Not a descent direction (numerical issue); reset history and use steepest descent
            s_history.clear();
            y_history.clear();
            let d_sd: Vec<f64> = g.iter().map(|gi| -gi).collect();
            let (x_new, f_new) = armijo_line_search(&x, &d_sd, f, &g, objective);
            let g_new = compute_gradient(&x_new, objective, &gradient_type);
            x = x_new;
            f = f_new;
            g = g_new;
            continue;
        }

        let (x_new, f_new) = armijo_line_search(&x, &d, f, &g, objective);
        let g_new = compute_gradient(&x_new, objective, &gradient_type);

        let s: Vec<f64> = x_new.iter().zip(x.iter()).map(|(xn, xo)| xn - xo).collect();
        let y: Vec<f64> = g_new.iter().zip(g.iter()).map(|(gn, go)| gn - go).collect();

        // Only update history if curvature condition holds (s^T y > 0)
        let sy: f64 = dot(&s, &y);
        if sy > 1e-10 {
            s_history.push(s);
            y_history.push(y);
            if s_history.len() > memory {
                s_history.remove(0);
                y_history.remove(0);
            }
        }

        x = x_new;
        f = f_new;
        g = g_new;
    }

    (x, f)
}

fn compute_gradient(
    x: &[f64],
    objective: &mut dyn FnMut(&[f64]) -> f64,
    gradient_type: &GradientType,
) -> Vec<f64> {
    match gradient_type {
        GradientType::FiniteDiff(eps) => finite_diff_gradient(x, objective, *eps),
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}
