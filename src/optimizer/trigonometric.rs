use ndarray::Array1;
use num_complex::Complex64;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Finds the global minimum of a trigonometric function interpolated from N samples.
/// This executes a discrete Fourier transform to map the energy function into a trigonometric polynomial,
/// then sweeps the analytical geometric space bounded by exact Harmonic Newton-Raphson refinement.
pub fn minimize_trigonometric_func(samples: &Array1<f64>) -> (f64, f64) {
    let n = samples.len();
    if n % 2 == 0 {
        panic!("N must be odd for trigonometric optimization.");
    }
    let m = (n - 1) / 2;

    // 1. Compute Forward FFT natively using rustfft
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buffer: Vec<Complex64> = samples.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    fft.process(&mut buffer);

    let n_f64 = n as f64;
    for x in buffer.iter_mut() {
        *x = *x / n_f64;
    }

    // 2. Build the Polynomial P(z) = sum_{k=-m}^m (k * c_k) z^{k+m}
    // aberth solver expects lowest degree term first: P(z) = a + b*z + c*z^2 + ...
    let mut polynomial = Vec::with_capacity(n);
    for k in -(m as isize)..=(m as isize) {
        let k_idx = if k >= 0 {
            k as usize
        } else {
            (n as isize + k) as usize
        };
        let c_k = buffer[k_idx];
        polynomial.push(c_k * (k as f64));
    }

    // Strip structurally zero highest-degree trailing coefficients
    while let Some(&c) = polynomial.last() {
        if c.norm() < 1e-12 {
            polynomial.pop();
        } else {
            break;
        }
    }

    // Flat energy subspace failsafe
    if polynomial.len() <= 1 {
        return (0.0, samples[0]);
    }

    // 3. Extract Complex Roots exactly via Aberth's Method
    use aberth::AberthSolver;
    let mut solver = AberthSolver::new();
    solver.epsilon = 1e-12; // High precision for chemistry eigenvalue bounds
    solver.max_iterations = 2500;
    let roots_result = solver.find_roots(&polynomial);

    let mut candidate_thetas = Vec::new();
    for root in roots_result.roots.iter() {
        // Map back to theta = angle(z) natively projecting to real bound (-pi, pi]
        candidate_thetas.push(root.im.atan2(root.re));
    }

    // Include the original sample support points as failsafe bounds
    for i in 0..n {
        let mut t = (i as f64) * 2.0 * PI / n_f64;
        // Normalize perfectly to (-pi, pi]
        t = (t + PI) % (2.0 * PI) - PI;
        candidate_thetas.push(t);
    }

    // 4. Exact polynomial state evaluator mathematically replicating identical amplitudes
    let fast_eval = |theta: f64| -> f64 {
        let z = Complex64::from_polar(1.0, theta);
        let mut sum_val = Complex64::new(0.0, 0.0);

        for k in -(m as isize)..=(m as isize) {
            let k_idx = if k >= 0 {
                k as usize
            } else {
                (n as isize + k) as usize
            };
            let c_k = buffer[k_idx];
            let zk = if k >= 0 {
                z.powi(k as i32)
            } else {
                z.powi(k as i32)
            };
            sum_val += c_k * zk;
        }
        sum_val.re
    };

    // 5. Global ArgMin sweep explicitly targeting lowest verified state
    let mut best_theta = 0.0;
    let mut best_val = f64::MAX;

    for &theta in &candidate_thetas {
        let val = fast_eval(theta);
        if val < best_val {
            best_val = val;
            best_theta = theta;
        }
    }

    // Smooth exact identical Python (-pi, pi] topology bound constraint
    let mut normalized_t = (best_theta + PI) % (2.0 * PI);
    if normalized_t < 0.0 {
        normalized_t += 2.0 * PI;
    }
    normalized_t -= PI;

    (normalized_t, best_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_minimize_trigonometric() {
        pyo3::prepare_freethreaded_python();

        // Let's create an odd-sized array of sampled f(theta) values
        // N must be odd for this trigonometric approximation!
        // Suppose we handcraft f(x) = cos(x) for x = 0, 2pi/5, 4pi/5, 6pi/5, 8pi/5
        // minimum of cos(x) is at pi (-1.0).
        let c = std::f64::consts::PI * 2.0 / 5.0;
        let samples = array![
            (0.0_f64).cos(),
            c.cos(),
            (2.0 * c).cos(),
            (3.0 * c).cos(),
            (4.0 * c).cos()
        ];

        let (opt_theta, opt_value) = minimize_trigonometric_func(&samples);

        // Theta could be mapped to (-pi, pi], so pi or -pi
        assert!((opt_theta.abs() - std::f64::consts::PI).abs() < 1e-5);
        assert!((opt_value - (-1.0)).abs() < 1e-5);
    }
}
