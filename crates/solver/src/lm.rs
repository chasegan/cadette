//! A small dense Levenberg-Marquardt least-squares solver.
//!
//! Given variables `x ∈ ℝⁿ` and residual functions `rᵢ(x)`, it drives the
//! residuals toward zero by minimizing `Σ rᵢ(x)²`. The Jacobian is approximated
//! by finite differences, so callers only supply residual closures — adding a
//! new constraint never requires hand-derived derivatives.
//!
//! LM interpolates between Gauss-Newton (fast near the solution) and gradient
//! descent (robust far from it) via a damping term `λ`, and the damped diagonal
//! keeps rank-deficient (under-constrained) systems well-behaved: free
//! variables simply stay near their initial guess instead of blowing up.

/// A residual: a function of the variable vector returning a value to drive to
/// zero.
pub type Residual = Box<dyn Fn(&[f64]) -> f64>;

/// Solver tuning.
#[derive(Clone, Copy, Debug)]
pub struct SolveOptions {
    pub max_iterations: usize,
    /// Convergence threshold on the largest absolute residual.
    pub tolerance: f64,
    pub initial_lambda: f64,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-9,
            initial_lambda: 1e-3,
        }
    }
}

/// Outcome of a solve.
#[derive(Clone, Copy, Debug)]
pub struct SolveReport {
    pub converged: bool,
    pub iterations: usize,
    /// Largest absolute residual at the returned solution.
    pub residual_norm: f64,
}

/// Solve in place: mutate `vars` toward values minimizing the residuals.
pub fn solve(vars: &mut [f64], residuals: &[Residual], opts: &SolveOptions) -> SolveReport {
    let n = vars.len();
    let m = residuals.len();

    let eval = |x: &[f64]| -> Vec<f64> { residuals.iter().map(|r| r(x)).collect() };
    let max_abs = |r: &[f64]| r.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let sum_sq = |r: &[f64]| r.iter().map(|v| v * v).sum::<f64>();

    if n == 0 || m == 0 {
        let r = eval(vars);
        return SolveReport {
            converged: max_abs(&r) <= opts.tolerance,
            iterations: 0,
            residual_norm: max_abs(&r),
        };
    }

    let mut x = vars.to_vec();
    let mut r = eval(&x);
    let mut cost = sum_sq(&r);
    let mut lambda = opts.initial_lambda;
    let mut iterations = 0;
    let mut converged = max_abs(&r) <= opts.tolerance;

    while !converged && iterations < opts.max_iterations {
        iterations += 1;

        // Forward-difference Jacobian J (m x n).
        let mut jac = vec![0.0; m * n];
        for j in 0..n {
            let step = 1e-7 * (1.0 + x[j].abs());
            let mut xp = x.clone();
            xp[j] += step;
            let rp = eval(&xp);
            for i in 0..m {
                jac[i * n + j] = (rp[i] - r[i]) / step;
            }
        }

        // Normal equations: A = JᵀJ (+ LM damping), g = Jᵀr.
        let mut a = vec![0.0; n * n];
        let mut g = vec![0.0; n];
        for col in 0..n {
            for k in 0..n {
                let mut s = 0.0;
                for i in 0..m {
                    s += jac[i * n + col] * jac[i * n + k];
                }
                a[col * n + k] = s;
            }
            let mut s = 0.0;
            for i in 0..m {
                s += jac[i * n + col] * r[i];
            }
            g[col] = s;
        }

        // Try damped steps, growing λ until one reduces the cost.
        let mut accepted = false;
        for _ in 0..12 {
            let mut a_damped = a.clone();
            for k in 0..n {
                // Marquardt scaling + a floor so unconstrained variables (zero
                // diagonal) get a small regularization and stay put.
                a_damped[k * n + k] = a[k * n + k] * (1.0 + lambda) + lambda * 1e-9 + 1e-12;
            }
            let neg_g: Vec<f64> = g.iter().map(|v| -v).collect();

            let Some(dx) = solve_linear(&a_damped, &neg_g, n) else {
                lambda *= 4.0;
                continue;
            };

            let x_new: Vec<f64> = x.iter().zip(&dx).map(|(xi, d)| xi + d).collect();
            let r_new = eval(&x_new);
            let cost_new = sum_sq(&r_new);

            if cost_new < cost {
                x = x_new;
                r = r_new;
                cost = cost_new;
                lambda = (lambda * 0.5).max(1e-12);
                accepted = true;
                break;
            } else {
                lambda *= 4.0;
            }
        }

        converged = max_abs(&r) <= opts.tolerance;
        if !accepted {
            // No downhill step found — at a local minimum for this system.
            break;
        }
    }

    vars.copy_from_slice(&x);
    SolveReport {
        converged,
        iterations,
        residual_norm: max_abs(&r),
    }
}

/// Solve the dense linear system `A x = b` (row-major `A`, size `n`) by Gaussian
/// elimination with partial pivoting. Returns `None` if `A` is singular.
fn solve_linear(a_in: &[f64], b_in: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = a_in.to_vec();
    let mut b = b_in.to_vec();

    for col in 0..n {
        // Partial pivot: largest magnitude in this column at/below the diagonal.
        let mut pivot = col;
        let mut best = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1e-15 {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
            }
            b.swap(col, pivot);
        }

        let diag = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / diag;
            if factor != 0.0 {
                for k in col..n {
                    a[row * n + k] -= factor * a[col * n + k];
                }
                b[row] -= factor * b[col];
            }
        }
    }

    // Back-substitution.
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let mut sum = b[col];
        for k in (col + 1)..n {
            sum -= a[col * n + k] * x[k];
        }
        x[col] = sum / a[col * n + col];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn solves_decoupled_linear() {
        // r0 = x0 - 3, r1 = x1 - 4  ->  (3, 4)
        let residuals: Vec<Residual> = vec![
            Box::new(|x| x[0] - 3.0),
            Box::new(|x| x[1] - 4.0),
        ];
        let mut x = vec![0.0, 0.0];
        let report = solve(&mut x, &residuals, &SolveOptions::default());
        assert!(report.converged);
        assert!(close(x[0], 3.0) && close(x[1], 4.0));
    }

    #[test]
    fn solves_coupled_linear() {
        // x0 + x1 = 10, x0 - x1 = 2  ->  (6, 4)
        let residuals: Vec<Residual> = vec![
            Box::new(|x| x[0] + x[1] - 10.0),
            Box::new(|x| x[0] - x[1] - 2.0),
        ];
        let mut x = vec![0.0, 0.0];
        let report = solve(&mut x, &residuals, &SolveOptions::default());
        assert!(report.converged);
        assert!(close(x[0], 6.0) && close(x[1], 4.0));
    }

    #[test]
    fn solves_nonlinear_circle_intersection() {
        // On the circle of radius 5 with x0 = 3  ->  x1 = 4 (from a +y guess).
        let residuals: Vec<Residual> = vec![
            Box::new(|x| (x[0] * x[0] + x[1] * x[1]).sqrt() - 5.0),
            Box::new(|x| x[0] - 3.0),
        ];
        let mut x = vec![3.0, 3.0];
        let report = solve(&mut x, &residuals, &SolveOptions::default());
        assert!(report.converged, "norm {}", report.residual_norm);
        assert!(close(x[0], 3.0) && close(x[1], 4.0));
    }

    #[test]
    fn underconstrained_stays_near_guess() {
        // Only x0 is constrained; x1 is free and should not blow up.
        let residuals: Vec<Residual> = vec![Box::new(|x| x[0] - 1.0)];
        let mut x = vec![5.0, 7.0];
        let report = solve(&mut x, &residuals, &SolveOptions::default());
        assert!(report.converged);
        assert!(close(x[0], 1.0));
        assert!((x[1] - 7.0).abs() < 1e-3, "free var drifted to {}", x[1]);
    }
}
