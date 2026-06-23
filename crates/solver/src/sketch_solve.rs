//! Compile a [`Sketch2d`] into a residual system and solve it.
//!
//! Each constraint becomes one or more residual closures over a flat variable
//! vector: every point contributes `(x, y)` and every circle a `radius`. The
//! solved values are written back into the sketch. A degrees-of-freedom count
//! (variables minus Jacobian rank) classifies the sketch as under-, well-, or
//! over-constrained.

use rmf_core::{CircleId, Constraint, LineId, PointId, Sketch2d};

use crate::lm::{solve, Residual, SolveOptions, SolveReport};

/// The result of solving a sketch.
#[derive(Clone, Copy, Debug)]
pub struct SketchSolution {
    pub report: SolveReport,
    /// Number of solver variables (`2·points + circles`).
    pub variables: usize,
    /// Number of residual equations.
    pub equations: usize,
    /// Variables minus independent constraints. `0` = well-constrained,
    /// `> 0` = under-constrained (free to move).
    pub degrees_of_freedom: usize,
}

impl SketchSolution {
    pub fn is_fully_constrained(&self) -> bool {
        self.degrees_of_freedom == 0
    }
}

/// Solve `sketch` in place, updating its point and circle values to satisfy the
/// constraints as closely as possible.
pub fn solve_sketch(sketch: &mut Sketch2d) -> SketchSolution {
    let n_points = sketch.points.len();
    let n_circles = sketch.circles.len();
    let n_vars = 2 * n_points + n_circles;

    // Variable indices.
    let px = |p: PointId| 2 * p.0;
    let py = |p: PointId| 2 * p.0 + 1;
    let cr = |c: CircleId| 2 * n_points + c.0;

    // Pack current state into the variable vector.
    let mut vars = vec![0.0; n_vars];
    for (i, p) in sketch.points.iter().enumerate() {
        vars[2 * i] = p.x;
        vars[2 * i + 1] = p.y;
    }
    for (j, c) in sketch.circles.iter().enumerate() {
        vars[2 * n_points + j] = c.radius;
    }

    let residuals = build_residuals(sketch, &px, &py, &cr);

    let report = solve(&mut vars, &residuals, &SolveOptions::default());

    // Write solved values back.
    for (i, p) in sketch.points.iter_mut().enumerate() {
        p.x = vars[2 * i];
        p.y = vars[2 * i + 1];
    }
    for (j, c) in sketch.circles.iter_mut().enumerate() {
        c.radius = vars[2 * n_points + j];
    }

    let rank = jacobian_rank(&vars, &residuals);
    SketchSolution {
        report,
        variables: n_vars,
        equations: residuals.len(),
        degrees_of_freedom: n_vars.saturating_sub(rank),
    }
}

fn build_residuals(
    sketch: &Sketch2d,
    px: &impl Fn(PointId) -> usize,
    py: &impl Fn(PointId) -> usize,
    cr: &impl Fn(CircleId) -> usize,
) -> Vec<Residual> {
    let line_pts = |l: LineId| {
        let line = sketch.line(l);
        (line.a, line.b)
    };
    let mut residuals: Vec<Residual> = Vec::new();

    for &constraint in &sketch.constraints {
        match constraint {
            Constraint::Fixed(p) => {
                let (ix, iy) = (px(p), py(p));
                let point = sketch.point(p);
                let (ax, ay) = (point.x, point.y);
                residuals.push(Box::new(move |x| x[ix] - ax));
                residuals.push(Box::new(move |x| x[iy] - ay));
            }
            Constraint::Coincident(a, b) => {
                let (ax, ay, bx, by) = (px(a), py(a), px(b), py(b));
                residuals.push(Box::new(move |x| x[ax] - x[bx]));
                residuals.push(Box::new(move |x| x[ay] - x[by]));
            }
            Constraint::Horizontal(l) => {
                let (a, b) = line_pts(l);
                let (ay, by) = (py(a), py(b));
                residuals.push(Box::new(move |x| x[ay] - x[by]));
            }
            Constraint::Vertical(l) => {
                let (a, b) = line_pts(l);
                let (ax, bx) = (px(a), px(b));
                residuals.push(Box::new(move |x| x[ax] - x[bx]));
            }
            Constraint::Distance(a, b, d) => {
                let (ax, ay, bx, by) = (px(a), py(a), px(b), py(b));
                residuals.push(Box::new(move |x| {
                    (x[ax] - x[bx]).hypot(x[ay] - x[by]) - d
                }));
            }
            Constraint::Parallel(l1, l2) => {
                let (a, b) = line_pts(l1);
                let (c, e) = line_pts(l2);
                let (ax, ay, bx, by) = (px(a), py(a), px(b), py(b));
                let (cx, cy, dx, dy) = (px(c), py(c), px(e), py(e));
                residuals.push(Box::new(move |x| {
                    let (u1, u2) = (x[bx] - x[ax], x[by] - x[ay]);
                    let (v1, v2) = (x[dx] - x[cx], x[dy] - x[cy]);
                    u1 * v2 - u2 * v1 // cross product = 0 when parallel
                }));
            }
            Constraint::Perpendicular(l1, l2) => {
                let (a, b) = line_pts(l1);
                let (c, e) = line_pts(l2);
                let (ax, ay, bx, by) = (px(a), py(a), px(b), py(b));
                let (cx, cy, dx, dy) = (px(c), py(c), px(e), py(e));
                residuals.push(Box::new(move |x| {
                    let (u1, u2) = (x[bx] - x[ax], x[by] - x[ay]);
                    let (v1, v2) = (x[dx] - x[cx], x[dy] - x[cy]);
                    u1 * v1 + u2 * v2 // dot product = 0 when perpendicular
                }));
            }
            Constraint::EqualLength(l1, l2) => {
                let (a, b) = line_pts(l1);
                let (c, e) = line_pts(l2);
                let (ax, ay, bx, by) = (px(a), py(a), px(b), py(b));
                let (cx, cy, dx, dy) = (px(c), py(c), px(e), py(e));
                residuals.push(Box::new(move |x| {
                    let len1 = (x[bx] - x[ax]).hypot(x[by] - x[ay]);
                    let len2 = (x[dx] - x[cx]).hypot(x[dy] - x[cy]);
                    len1 - len2
                }));
            }
            Constraint::Radius(c, r) => {
                let ir = cr(c);
                residuals.push(Box::new(move |x| x[ir] - r));
            }
        }
    }

    residuals
}

/// Numerical rank of the residual Jacobian at `vars` — used to count degrees of
/// freedom. Row-reduces the `m × n` Jacobian and counts pivots above a
/// tolerance.
fn jacobian_rank(vars: &[f64], residuals: &[Residual]) -> usize {
    let n = vars.len();
    let m = residuals.len();
    if n == 0 || m == 0 {
        return 0;
    }

    let r0: Vec<f64> = residuals.iter().map(|r| r(vars)).collect();
    let mut jac = vec![0.0; m * n];
    for j in 0..n {
        let step = 1e-7 * (1.0 + vars[j].abs());
        let mut xp = vars.to_vec();
        xp[j] += step;
        for (i, res) in residuals.iter().enumerate() {
            jac[i * n + j] = (res(&xp) - r0[i]) / step;
        }
    }

    // Gaussian elimination over rows, counting independent pivots.
    let tol = 1e-7;
    let mut rank = 0;
    let mut pivot_col = 0;
    let mut row = 0;
    while row < m && pivot_col < n {
        // Find the row at/below `row` with the largest entry in `pivot_col`.
        let mut best_row = row;
        let mut best = jac[row * n + pivot_col].abs();
        for r in (row + 1)..m {
            let v = jac[r * n + pivot_col].abs();
            if v > best {
                best = v;
                best_row = r;
            }
        }
        if best < tol {
            pivot_col += 1;
            continue;
        }
        for k in 0..n {
            jac.swap(row * n + k, best_row * n + k);
        }
        let pivot = jac[row * n + pivot_col];
        for r in (row + 1)..m {
            let factor = jac[r * n + pivot_col] / pivot;
            for k in pivot_col..n {
                jac[r * n + k] -= factor * jac[row * n + k];
            }
        }
        rank += 1;
        row += 1;
        pivot_col += 1;
    }
    rank
}
