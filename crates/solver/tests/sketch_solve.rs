//! End-to-end constraint-solving tests: build a sketch, solve it, and assert
//! the geometry lands where the constraints demand.

use cdt_core::{CircleId, Constraint, LineId, PointId, Sketch2d, SketchSolver, SolveStatus};
use cdt_solver::{solve_sketch, LmSolver};

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-5
}

#[test]
fn fully_constrained_rectangle_solves_to_exact_corners() {
    // Four corners, four edges, anchored at the origin, sized 40 x 20.
    let mut s = Sketch2d::new();
    let p0 = s.add_point(0.0, 0.0);
    let p1 = s.add_point(38.0, 2.0); // deliberately off — the solver snaps it
    let p2 = s.add_point(41.0, 19.0);
    let p3 = s.add_point(-1.0, 21.0);

    let bottom = s.add_line(p0, p1);
    let right = s.add_line(p1, p2);
    let top = s.add_line(p2, p3);
    let left = s.add_line(p3, p0);

    s.add_constraint(Constraint::Fixed(p0));
    s.add_constraint(Constraint::Horizontal(bottom));
    s.add_constraint(Constraint::Vertical(right));
    s.add_constraint(Constraint::Horizontal(top));
    s.add_constraint(Constraint::Vertical(left));
    s.add_constraint(Constraint::Distance(p0, p1, 40.0));
    s.add_constraint(Constraint::Distance(p1, p2, 20.0));

    let solution = solve_sketch(&mut s);
    assert!(solution.report.converged, "norm {}", solution.report.residual_norm);
    assert!(solution.is_fully_constrained(), "dof {}", solution.degrees_of_freedom);

    let at = |p, x, y| {
        let pt = s.point(p);
        assert!(close(pt.x, x) && close(pt.y, y), "got ({}, {})", pt.x, pt.y);
    };
    at(p0, 0.0, 0.0);
    at(p1, 40.0, 0.0);
    at(p2, 40.0, 20.0);
    at(p3, 0.0, 20.0);
}

#[test]
fn equal_and_perpendicular_make_a_square() {
    // Anchor + horizontal base + perpendicular side + equal lengths + one size.
    let mut s = Sketch2d::new();
    let p0 = s.add_point(0.0, 0.0);
    let p1 = s.add_point(9.0, 1.0);
    let p2 = s.add_point(8.0, 11.0);

    let base = s.add_line(p0, p1);
    let side = s.add_line(p1, p2);

    s.add_constraint(Constraint::Fixed(p0));
    s.add_constraint(Constraint::Horizontal(base));
    s.add_constraint(Constraint::Perpendicular(base, side));
    s.add_constraint(Constraint::EqualLength(base, side));
    s.add_constraint(Constraint::Distance(p0, p1, 10.0));

    let solution = solve_sketch(&mut s);
    assert!(solution.report.converged);

    let p1s = s.point(p1);
    let p2s = s.point(p2);
    assert!(close(p1s.x, 10.0) && close(p1s.y, 0.0), "p1 ({}, {})", p1s.x, p1s.y);
    // The side is perpendicular to a horizontal base and equal in length, so p2
    // is directly above p1 by 10 (sign depends on the initial guess: +y here).
    assert!(close(p2s.x, 10.0) && close(p2s.y.abs(), 10.0), "p2 ({}, {})", p2s.x, p2s.y);
}

#[test]
fn radius_constraint_drives_circle_size() {
    let mut s = Sketch2d::new();
    let c = s.add_point(0.0, 0.0);
    let circle = s.add_circle(c, 3.0);
    s.add_constraint(Constraint::Fixed(c));
    s.add_constraint(Constraint::Radius(circle, 12.5));

    let solution = solve_sketch(&mut s);
    assert!(solution.report.converged);
    assert!(close(s.circle(circle).radius, 12.5));
}

#[test]
fn parallel_converges_at_large_scale() {
    // Two ~800 mm lines, slightly skewed. The raw cross-product residual is in
    // mm² (~1e5 per milliradian here), which spuriously failed the solver's
    // absolute tolerance; normalized to sin(angle) it must converge.
    let mut s = Sketch2d::new();
    let p0 = s.add_point(0.0, 0.0);
    let p1 = s.add_point(800.0, 3.0);
    let p2 = s.add_point(0.0, 200.0);
    let p3 = s.add_point(790.0, 212.0);

    let a = s.add_line(p0, p1);
    let b = s.add_line(p2, p3);

    s.add_constraint(Constraint::Fixed(p0));
    s.add_constraint(Constraint::Fixed(p1));
    s.add_constraint(Constraint::Fixed(p2));
    s.add_constraint(Constraint::Parallel(a, b));

    let solution = solve_sketch(&mut s);
    assert!(solution.report.converged, "norm {}", solution.report.residual_norm);

    // b ends up truly parallel to a.
    let (q0, q1, q2, q3) = (s.point(p0), s.point(p1), s.point(p2), s.point(p3));
    let cross = (q1.x - q0.x) * (q3.y - q2.y) - (q1.y - q0.y) * (q3.x - q2.x);
    let sin = cross / ((q1.x - q0.x).hypot(q1.y - q0.y) * (q3.x - q2.x).hypot(q3.y - q2.y));
    assert!(sin.abs() < 1e-8, "sin(angle) {sin}");
}

#[test]
fn coincident_merges_two_points() {
    let mut s = Sketch2d::new();
    let p0 = s.add_point(0.0, 0.0);
    let p1 = s.add_point(5.0, 4.0);
    s.add_constraint(Constraint::Fixed(p0));
    s.add_constraint(Constraint::Coincident(p0, p1));

    let solution = solve_sketch(&mut s);
    assert!(solution.report.converged);
    let q = s.point(p1);
    assert!(close(q.x, 0.0) && close(q.y, 0.0), "p1 ({}, {})", q.x, q.y);
    assert!(solution.is_fully_constrained());
    assert!(!solution.is_over_constrained());
}

#[test]
fn contradictory_distances_fail_and_report_over_constraint() {
    // The same two points must be both 10 and 12 apart — unsatisfiable.
    let mut s = Sketch2d::new();
    let p0 = s.add_point(0.0, 0.0);
    let p1 = s.add_point(11.0, 0.0);
    let line = s.add_line(p0, p1);
    s.add_constraint(Constraint::Fixed(p0));
    s.add_constraint(Constraint::Horizontal(line));
    s.add_constraint(Constraint::Distance(p0, p1, 10.0));
    s.add_constraint(Constraint::Distance(p0, p1, 12.0));

    let solution = solve_sketch(&mut s);
    assert!(!solution.report.converged, "a contradiction must not converge");
    assert!(solution.is_over_constrained(), "redundant {}", solution.redundant);
    // The two distance rows collapse to one independent equation, so the
    // geometry still reads as fully constrained — just inconsistently so.
    assert!(solution.redundant >= 1);
}

#[test]
fn underconstrained_sketch_reports_free_dofs() {
    // A lone line with only one endpoint fixed: the other end is free (2 dof).
    let mut s = Sketch2d::new();
    let p0 = s.add_point(0.0, 0.0);
    let _p1 = s.add_point(10.0, 0.0);
    s.add_constraint(Constraint::Fixed(p0));

    let solution = solve_sketch(&mut s);
    assert!(solution.report.converged);
    assert_eq!(solution.variables, 4);
    assert_eq!(solution.degrees_of_freedom, 2, "the free endpoint has 2 dof");
    assert!(!solution.is_fully_constrained());
}

#[test]
fn an_impossible_triangle_is_reported_not_silently_written() {
    // Sides 10, 10 and 30 can't close: the triangle inequality fails, so no
    // arrangement satisfies every distance. The Jacobian is still full rank
    // (nothing is *redundant*), which is exactly the case the old
    // "over-constrained" check missed.
    let mut s = Sketch2d::new();
    let p0 = s.add_point(0.0, 0.0);
    let p1 = s.add_point(9.0, 0.5);
    let p2 = s.add_point(20.0, 8.0);
    let base = s.add_line(p0, p1);
    s.add_constraint(Constraint::Fixed(p0));
    s.add_constraint(Constraint::Horizontal(base));
    s.add_constraint(Constraint::Distance(p0, p1, 10.0));
    s.add_constraint(Constraint::Distance(p1, p2, 10.0));
    s.add_constraint(Constraint::Distance(p0, p2, 30.0));
    let solution = solve_sketch(&mut s);
    assert!(!solution.report.converged);
    assert!(!solution.status().is_converged(), "{:?}", solution.status());
    assert!(
        matches!(solution.status(), SolveStatus::Stalled { residual_norm } if residual_norm > 1.0),
        "{:?}",
        solution.status()
    );
    // The trait form agrees.
    let mut again = s.clone();
    assert!(!LmSolver.solve(&mut again).is_converged());
}

#[test]
fn constraints_with_stale_ids_are_skipped_not_indexed() {
    // A hand-edited file naming a line that doesn't exist must not panic.
    let mut s = Sketch2d::new();
    let p0 = s.add_point(0.0, 0.0);
    let p1 = s.add_point(10.0, 1.0);
    let base = s.add_line(p0, p1);
    s.add_constraint(Constraint::Horizontal(base));
    s.add_constraint(Constraint::Horizontal(LineId(7)));
    s.add_constraint(Constraint::Distance(p0, PointId(42), 5.0));
    s.add_constraint(Constraint::Radius(CircleId(0), 2.0));
    let solution = solve_sketch(&mut s);
    assert!(solution.report.converged);
    assert_eq!(solution.invalid_constraints, 3);
    assert!((s.point(p0).y - s.point(p1).y).abs() < 1e-6, "the valid constraint still solved");
}
