//! End-to-end constraint-solving tests: build a sketch, solve it, and assert
//! the geometry lands where the constraints demand.

use rmf_core::{Constraint, Sketch2d};
use rmf_solver::solve_sketch;

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
