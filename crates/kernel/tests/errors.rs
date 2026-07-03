//! The kernel's central safety promise: an OCCT `Standard_Failure` surfaces as
//! a Rust `Err` naming the operation — never a crash or silent corruption.

use cdt_kernel::Solid;

#[test]
fn infeasible_fillet_radius_is_an_error() {
    let cube = Solid::cuboid(10.0, 10.0, 10.0).unwrap();
    // A radius larger than the cube itself cannot be constructed; OCCT throws
    // and the guard must translate that into an Err, not UB or a panic.
    let result = cube.fillet_all_edges(50.0);
    let err = result.err().expect("oversized fillet must fail");
    assert!(
        err.to_string().contains("fillet_all_edges"),
        "error names the operation: {err}"
    );
}

#[test]
fn extruding_a_solid_with_no_planar_profile_is_an_error() {
    // `extrude` expects a face profile; a sphere has no planar face to pull.
    let sphere = Solid::sphere(5.0).unwrap();
    assert!(sphere.extrude(3.0).is_err(), "extruding a sphere must fail");
}

#[test]
fn push_pull_with_no_matching_face_is_an_error() {
    let cube = Solid::cuboid(10.0, 10.0, 10.0).unwrap();
    // No face of the cube is anywhere near this point/normal pair.
    let result = cube.push_pull([500.0, 500.0, 500.0], [0.0, 0.0, 1.0], 2.0);
    assert!(result.is_err(), "push_pull far from any face must fail");
}

#[test]
fn degenerate_primitive_dimensions_are_errors() {
    assert!(Solid::cuboid(0.0, 10.0, 10.0).is_err(), "zero-size box");
    assert!(Solid::sphere(-1.0).is_err(), "negative-radius sphere");
}
