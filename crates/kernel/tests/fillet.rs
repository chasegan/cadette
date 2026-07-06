//! Filleting edges whose rounded surfaces overlap. When a fillet radius is too
//! large for the face between two rounded edges, OCCT's MakeFillet reports done
//! but emits a self-intersecting sliver: an INVALID solid that renders fine yet
//! silently corrupts downstream booleans/STL. Healing only strips it into an
//! OPEN shell (not watertight), so the fillet must fail loudly with an
//! actionable message instead — while radii that actually fit still succeed.

use cdt_kernel::Solid;

/// The "microwave power hole" ring the user hit: a r35.5 stack unioned with a
/// r39 flange, bored by a r33 cylinder, leaving a 2.5mm-wide top annulus at
/// z=9.75 whose outer (r35.5) and inner (r33) top edges we fillet.
fn ring() -> Solid {
    let cyl2 = Solid::cylinder(35.5, 9.75).unwrap();
    let cyl1 = Solid::cylinder(39.0, 2.0).unwrap();
    let bore = Solid::cylinder(33.0, 20.0)
        .unwrap()
        .translate(0.0, 0.0, -6.0)
        .unwrap();
    cyl2.fuse(&cyl1).unwrap().cut(&bore).unwrap()
}

#[test]
fn overlapping_fillets_are_a_clear_error_not_a_broken_body() {
    // Radius 2.0 on both top edges: 2 + 2 > the 2.5mm annulus, so the fillets
    // overlap. Pre-fix this returned an INVALID self-intersecting solid (the
    // "floating sliver") that rendered but corrupted booleans/STL.
    let err = ring()
        .fillet_edges(&[[-35.5, 0.0, 9.75], [-33.0, 0.0, 9.75]], 2.0)
        .err()
        .expect("overlapping fillets must fail, not emit a broken body");
    assert!(
        err.to_string().contains("radius is too large"),
        "the error explains the cause: {err}"
    );
}

#[test]
fn a_fillet_that_fits_still_produces_a_valid_solid() {
    // Radius 1.0 on both edges: 1 + 1 < 2.5, so the fillets fit — a clean,
    // watertight solid (guards against the error path being too aggressive).
    let filleted = ring()
        .fillet_edges(&[[-35.5, 0.0, 9.75], [-33.0, 0.0, 9.75]], 1.0)
        .unwrap();
    let desc = filleted.describe().unwrap();
    assert!(desc.contains("solids=1"), "a fitting fillet is one solid: {desc}");
    assert!(desc.contains("closed 1"), "the solid is watertight: {desc}");
    assert!(!desc.contains("INVALID"), "and it passes BRepCheck: {desc}");

    // A downstream boolean operates on it (an invalid input would be dropped).
    let v = filleted.volume().unwrap();
    let tool = Solid::cuboid(10.0, 12.0, 60.0)
        .unwrap()
        .translate(30.0, -6.0, -10.0)
        .unwrap();
    let vc = filleted.cut(&tool).unwrap().volume().unwrap();
    assert!(vc > 0.0 && vc < v, "the cut removes part of the body: {vc} of {v}");
}
