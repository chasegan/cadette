//! Viewport drag state: the in-progress push/pull [`Manipulation`], gizmo
//! [`TransformDrag`], and bounding-box-grip [`ResizeDrag`] (with its
//! [`ResizeMode`]), plus the axis/bounds helpers the drag code shares.

use cdt_core::{FaceAnchor, FeatureId, FeatureKind, SketchPlane, DVec3};
use cdt_render::{Axis3, Pick};

use crate::Modeler;

/// An in-progress push/pull drag: the body being modified, the face anchor, and
/// the feature once it's been created (on the first non-trivial drag).
#[derive(Clone, Copy)]
pub(crate) struct Manipulation {
    pub(crate) source: FeatureId,
    pub(crate) anchor: FaceAnchor,
    pub(crate) feature: Option<FeatureId>,
}

/// An in-progress gizmo transform: the body being moved/rotated, the feature
/// once the first drag created it (or the body's own tip `Translate` when the
/// drag reuses it — see `base_offset`), and (for rotation) the fixed pivot —
/// the body's AABB center drifts as it turns, so the gizmo and rotation axis
/// stay pinned at the center captured when the drag began.
#[derive(Clone, Copy)]
pub(crate) struct TransformDrag {
    pub(crate) source: FeatureId,
    pub(crate) feature: Option<FeatureId>,
    pub(crate) pivot: Option<[f64; 3]>,
    /// The tip `Translate`'s offset at grab when this drag REUSES the body's
    /// own trailing Move (the drag's delta adds onto it, so a body keeps a
    /// single trailing `Translate` instead of stacking one per drag). `ZERO`
    /// when the drag splices a fresh feature.
    pub(crate) base_offset: DVec3,
    /// If the source is a SKETCH, its original plane — the drag moves the sketch
    /// IN PLACE (transforms this plane) instead of wrapping it in a Translate/
    /// Rotate, so it stays an editable, extrudable sketch.
    pub(crate) orig_plane: Option<SketchPlane>,
    /// True once a non-trivial drag has recorded an undo and begun editing.
    pub(crate) started: bool,
}

/// How an active resize edits the model.
pub(crate) enum ResizeMode {
    /// Edit a bare primitive `Box`'s `size` in place — parametric, no new
    /// feature (only when the selected body's tip feature is a `Box`).
    BoxDim,
    /// Edit a bare `Cylinder`'s `radius` (X/Y grips) or `height` (Z grip) in
    /// place — keeps it circular, instead of scaling it elliptical.
    CylDim,
    /// Edit a bare `Sphere`'s `radius` in place (any grip) — stays a sphere.
    SphereDim,
    /// Add a non-uniform `Scale` feature to a body with no existing one —
    /// works on any body (extrudes, booleans, fillets, …), Tinkercad-style.
    NewScale,
    /// Edit the factors of an existing tip `Scale` feature. Resizing again
    /// reuses the one `Scale` rather than stacking a second — two GTransforms
    /// in a row segfault OCCT on complex (filleted/NURBS) bodies.
    EditScale,
}

/// Active resize drag (a bounding-box grip).
pub(crate) struct ResizeDrag {
    /// BoxDim: the `Box` feature (edited in place). NewScale: the body to scale
    /// (the new `Scale`'s input). EditScale: unused (`feature` holds the Scale).
    pub(crate) source: FeatureId,
    pub(crate) mode: ResizeMode,
    /// The body's bbox extents at grab — the base for the new dimension (BoxDim)
    /// or the scale factor. For a `Box` this equals its `size`.
    pub(crate) base_extent: DVec3,
    /// The existing `Scale`'s factors at grab (`ONE` if none) — EditScale
    /// multiplies onto these.
    pub(crate) base_factors: DVec3,
    /// The fixed scale anchor: the existing `Scale`'s anchor, or the bbox-min
    /// corner for a fresh one.
    pub(crate) anchor: DVec3,
    /// The `Scale` feature being edited: `Some` from the start for EditScale,
    /// `None` until the first change for NewScale, always `None` for BoxDim.
    pub(crate) feature: Option<FeatureId>,
    /// Whether any update has changed geometry yet (the first records one undo).
    pub(crate) changed: bool,
}

/// The value of `v`'s component along `axis`.
pub(crate) fn axis_get(v: DVec3, axis: Axis3) -> f64 {
    match axis {
        Axis3::X => v.x,
        Axis3::Y => v.y,
        Axis3::Z => v.z,
    }
}

/// Set `v`'s component along `axis` to `val`.
pub(crate) fn axis_set(v: &mut DVec3, axis: Axis3, val: f64) {
    match axis {
        Axis3::X => v.x = val,
        Axis3::Y => v.y = val,
        Axis3::Z => v.z = val,
    }
}

/// Axis-aligned bounds `(min, max)` of a flat `[x,y,z,...]` position array.
pub(crate) fn positions_bounds(positions: &[f32]) -> ([f64; 3], [f64; 3]) {
    if positions.is_empty() {
        return ([0.0; 3], [0.0; 3]);
    }
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for p in positions.chunks_exact(3) {
        for k in 0..3 {
            min[k] = min[k].min(p[k]);
            max[k] = max[k].max(p[k]);
        }
    }
    (
        [min[0] as f64, min[1] as f64, min[2] as f64],
        [max[0] as f64, max[1] as f64, max[2] as f64],
    )
}

/// Bounding-box center of a flat `[x,y,z,...]` position array.
pub(crate) fn positions_center(positions: &[f32]) -> [f64; 3] {
    let (min, max) = positions_bounds(positions);
    [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ]
}

impl Modeler {
    /// A short readout for an in-progress gizmo drag (angle / distance), shown
    /// floating next to the gizmo so the manipulation has a numeric value.
    pub(crate) fn transform_readout(&self) -> Option<String> {
        let id = self.transform?.feature?;
        match self.doc.history.get(id).map(|f| &f.kind) {
            Some(FeatureKind::Rotate { angle, .. }) => Some(format!("{:.0}°", angle.to_degrees())),
            Some(FeatureKind::Translate { offset, .. }) => {
                Some(format!("{:.1} mm", offset.length()))
            }
            _ => None,
        }
    }

    /// The gizmo origin — the bounding-box center of a single selected face's
    /// body (the only selection that shows the gizmo).
    pub(crate) fn gizmo_origin(&self) -> Option<[f64; 3]> {
        match self.selection.selected() {
            [Pick::Face(_)] => {
                let body = self.selected_body()?;
                let idx = self.ui.visible.iter().position(|&f| f == body)?;
                self.body_centers.get(idx).copied()
            }
            _ => None,
        }
    }
}
