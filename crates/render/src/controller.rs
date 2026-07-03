//! The crate's public contract: the [`Controller`] trait the application
//! implements, plus the data types exchanged across it (mesh, highlights,
//! grid, gizmo handles, transform deltas).

use glam::Vec3;

use crate::theme::ViewportTheme;
use crate::view::ViewContext;
use crate::{EdgeVertex, Vertex};

pub use cdt_core::Pick;

/// A mesh ready for display: interleaved face vertices + triangle indices, plus
/// crisp edge lines.
#[derive(Clone, Default)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub edge_vertices: Vec<EdgeVertex>,
    pub edge_indices: Vec<u32>,
}

/// The application's hook into the viewport.
///
/// `ui` draws egui for the frame and returns `true` if the displayed model
/// needs rebuilding; `mesh` then produces the new geometry. `mesh` is also
/// called once at startup for the initial display.
pub trait Controller {
    /// Draw this frame's egui UI. `view` projects between the sketch plane and
    /// the screen for overlay drawing. Return true if the model changed.
    fn ui(&mut self, ctx: &egui::Context, view: &ViewContext) -> bool;
    /// Produce the mesh to display.
    fn mesh(&mut self) -> MeshData;
    /// The faces to emphasize this frame (selected + hovered).
    fn highlights(&self) -> Highlights {
        Highlights::default()
    }
    /// A viewport click resolved to this entity (or `None` for empty space).
    /// `point` is the clicked world point on it (a face's surface or a point on
    /// an edge), used to build durable anchors; `None` if unavailable.
    /// `additive` is true when ⇧/⌘ is held — add/remove from a multi-selection.
    fn on_pick(&mut self, _pick: Option<Pick>, _point: Option<[f64; 3]>, _additive: bool) {}
    /// The entity currently under the cursor (or `None`), updated as it moves.
    fn on_hover(&mut self, _pick: Option<Pick>) {}
    /// Whether the viewport should resolve picks right now. (Disabled e.g.
    /// while drawing a sketch, where clicks mean something else.)
    fn wants_picking(&self) -> bool {
        false
    }

    /// The window title — typically the app name plus the open file's name, with
    /// a mark for unsaved changes. The host re-applies it when it changes.
    fn title(&self) -> String {
        "Cadette".to_string()
    }

    /// Begin a push/pull on `pick` if it's manipulable. `point` is the clicked
    /// world point (on the face); the controller captures the anchor (orienting
    /// the normal toward `eye`) and returns the drag axis `(point, outward
    /// normal)`. `None` means not manipulable, so the viewport orbits instead.
    fn start_manipulation(
        &mut self,
        _pick: Pick,
        _point: [f64; 3],
        _eye: [f64; 3],
    ) -> Option<([f64; 3], [f64; 3])> {
        None
    }
    /// Update the active push/pull to `distance` along the axis. Returns whether
    /// geometry changed (so the host re-meshes).
    fn update_manipulation(&mut self, _distance: f64) -> bool {
        false
    }
    /// Finish the active push/pull: commit it, or cancel and discard.
    fn finish_manipulation(&mut self, _commit: bool) {}

    /// The transform gizmo to show this frame, or `None` to hide it. Shown at a
    /// selected body (gumball-style).
    fn gizmo(&self) -> Option<Gizmo> {
        None
    }
    /// Begin a transform on `handle` (a gizmo handle was grabbed).
    fn start_transform(&mut self, _handle: GizmoHandle) {}
    /// Update the active transform by `delta` (an absolute translation/rotation
    /// from the grab point). Returns whether geometry changed (so we re-mesh).
    fn update_transform(&mut self, _delta: TransformDelta) -> bool {
        false
    }
    /// Finish the active transform: commit it, or cancel and discard.
    fn finish_transform(&mut self, _commit: bool) {}

    /// If the selection is a resizable primitive, its world-space axis-aligned
    /// bounds. The gizmo then shows resize grips on the +X/+Y/+Z faces, edited
    /// by dragging. Default: not resizable.
    fn resize_box(&self) -> Option<ResizeBox> {
        None
    }
    /// Begin a resize drag from the `positive` (or negative) face along `axis`
    /// (the whole drag is one undo entry).
    fn start_resize(&mut self, _axis: Axis3, _positive: bool) {}
    /// Set the box's extent (mm) along `axis` to `extent`. Returns true if the
    /// dimension changed.
    fn update_resize(&mut self, _axis: Axis3, _extent: f64) -> bool {
        false
    }
    /// Finish the active resize: commit it, or cancel and restore the size.
    fn finish_resize(&mut self, _commit: bool) {}

    /// The ground-plane grid: its spacing (mm) drives translation snapping and,
    /// when visible, the workplane overlay. Default = no grid / no snap.
    fn grid(&self) -> GridSpec {
        GridSpec::default()
    }

    /// The viewport's color scheme (background, material, grid, ViewCube), so it
    /// can follow the app's light/dark theme. Defaults to the dark studio look.
    fn viewport_theme(&self) -> ViewportTheme {
        ViewportTheme::DARK
    }
}

/// The construction-grid configuration: spacing/extent shared by the three
/// origin planes, plus which of them to draw as guides.
#[derive(Clone, Copy)]
pub struct GridSpec {
    /// Cell spacing (mm) — line density and snap increment. `<= 0` disables.
    pub spacing: f32,
    /// Half the total grid size (mm); the grid spans `[-half_extent, half_extent]`.
    pub half_extent: f32,
    /// Whether to draw the XY ground plane (the primary workplane).
    pub visible: bool,
    /// Whether to draw the XZ origin plane (vertical, front) as a guide.
    pub xz: bool,
    /// Whether to draw the YZ origin plane (vertical, side) as a guide.
    pub yz: bool,
    /// Whether moves/resizes snap to the grid.
    pub snap: bool,
}

impl Default for GridSpec {
    fn default() -> Self {
        GridSpec {
            spacing: 0.0,
            half_extent: 0.0,
            visible: false,
            xz: false,
            yz: false,
            snap: false,
        }
    }
}

impl GridSpec {
    /// The effective snap increment: the spacing if snapping is on, else 0 (off).
    pub(crate) fn snap_step(&self) -> f32 {
        if self.snap {
            self.spacing
        } else {
            0.0
        }
    }
}

/// A transform gizmo shown at a selected body.
pub struct Gizmo {
    /// World-space origin — the body's bounding-box center.
    pub origin: [f64; 3],
}

/// The world-space bounds of a resizable primitive. The gizmo draws a small
/// grip at the center of each +X/+Y/+Z face; dragging one edits that dimension.
#[derive(Clone, Copy)]
pub struct ResizeBox {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

/// A draggable gizmo handle.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GizmoHandle {
    /// Translate along a world axis (an arrow).
    TranslateAxis(Axis3),
    /// Translate within a world plane (a corner square).
    TranslatePlane(Plane3),
    /// Rotate about a world axis (a ring).
    RotateAxis(Axis3),
    /// Resize a primitive along a world axis (a bounding-box grip).
    Resize(Axis3),
}

/// An absolute transform from the grab point, applied by the controller.
#[derive(Clone, Copy, Debug)]
pub enum TransformDelta {
    /// World-space translation offset.
    Translate([f64; 3]),
    /// Rotation by `angle` radians about `axis` (through the gizmo pivot).
    Rotate { axis: [f64; 3], angle: f64 },
}

/// One of the three world axes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Axis3 {
    X,
    Y,
    Z,
}

impl Axis3 {
    pub(crate) const ALL: [Axis3; 3] = [Axis3::X, Axis3::Y, Axis3::Z];

    /// Unit direction.
    pub(crate) fn dir(self) -> Vec3 {
        match self {
            Axis3::X => Vec3::X,
            Axis3::Y => Vec3::Y,
            Axis3::Z => Vec3::Z,
        }
    }

    /// The two unit vectors spanning the plane perpendicular to this axis (the
    /// in-plane basis for the rotation ring).
    pub(crate) fn perps(self) -> (Vec3, Vec3) {
        match self {
            Axis3::X => (Vec3::Y, Vec3::Z),
            Axis3::Y => (Vec3::Z, Vec3::X),
            Axis3::Z => (Vec3::X, Vec3::Y),
        }
    }

    /// Base color (X red, Y green, Z blue).
    pub(crate) fn color(self) -> [f32; 3] {
        match self {
            Axis3::X => [0.90, 0.25, 0.28],
            Axis3::Y => [0.35, 0.78, 0.30],
            Axis3::Z => [0.30, 0.45, 0.95],
        }
    }
}

/// One of the three world planes (for plane-constrained translation).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Plane3 {
    Xy,
    Yz,
    Zx,
}

impl Plane3 {
    pub(crate) const ALL: [Plane3; 3] = [Plane3::Xy, Plane3::Yz, Plane3::Zx];

    /// The two in-plane axes.
    pub(crate) fn axes(self) -> (Axis3, Axis3) {
        match self {
            Plane3::Xy => (Axis3::X, Axis3::Y),
            Plane3::Yz => (Axis3::Y, Axis3::Z),
            Plane3::Zx => (Axis3::Z, Axis3::X),
        }
    }

    /// The axis normal to the plane.
    pub(crate) fn normal(self) -> Axis3 {
        match self {
            Plane3::Xy => Axis3::Z,
            Plane3::Yz => Axis3::X,
            Plane3::Zx => Axis3::Y,
        }
    }

    /// Handle color — the normal axis's color, lightened so a plane reads as a
    /// plane rather than an axis.
    pub(crate) fn color(self) -> [f32; 3] {
        let c = self.normal().color();
        [
            c[0] + (1.0 - c[0]) * 0.35,
            c[1] + (1.0 - c[1]) * 0.35,
            c[2] + (1.0 - c[2]) * 0.35,
        ]
    }
}

/// Entities to emphasize this frame: a clicked selection set (strong) and a
/// single hover pre-highlight (subtle). Faces and edges both highlight as a
/// set (for multi-face fillet/group and multi-edge fillet).
#[derive(Clone, Default)]
pub struct Highlights {
    pub selected: Vec<Pick>,
    pub hovered: Option<Pick>,
}
