//! 2D sketch primitives: the planar profiles that become solids via extrude.
//!
//! This is the MVP seed of the sketcher. A sketch lives on a [`SketchPlane`] and
//! carries one closed [`Profile`]. Later phases grow this into multi-curve
//! sketches with a constraint graph (line/arc/spline + dimensions); for now a
//! rectangle or circle is enough to drive a real sketch → extrude pipeline.

use glam::DVec3;
use serde::{Deserialize, Serialize};

/// The plane a sketch lives on: one of the world base planes, or a custom frame
/// (e.g. a selected face).
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum SketchPlane {
    /// X-Y plane, normal +Z (the default; matches the Z-up world).
    Xy,
    /// X-Z plane, normal -Y.
    Xz,
    /// Y-Z plane, normal +X.
    Yz,
    /// An arbitrary plane: origin plus in-plane x/y axes (from a face).
    Custom {
        origin: DVec3,
        x_dir: DVec3,
        y_dir: DVec3,
    },
}

impl SketchPlane {
    /// A custom plane from a face frame.
    pub fn from_frame(origin: DVec3, x_dir: DVec3, y_dir: DVec3) -> Self {
        SketchPlane::Custom {
            origin,
            x_dir,
            y_dir,
        }
    }

    /// Origin of the sketch frame.
    pub fn origin(self) -> DVec3 {
        match self {
            SketchPlane::Custom { origin, .. } => origin,
            _ => DVec3::ZERO,
        }
    }

    /// In-plane "right" axis (local +x).
    pub fn x_dir(self) -> DVec3 {
        match self {
            SketchPlane::Xy | SketchPlane::Xz => DVec3::X,
            SketchPlane::Yz => DVec3::Y,
            SketchPlane::Custom { x_dir, .. } => x_dir,
        }
    }

    /// In-plane "up" axis (local +y).
    pub fn y_dir(self) -> DVec3 {
        match self {
            SketchPlane::Xy => DVec3::Y,
            SketchPlane::Xz | SketchPlane::Yz => DVec3::Z,
            SketchPlane::Custom { y_dir, .. } => y_dir,
        }
    }

    /// Plane normal (right-handed: `x_dir × y_dir`). Extrude defaults to this.
    pub fn normal(self) -> DVec3 {
        self.x_dir().cross(self.y_dir())
    }

    /// Short label for UI display.
    pub fn label(self) -> &'static str {
        match self {
            SketchPlane::Xy => "XY",
            SketchPlane::Xz => "XZ",
            SketchPlane::Yz => "YZ",
            SketchPlane::Custom { .. } => "Face",
        }
    }
}

/// A closed 2D profile, centered on the sketch origin.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum Profile {
    /// Axis-aligned rectangle, centered on the origin.
    Rectangle { width: f64, height: f64 },
    /// Circle centered on the origin.
    Circle { radius: f64 },
}

impl Profile {
    pub fn type_name(self) -> &'static str {
        match self {
            Profile::Rectangle { .. } => "Rectangle",
            Profile::Circle { .. } => "Circle",
        }
    }
}

// ---------------------------------------------------------------------------
// Constraint-based 2D sketch
//
// The richer sketcher: points, lines, and circles related by geometric and
// dimensional constraints, solved numerically (see the `rmf-solver` crate).
// `Profile` above is the simple shortcut still used by the current extrude
// path; this model is what grows into real CAD sketching.
// ---------------------------------------------------------------------------

/// Index of a point in [`Sketch2d::points`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct PointId(pub usize);

/// Index of a line in [`Sketch2d::lines`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct LineId(pub usize);

/// Index of a circle in [`Sketch2d::circles`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct CircleId(pub usize);

/// A sketch point. `x`/`y` are the current/initial position in sketch-plane
/// coordinates; the solver updates them to satisfy the constraints.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SketchPoint {
    pub x: f64,
    pub y: f64,
}

/// A straight segment between two points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SketchLine {
    pub a: PointId,
    pub b: PointId,
}

/// A circle defined by a center point and a radius.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SketchCircle {
    pub center: PointId,
    pub radius: f64,
}

/// A geometric or dimensional constraint relating sketch entities.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Constraint {
    /// Pin a point at its current location.
    Fixed(PointId),
    /// Two points share a position.
    Coincident(PointId, PointId),
    /// A line is horizontal (its endpoints share `y`).
    Horizontal(LineId),
    /// A line is vertical (its endpoints share `x`).
    Vertical(LineId),
    /// The distance between two points equals `value`.
    Distance(PointId, PointId, f64),
    /// Two lines are parallel.
    Parallel(LineId, LineId),
    /// Two lines are perpendicular.
    Perpendicular(LineId, LineId),
    /// Two lines have equal length.
    EqualLength(LineId, LineId),
    /// A circle's radius equals `value`.
    Radius(CircleId, f64),
}

/// A 2D constraint sketch: entities plus the constraints relating them.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Sketch2d {
    pub points: Vec<SketchPoint>,
    pub lines: Vec<SketchLine>,
    pub circles: Vec<SketchCircle>,
    pub constraints: Vec<Constraint>,
}

impl Sketch2d {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_point(&mut self, x: f64, y: f64) -> PointId {
        self.points.push(SketchPoint { x, y });
        PointId(self.points.len() - 1)
    }

    pub fn add_line(&mut self, a: PointId, b: PointId) -> LineId {
        self.lines.push(SketchLine { a, b });
        LineId(self.lines.len() - 1)
    }

    pub fn add_circle(&mut self, center: PointId, radius: f64) -> CircleId {
        self.circles.push(SketchCircle { center, radius });
        CircleId(self.circles.len() - 1)
    }

    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    pub fn point(&self, id: PointId) -> SketchPoint {
        self.points[id.0]
    }

    pub fn line(&self, id: LineId) -> SketchLine {
        self.lines[id.0]
    }

    pub fn circle(&self, id: CircleId) -> SketchCircle {
        self.circles[id.0]
    }

    /// Order the line segments into a single closed loop of point coordinates
    /// `[x, y]` (in plane coordinates), ready to become a face. Returns `None`
    /// unless the lines form exactly one closed loop (each visited once,
    /// returning to the start) — the case the MVP extrude path supports.
    pub fn profile_loop(&self) -> Option<Vec<[f64; 2]>> {
        let n = self.lines.len();
        if n < 3 {
            return None;
        }

        let start = self.lines[0].a;
        let mut current = start;
        let mut visited = vec![false; n];
        let mut loop_points = Vec::with_capacity(n);

        for _ in 0..n {
            let p = self.point(current);
            loop_points.push([p.x, p.y]);

            // Follow an unvisited line incident to the current point.
            let next = self
                .lines
                .iter()
                .enumerate()
                .find(|(i, l)| !visited[*i] && (l.a == current || l.b == current));
            let (line_index, line) = next?;
            visited[line_index] = true;
            current = if line.a == current { line.b } else { line.a };
        }

        // A single closed loop returns to its start after visiting every line.
        (current == start).then_some(loop_points)
    }
}
