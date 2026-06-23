//! 2D sketch primitives: the planar profiles that become solids via extrude.
//!
//! This is the MVP seed of the sketcher. A sketch lives on a [`SketchPlane`] and
//! carries one closed [`Profile`]. Later phases grow this into multi-curve
//! sketches with a constraint graph (line/arc/spline + dimensions); for now a
//! rectangle or circle is enough to drive a real sketch → extrude pipeline.

use glam::DVec3;
use serde::{Deserialize, Serialize};

/// One of the three world-aligned base planes, passing through the origin.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum SketchPlane {
    /// X-Y plane, normal +Z (the default; matches the Z-up world).
    Xy,
    /// X-Z plane, normal -Y.
    Xz,
    /// Y-Z plane, normal +X.
    Yz,
}

impl SketchPlane {
    /// Origin of the sketch frame (the world origin for MVP).
    pub fn origin(self) -> DVec3 {
        DVec3::ZERO
    }

    /// In-plane "right" axis (local +x).
    pub fn x_dir(self) -> DVec3 {
        match self {
            SketchPlane::Xy | SketchPlane::Xz => DVec3::X,
            SketchPlane::Yz => DVec3::Y,
        }
    }

    /// In-plane "up" axis (local +y).
    pub fn y_dir(self) -> DVec3 {
        match self {
            SketchPlane::Xy => DVec3::Y,
            SketchPlane::Xz | SketchPlane::Yz => DVec3::Z,
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
