//! The serializable view of a session: every feature with its status, every
//! visible body with exact measurements, and the errors — what an agent reads
//! instead of pixels.

use cdt_core::{Document, FeatureId, FeatureKind, RegenError, Regeneration};
use cdt_kernel::{Body, CurveKind, KernelError, Solid, SurfaceKind};
use serde::{Deserialize, Serialize};

use crate::{error_message, kind_summary, SessionError};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub name: String,
    pub units: String,
    /// Count of leading features that are active.
    pub rollback: usize,
    /// True when every active feature built.
    pub ok: bool,
    pub features: Vec<FeatureReport>,
    /// The visible bodies, in history order.
    pub bodies: Vec<BodyReport>,
    pub errors: Vec<ErrorReport>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureStatus {
    Built,
    Failed,
    Suppressed,
    RolledBack,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureReport {
    pub id: FeatureId,
    pub name: String,
    /// One line, e.g. `Box 40×13×26` or `Subtract #12 with #9`.
    pub summary: String,
    /// The full definition, in the `.cdt` form.
    pub kind: FeatureKind,
    pub inputs: Vec<FeatureId>,
    pub suppressed: bool,
    pub status: FeatureStatus,
    /// Whether this feature's body is one of the visible ones.
    pub visible: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<[f32; 3]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BodyReport {
    /// The feature that produced this body.
    pub feature: FeatureId,
    pub name: String,
    /// Tight axis-aligned bounds `[min, max]`.
    pub bbox: [[f64; 3]; 2],
    pub volume: f64,
    pub surface_area: f64,
    pub centroid: [f64; 3],
    /// How many separate solids the body holds — more than one for a
    /// Pattern, a Group, or an imported assembly.
    pub solids: usize,
    pub faces: usize,
    pub edges: usize,
    /// OCCT's validity check (closed, non-self-intersecting).
    pub valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ErrorReport {
    pub feature: FeatureId,
    pub name: String,
    pub message: String,
}

/// Per-face and per-edge detail of one body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BodyDetail {
    pub feature: FeatureId,
    pub faces: Vec<FaceReport>,
    pub edges: Vec<EdgeReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FaceReport {
    /// The face index tessellation, picking and highlights use.
    pub index: u32,
    pub surface: String,
    pub area: f64,
    pub centroid: [f64; 3],
    /// Outward unit normal.
    pub normal: [f64; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub radius: Option<f64>,
    /// For cylinders, cones and tori: a point on the surface axis and its
    /// unit direction — what a Thread's helix takes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub axis_point: Option<[f64; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub axis: Option<[f64; 3]>,
    pub edges: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeReport {
    pub index: u32,
    pub curve: String,
    pub length: f64,
    pub start: [f64; 3],
    pub end: [f64; 3],
    /// A point on the edge — what a fillet's or chamfer's `EdgeAnchor` takes.
    pub midpoint: [f64; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub radius: Option<f64>,
    pub faces: Vec<u32>,
}

/// Build the report for a document and its regeneration — for hosts that own
/// their own backend and cache (the app) rather than a [`crate::Session`].
pub fn build_report(doc: &Document, regen: &Regeneration<Body, KernelError>) -> Report {
    let active = doc.rollback();
    let errors: Vec<&RegenError<KernelError>> = regen.errors().iter().collect();
    let error_for = |id: FeatureId| errors.iter().find(|e| e.feature() == id).map(|e| error_message(e, doc));
    let features = doc
        .history
        .features()
        .iter()
        .enumerate()
        .map(|(index, f)| {
            let error = error_for(f.id);
            let status = if f.suppressed {
                FeatureStatus::Suppressed
            } else if index >= active {
                FeatureStatus::RolledBack
            } else if error.is_some() {
                FeatureStatus::Failed
            } else {
                FeatureStatus::Built
            };
            FeatureReport {
                id: f.id,
                name: f.name.clone(),
                summary: kind_summary(&f.kind),
                kind: f.kind.clone(),
                inputs: f.kind.inputs(),
                suppressed: f.suppressed,
                status,
                visible: regen.visible().contains(&f.id),
                error,
                color: f.color,
            }
        })
        .collect();
    let bodies = regen
        .visible()
        .iter()
        .filter_map(|&id| {
            let body = regen.body(id)?;
            let name = doc.history.get(id).map(|f| f.name.clone()).unwrap_or_default();
            let props = body.mass_properties().ok();
            let topo = body.topology().ok();
            Some(BodyReport {
                feature: id,
                name,
                bbox: props.map(|p| [p.bbox.0, p.bbox.1]).unwrap_or_default(),
                volume: props.map(|p| p.volume).unwrap_or(f64::NAN),
                surface_area: props.map(|p| p.surface_area).unwrap_or(f64::NAN),
                centroid: props.map(|p| p.centroid).unwrap_or([f64::NAN; 3]),
                solids: topo.as_ref().map(|t| t.solids as usize).unwrap_or(0),
                faces: topo.as_ref().map(|t| t.faces as usize).unwrap_or(0),
                edges: topo.as_ref().map(|t| t.edges as usize).unwrap_or(0),
                valid: topo.as_ref().map(|t| t.is_valid).unwrap_or(false),
            })
        })
        .collect();
    let errors = errors
        .iter()
        .map(|e| ErrorReport {
            feature: e.feature(),
            name: doc.history.get(e.feature()).map(|f| f.name.clone()).unwrap_or_default(),
            message: error_message(e, doc),
        })
        .collect();
    Report {
        name: doc.name.clone(),
        units: format!("{:?}", doc.units),
        rollback: active,
        ok: regen.is_ok(),
        features,
        bodies,
        errors,
    }
}

/// Per-face and per-edge detail of one solid.
pub fn body_detail(id: FeatureId, solid: &Solid) -> Result<BodyDetail, SessionError> {
    let faces = solid
        .faces()?
        .into_iter()
        .map(|f| FaceReport {
            index: f.index,
            surface: surface_name(f.surface).to_string(),
            area: f.area,
            centroid: f.centroid,
            normal: f.normal,
            radius: f.radius,
            axis_point: f.axis.map(|(p, _)| p),
            axis: f.axis.map(|(_, d)| d),
            edges: f.edges,
        })
        .collect();
    let edges = solid
        .edges()?
        .into_iter()
        .map(|e| EdgeReport {
            index: e.index,
            curve: curve_name(e.curve).to_string(),
            length: e.length,
            start: e.start,
            end: e.end,
            midpoint: e.midpoint,
            radius: e.radius,
            faces: e.faces,
        })
        .collect();
    Ok(BodyDetail { feature: id, faces, edges })
}

fn surface_name(s: SurfaceKind) -> &'static str {
    match s {
        SurfaceKind::Plane => "plane",
        SurfaceKind::Cylinder => "cylinder",
        SurfaceKind::Cone => "cone",
        SurfaceKind::Sphere => "sphere",
        SurfaceKind::Torus => "torus",
        SurfaceKind::Bezier => "bezier",
        SurfaceKind::BSpline => "bspline",
        SurfaceKind::Other => "other",
    }
}

fn curve_name(c: CurveKind) -> &'static str {
    match c {
        CurveKind::Line => "line",
        CurveKind::Circle => "circle",
        CurveKind::Ellipse => "ellipse",
        CurveKind::Bezier => "bezier",
        CurveKind::BSpline => "bspline",
        CurveKind::Other => "other",
    }
}
