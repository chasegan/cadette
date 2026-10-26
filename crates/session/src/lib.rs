//! # cdt-session
//!
//! One place that composes the headless pieces — [`Document`], the OCCT
//! [`KernelBackend`], the sketch solver, the regeneration cache, undo/redo and
//! the `.cdt` file — into a [`Session`] that anything can drive: the CLI, the
//! app's control socket, tests, an AI agent. Edits go in as [`DocEdit`]
//! batches (one undo entry each); state comes out as a serializable
//! [`Report`] of every feature's status and every visible body's exact
//! measurements, so a caller can verify a model without looking at pixels.
//!
//! No GUI types: the `render` feature (on by default) adds
//! [`Session::scene_mesh`], which builds `cdt-render`'s display mesh, but
//! that crate is used without its windowing stack.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use cdt_core::{
    BatchError, DocEdit, Document, EditOutcome, FeatureId, FeatureKind, MissingReason,
    RegenCache, RegenError, Regeneration, SolveStatus,
};
use cdt_kernel::{Body, KernelBackend, KernelError, Solid};
use cdt_project::{ProjectError, ProjectFile};
use cdt_solver::LmSolver;
use serde::{Deserialize, Serialize};

pub mod link;
mod report;
#[cfg(feature = "render")]
mod scene;

pub use report::{
    build_report, body_detail, BodyDetail, BodyReport, EdgeReport, ErrorReport, FaceReport,
    FeatureReport, FeatureStatus, Report,
};
#[cfg(feature = "render")]
pub use scene::{BodyRange, SceneMesh};

/// Chord deviation for on-screen tessellation (mm).
pub const PREVIEW_DEFLECTION_MM: f64 = 0.1;
/// Finer chord deviation for STL export (mm).
pub const EXPORT_DEFLECTION_MM: f64 = 0.05;
/// How many undo snapshots a session keeps.
pub const UNDO_LIMIT: usize = 200;

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("{0}")]
    Project(#[from] ProjectError),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Kernel(#[from] KernelError),
    #[error("{0}")]
    Edit(#[from] BatchError),
    #[error("the session has no file path; use save_as")]
    NoPath,
    #[error("nothing to export: no visible body")]
    NoBodies,
    #[error("no feature with id {0:?}")]
    UnknownFeature(FeatureId),
}

/// A headless modelling session over one document.
pub struct Session {
    doc: Document,
    path: Option<PathBuf>,
    /// Archive entries other than the document (the app's `view.json`),
    /// preserved across save so a headless edit never strips them.
    extras: BTreeMap<String, Vec<u8>>,
    backend: KernelBackend,
    cache: RegenCache<Body>,
    solver: LmSolver,
    /// The last regeneration, `None` until the document changed since.
    regen: Option<Regeneration<Body, KernelError>>,
    undo: Vec<Document>,
    redo: Vec<Document>,
}

impl Session {
    /// A session over an in-memory document.
    pub fn new(doc: Document) -> Self {
        Self {
            doc,
            path: None,
            extras: BTreeMap::new(),
            backend: KernelBackend::default(),
            cache: RegenCache::new(),
            solver: LmSolver,
            regen: None,
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }

    /// Open a `.cdt` file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SessionError> {
        let project = cdt_project::read_path(path.as_ref())?;
        let mut session = Self::new(project.document);
        session.path = Some(path.as_ref().to_path_buf());
        session.extras = project.extras;
        Ok(session)
    }

    /// Save to the file this session was opened from (or last saved to).
    pub fn save(&self) -> Result<(), SessionError> {
        let path = self.path.clone().ok_or(SessionError::NoPath)?;
        self.save_as(path)
    }

    /// Save to `path`, preserving any extra archive entries, and bind the
    /// session to it.
    pub fn save_as(&self, path: impl AsRef<Path>) -> Result<(), SessionError> {
        let project = ProjectFile { document: self.doc.clone(), extras: self.extras.clone() };
        cdt_project::write_path(path, &project)?;
        Ok(())
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn document(&self) -> &Document {
        &self.doc
    }

    /// Opaque archive entries that travel with the document.
    pub fn extras(&self) -> &BTreeMap<String, Vec<u8>> {
        &self.extras
    }

    pub fn extras_mut(&mut self) -> &mut BTreeMap<String, Vec<u8>> {
        &mut self.extras
    }

    /// Solve the sketches and rebuild the model (incrementally, through the
    /// cache). Cheap when nothing changed.
    pub fn regenerate(&mut self) -> &Regeneration<Body, KernelError> {
        if self.regen.is_none() {
            let regen = self.cache.regenerate_solved(&mut self.doc, &mut self.backend, &mut self.solver);
            self.regen = Some(regen);
        }
        self.regen.as_ref().expect("just regenerated")
    }

    /// Apply a batch of edits as one undo entry (all-or-nothing, see
    /// [`Document::apply_edits`]) and rebuild.
    pub fn apply(&mut self, edits: &[DocEdit]) -> Result<Vec<EditOutcome>, BatchError> {
        let before = self.doc.clone();
        let outcomes = self.doc.apply_edits(edits)?;
        self.push_undo(before);
        self.regen = None;
        Ok(outcomes)
    }

    /// Replace the whole document (one undo entry).
    pub fn replace_document(&mut self, doc: Document) {
        let before = std::mem::replace(&mut self.doc, doc);
        self.push_undo(before);
        self.regen = None;
    }

    fn push_undo(&mut self, before: Document) {
        if self.undo.len() >= UNDO_LIMIT {
            self.undo.remove(0);
        }
        self.undo.push(before);
        self.redo.clear();
    }

    pub fn undo(&mut self) -> bool {
        match self.undo.pop() {
            Some(prev) => {
                self.redo.push(std::mem::replace(&mut self.doc, prev));
                self.regen = None;
                true
            }
            None => false,
        }
    }

    pub fn redo(&mut self) -> bool {
        match self.redo.pop() {
            Some(next) => {
                self.undo.push(std::mem::replace(&mut self.doc, next));
                self.regen = None;
                true
            }
            None => false,
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// The body a feature produced in the last rebuild, if it built.
    pub fn body(&mut self, id: FeatureId) -> Option<&Body> {
        self.regenerate().body(id)
    }

    /// The visible bodies (those no later feature consumed), with the feature
    /// each came from, in history order.
    pub fn visible_bodies(&mut self) -> Vec<(FeatureId, &Body)> {
        let regen = self.regenerate();
        regen
            .visible()
            .iter()
            .filter_map(|&id| regen.body(id).map(|b| (id, b)))
            .collect()
    }

    /// Everything a caller needs to know about the model, serializable.
    pub fn report(&mut self) -> Report {
        self.regenerate();
        build_report(&self.doc, self.regen.as_ref().expect("regenerated"))
    }

    /// Per-face and per-edge detail of one feature's body.
    pub fn body_detail(&mut self, id: FeatureId) -> Result<BodyDetail, SessionError> {
        let body = self.body(id).ok_or(SessionError::UnknownFeature(id))?;
        body_detail(id, &body.solid)
    }

    /// Combine the visible bodies into one compound.
    pub fn combined_solid(&mut self) -> Result<Solid, SessionError> {
        let bodies = self.visible_bodies();
        let mut iter = bodies.iter();
        let first = iter.next().ok_or(SessionError::NoBodies)?.1.solid.clone();
        Ok(iter.try_fold(first, |acc, (_, b)| acc.compound(&b.solid))?)
    }

    /// Write the visible model to `path` as a binary STL.
    pub fn export_stl(&mut self, path: impl AsRef<Path>, deflection: f64) -> Result<(), SessionError> {
        let solid = self.combined_solid()?;
        solid.write_stl(&path.as_ref().to_string_lossy(), deflection)?;
        Ok(())
    }
}

/// A human-readable message for a regeneration error, naming the features
/// involved — the text the app's Problems panel and the CLI both show.
pub fn error_message(err: &RegenError<KernelError>, doc: &Document) -> String {
    let name = |id: FeatureId| doc.history.get(id).map(|f| f.name.as_str()).unwrap_or("?").to_string();
    match err {
        RegenError::Backend { source, .. } => source.to_string(),
        RegenError::MissingInput { input, reason, .. } => {
            let n = name(*input);
            match reason {
                MissingReason::Failed => format!("needs '{n}', which failed"),
                MissingReason::Suppressed => format!("needs '{n}' (suppressed)"),
                MissingReason::RolledBack => format!("needs '{n}' (rolled back)"),
                MissingReason::Later => format!("needs '{n}', which comes later in history"),
                MissingReason::Deleted => "needs a step that was deleted".to_string(),
            }
        }
        RegenError::Invalid { reason, .. } => reason.to_string(),
        RegenError::UnsolvedSketch { status, .. } => match status {
            SolveStatus::Contradictory { .. } => {
                "sketch constraints contradict each other — remove one".to_string()
            }
            SolveStatus::Stalled { .. } => {
                "sketch constraints can't all be satisfied — check the dimensions".to_string()
            }
            SolveStatus::Converged => "sketch solved".to_string(),
        },
    }
}

/// A short, agent-friendly one-line summary of a feature kind (the full
/// definition is in the report's `kind`).
pub fn kind_summary(kind: &FeatureKind) -> String {
    match kind {
        FeatureKind::Box { size } => format!("Box {}×{}×{}", size.x, size.y, size.z),
        FeatureKind::Cylinder { radius, height } => format!("Cylinder r={radius} h={height}"),
        FeatureKind::Sphere { radius } => format!("Sphere r={radius}"),
        FeatureKind::Sketch { .. } | FeatureKind::ConstraintSketch { .. } => "Sketch".into(),
        FeatureKind::Extrude { source, distance } => format!("Extrude #{} by {distance}", source.0),
        FeatureKind::Translate { source, offset } => {
            format!("Move #{} by ({}, {}, {})", source.0, offset.x, offset.y, offset.z)
        }
        FeatureKind::Boolean { op, target, tool } => format!("{op:?} #{} with #{}", target.0, tool.0),
        FeatureKind::FilletAll { source, radius } => format!("Fillet all edges of #{} r={radius}", source.0),
        FeatureKind::Fillet { source, edges, radius } => {
            format!("Fillet {} edge(s) of #{} r={radius}", edges.len(), source.0)
        }
        FeatureKind::PushPull { source, distance, .. } => format!("Push/pull a face of #{} by {distance}", source.0),
        FeatureKind::Rotate { source, angle, .. } => format!("Rotate #{} by {:.1}°", source.0, angle.to_degrees()),
        FeatureKind::Scale { source, factors, .. } => {
            format!("Scale #{} by ({}, {}, {})", source.0, factors.x, factors.y, factors.z)
        }
        FeatureKind::Revolve { source, angle, .. } => format!("Revolve #{} by {:.1}°", source.0, angle.to_degrees()),
        FeatureKind::Mirror { source, .. } => format!("Mirror #{}", source.0),
        FeatureKind::Group { members } => {
            format!("Group of {}", members.iter().map(|m| format!("#{}", m.0)).collect::<Vec<_>>().join(", "))
        }
        FeatureKind::Sweep { profile, path: cdt_core::SweepPath::Planar { .. } } => {
            format!("Sweep #{} along a path", profile.0)
        }
        FeatureKind::Sweep { profile, path: cdt_core::SweepPath::Helix { helix } } => {
            format!("Sweep #{} along a helix, {:.1} turns", profile.0, helix.turns())
        }
        FeatureKind::Thread { source, helix, internal, starts, .. } => format!(
            "{} thread on #{} pitch {} × {:.1} turns{}",
            if *internal { "Internal" } else { "External" },
            source.0,
            helix.pitch,
            helix.turns(),
            if *starts > 1 { format!(", {starts} starts") } else { String::new() }
        ),
        FeatureKind::Pattern { source, layout, count, skip } => {
            let skipped = if skip.is_empty() { String::new() } else { format!(" ({} skipped)", skip.len()) };
            match layout {
                cdt_core::PatternLayout::Linear { step } => {
                    format!("Pattern #{} ×{count} every ({}, {}, {}){skipped}", source.0, step.x, step.y, step.z)
                }
                cdt_core::PatternLayout::Circular { angle, .. } => {
                    format!("Pattern #{} ×{count} every {:.1}°{skipped}", source.0, angle.to_degrees())
                }
                cdt_core::PatternLayout::Helical { angle, rise, .. } => {
                    format!("Pattern #{} ×{count} every {:.1}° and {rise} along the axis{skipped}", source.0, angle.to_degrees())
                }
            }
        }
    }
}

/// Serialized shape of an edit batch as the CLI and socket accept it.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EditBatch {
    /// Shown in the app's history as one undoable step.
    #[serde(default)]
    pub label: Option<String>,
    pub edits: Vec<DocEdit>,
}
