//! Project and STL file I/O: saving/opening `.cdt` project archives and
//! exporting the visible model as STL, including the `rfd` file dialogs.

use cdt_core::{Document, RegenCache};
use cdt_kernel::Solid;
use cdt_project::ProjectFile;
use cdt_render::CameraPose;

use crate::Modeler;

/// Finer mesh tolerance for STL export than for the on-screen preview.
const EXPORT_DEFLECTION_MM: f64 = 0.05;
/// The archive entry holding the viewport state (see [`ViewFile`]).
const VIEW_ENTRY: &str = "view.json";

/// The `view.json` zip entry: viewport state that travels with the project —
/// camera pose + grid settings. Deliberately SEPARATE from `document.json` so
/// the model document (and its dirty/undo tracking) stays pure: orbiting the
/// camera must never mark the file unsaved. It rides in the archive as one of
/// `cdt-project`'s opaque extras: read leniently — an absent or malformed
/// entry never fails a load (pre-view files have none), and other readers
/// simply ignore it.
#[derive(serde::Serialize, serde::Deserialize)]
struct ViewFile {
    #[serde(default)]
    camera: Option<CameraJson>,
    #[serde(default)]
    grid: Option<GridJson>,
}

/// Serialized form of [`CameraPose`] (kept here so the render crate stays
/// serde-free).
#[derive(serde::Serialize, serde::Deserialize)]
struct CameraJson {
    target: [f32; 3],
    yaw: f32,
    pitch: f32,
    distance: f32,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct GridJson {
    spacing: f64,
    extent: f64,
    show_xy: bool,
    show_xz: bool,
    show_yz: bool,
    snap: bool,
}

impl Modeler {
    /// Combine the visible bodies into one shape (a compound) for export.
    fn combined_body(&self) -> Result<Solid, String> {
        let mut bodies = self.bodies.iter();
        let first = bodies.next().ok_or("nothing to export")?.solid.clone();
        bodies
            .try_fold(first, |acc, b| acc.compound(&b.solid))
            .map_err(|e| e.to_string())
    }

    /// Write the visible model to `path` as a binary STL.
    pub(crate) fn export_stl_to(&self, path: &str) -> Result<(), String> {
        self.combined_body()?
            .write_stl(path, EXPORT_DEFLECTION_MM)
            .map_err(|e| e.to_string())
    }

    /// Prompt for a path and export the model as STL.
    pub(crate) fn export_stl(&mut self) {
        if self.bodies.is_empty() {
            self.status = Some("Nothing to export".into());
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .add_filter("STL", &["stl"])
            .set_file_name(format!("{}.stl", self.doc.name))
            .save_file()
        else {
            return; // user cancelled
        };
        let path = path.to_string_lossy().to_string();
        self.status = Some(match self.export_stl_to(&path) {
            Ok(()) => format!("Exported {path}"),
            Err(e) => format!("Export failed: {e}"),
        });
    }

    /// Write the project (the parametric document) to `path` as a `.cdt` zip.
    pub(crate) fn save_project_to(&self, path: &str) -> Result<(), String> {
        let mut project = ProjectFile::new(self.doc.clone());
        // The viewport state rides along in its own entry (see [`ViewFile`]).
        let view = ViewFile {
            camera: self.camera_pose.map(|p| CameraJson {
                target: p.target,
                yaw: p.yaw,
                pitch: p.pitch,
                distance: p.distance,
            }),
            grid: Some(GridJson {
                spacing: self.ui.grid_spacing,
                extent: self.ui.grid_extent,
                show_xy: self.ui.show_grid,
                show_xz: self.ui.show_xz,
                show_yz: self.ui.show_yz,
                snap: self.ui.snap_to_grid,
            }),
        };
        project.set_json_extra(VIEW_ENTRY, &view).map_err(|e| e.to_string())?;
        cdt_project::write_path(path, &project).map_err(|e| e.to_string())
    }

    /// Load a `.cdt` project from `path`, replacing the current document and
    /// resetting transient state. Returns the loaded document.
    pub(crate) fn load_project_from(&mut self, path: &str) -> Result<(), String> {
        let project = cdt_project::read_path(path).map_err(|e| e.to_string())?;
        // Viewport state, if the file carries it (lenient: never fails a load).
        let view: Option<ViewFile> = project.json_extra(VIEW_ENTRY);

        // Swap in the new document and drop all state tied to the old one.
        self.reset_to(project.document);
        if let Some(view) = view {
            if let Some(g) = view.grid {
                // Take only sane finite values (matching the panel's ranges) so
                // a hand-edited file can't wedge the grid controls.
                if g.spacing.is_finite() && (0.05..=1000.0).contains(&g.spacing) {
                    self.ui.grid_spacing = g.spacing;
                }
                if g.extent.is_finite() && (10.0..=5000.0).contains(&g.extent) {
                    self.ui.grid_extent = g.extent;
                }
                self.ui.show_grid = g.show_xy;
                self.ui.show_xz = g.show_xz;
                self.ui.show_yz = g.show_yz;
                self.ui.snap_to_grid = g.snap;
            }
            // Hand the saved camera to the viewport (applied next frame;
            // set_pose clamps, so a hand-edited pose can't go degenerate).
            self.camera_request = view.camera.map(|c| CameraPose {
                target: c.target,
                yaw: c.yaw,
                pitch: c.pitch,
                distance: c.distance,
            });
        }
        Ok(())
    }

    /// Replace the document and drop every piece of transient state tied to the
    /// old one (regen cache, undo/redo, selection, in-flight drags/sketch). Used
    /// by both Open and New so they reset identically.
    fn reset_to(&mut self, document: Document) {
        self.doc = document;
        self.regen_cache = RegenCache::new();
        self.undo.clear();
        self.redo.clear();
        self.selection.clear();
        self.edge_anchors.clear();
        self.transform = None;
        self.manipulation = None;
        self.resize = None;
        self.sketch_session = None;
        self.ui.selected = None;
    }

    /// If the document has unsaved changes, ask whether to save them first.
    /// Returns `true` if it's safe to proceed with a discarding action (New /
    /// Open / close): the doc was clean, was saved just now, or the user chose to
    /// discard. Returns `false` if the user cancelled (or a prompted save was
    /// itself cancelled) — the caller should abort.
    pub(crate) fn confirm_discard(&mut self) -> bool {
        if !self.dirty {
            return true;
        }
        match rfd::MessageDialog::new()
            .set_title("Unsaved changes")
            .set_description("Save changes to the current project first?")
            .set_buttons(rfd::MessageButtons::YesNoCancel)
            .show()
        {
            // Save; only proceed if the save actually completed (a cancelled
            // save dialog must not silently discard the work).
            rfd::MessageDialogResult::Yes => self.save_project(),
            // Discard the changes and proceed.
            rfd::MessageDialogResult::No => true,
            // Cancel (or the dialog was dismissed): abort the action.
            _ => false,
        }
    }

    /// Reset to a blank project, confirming first if there are unsaved changes.
    /// Returns true if a new document was started (so the host re-meshes).
    pub(crate) fn new_document(&mut self) -> bool {
        if !self.confirm_discard() {
            return false;
        }
        self.reset_to(Document::new("untitled"));
        self.current_path = None;
        self.mark_saved(); // a blank doc is its own clean baseline
        self.status = Some("New project".into());
        true
    }

    /// Prompt for a path and save the project. Returns true if it was written
    /// (false if the user cancelled the dialog or the write failed) — the
    /// unsaved-changes prompt relies on this to know a save completed.
    pub(crate) fn save_project(&mut self) -> bool {
        let default_name = self
            .current_path
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "untitled.cdt".to_string());
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Cadette project", &["cdt"])
            .set_file_name(default_name)
            .save_file()
        else {
            return false; // cancelled
        };
        let path_str = path.to_string_lossy().to_string();
        match self.save_project_to(&path_str) {
            Ok(()) => {
                self.current_path = Some(path);
                self.mark_saved(); // new clean baseline
                self.status = Some(format!("Saved {path_str}"));
                true
            }
            Err(e) => {
                self.status = Some(format!("Save failed: {e}"));
                false
            }
        }
    }

    /// Prompt for a path and open a project. Returns true if a project loaded
    /// (so the host re-meshes). Confirms first if the current doc is unsaved.
    pub(crate) fn open_project(&mut self) -> bool {
        if !self.confirm_discard() {
            return false;
        }
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Cadette project", &["cdt"])
            .pick_file()
        else {
            return false; // cancelled
        };
        let path_str = path.to_string_lossy().to_string();
        match self.load_project_from(&path_str) {
            Ok(()) => {
                self.current_path = Some(path);
                self.mark_saved(); // freshly-opened baseline
                self.status = Some(format!("Opened {path_str}"));
                true
            }
            Err(e) => {
                self.status = Some(format!("Open failed: {e}"));
                false
            }
        }
    }
}
