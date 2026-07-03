//! Project and STL file I/O: saving/opening `.cdt` project archives and
//! exporting the visible model as STL, including the `rfd` file dialogs.

use cdt_core::{Document, RegenCache};
use cdt_kernel::Solid;

use crate::Modeler;

/// Finer mesh tolerance for STL export than for the on-screen preview.
const EXPORT_DEFLECTION_MM: f64 = 0.05;
/// Version stamped into `.cdt` project files (a zip holding `document.json`).
const PROJECT_FORMAT_VERSION: u64 = 1;

impl Modeler {
    /// Combine the visible bodies into one shape (a compound) for export.
    fn combined_body(&self) -> Result<Solid, String> {
        let mut bodies = self.bodies.iter();
        let first = bodies.next().ok_or("nothing to export")?.clone();
        bodies
            .try_fold(first, |acc, b| acc.compound(b))
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
        use std::io::Write;
        let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        let mut zip = zip::ZipWriter::new(file);
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        zip.start_file("document.json", opts).map_err(|e| e.to_string())?;
        // Embedded STL imports (resources/) will join this archive later.
        let wrapped = serde_json::json!({
            "format_version": PROJECT_FORMAT_VERSION,
            "document": &self.doc,
        });
        let bytes = serde_json::to_vec_pretty(&wrapped).map_err(|e| e.to_string())?;
        zip.write_all(&bytes).map_err(|e| e.to_string())?;
        zip.finish().map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Load a `.cdt` project from `path`, replacing the current document and
    /// resetting transient state. Returns the loaded document.
    pub(crate) fn load_project_from(&mut self, path: &str) -> Result<(), String> {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
        let entry = archive
            .by_name("document.json")
            .map_err(|e| format!("not a Cadette project: {e}"))?;
        let value: serde_json::Value =
            serde_json::from_reader(entry).map_err(|e| e.to_string())?;
        let version = value.get("format_version").and_then(|v| v.as_u64()).unwrap_or(0);
        if version != PROJECT_FORMAT_VERSION {
            return Err(format!("unsupported project version {version}"));
        }
        let doc_value = value.get("document").cloned().ok_or("missing document")?;
        let document: Document =
            serde_json::from_value(doc_value).map_err(|e| e.to_string())?;

        // Swap in the new document and drop all state tied to the old one.
        self.doc = document;
        self.regen_cache = RegenCache::new();
        self.undo.clear();
        self.redo.clear();
        self.selection.clear();
        self.edge_anchors.clear();
        self.transform = None;
        self.manipulation = None;
        self.sketch_session = None;
        self.ui.selected = None;
        Ok(())
    }

    /// Prompt for a path and save the project.
    pub(crate) fn save_project(&mut self) {
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
            return; // cancelled
        };
        let path_str = path.to_string_lossy().to_string();
        self.status = Some(match self.save_project_to(&path_str) {
            Ok(()) => {
                self.current_path = Some(path);
                self.mark_saved(); // new clean baseline
                format!("Saved {path_str}")
            }
            Err(e) => format!("Save failed: {e}"),
        });
    }

    /// Prompt for a path and open a project. Returns true if a project loaded
    /// (so the host re-meshes).
    pub(crate) fn open_project(&mut self) -> bool {
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
