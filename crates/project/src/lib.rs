//! # cdt-project
//!
//! The `.cdt` project file format, usable by anything that can link
//! `cdt-core`: the app, a CLI, a test, an agent. A project is a zip archive:
//!
//! - `document.json` — the parametric [`Document`], wrapped in a versioned
//!   envelope `{ "format_version": N, "document": { ... } }`.
//! - `assets/<name>` — the files the history replays (an `Import` feature's
//!   STEP), loaded into [`Document::assets`] and written back from it. Only
//!   assets some feature still refers to are written, so a deleted import's
//!   file does not ride along forever.
//! - any other entries — opaque *extras* that travel with the project but are
//!   not part of the model (the app stores its camera pose and grid settings
//!   in `view.json`, so orbiting never dirties the document). Readers ignore
//!   extras they don't know; writers preserve the ones they were given.
//!
//! **Compatibility.** `document.json` is the serde form of [`Document`] with
//! externally-tagged `FeatureKind` variants — readable and hand-editable. New
//! fields are added with `#[serde(default)]` so older files keep loading
//! without a version bump (see the policy on [`cdt_core::FeatureKind`]); the
//! envelope version only changes for a breaking reshape, and [`read`] then
//! routes the old document through [`migrate`] before deserializing it.
//!
//! **Loading is tolerant, then strict.** A file may omit reconstructible
//! fields (the id allocator, the rollback bar); [`Document::sanitize`] repairs
//! what it can. But a history whose references don't hold — a duplicate id, a
//! forward reference, a missing input — is refused with
//! [`ProjectError::InvalidHistory`] rather than loaded to fail later as a
//! confusing regeneration error.

use std::collections::BTreeMap;
use std::io::{Read, Seek, Write};
use std::path::Path;

use cdt_core::{DependencyError, Document};
use serde::{Deserialize, Serialize};

/// The envelope version this build writes.
pub const FORMAT_VERSION: u64 = 1;
/// The oldest envelope version this build can still read (via [`migrate`]).
pub const MIN_SUPPORTED_VERSION: u64 = 1;
/// The archive entry holding the document.
pub const DOCUMENT_ENTRY: &str = "document.json";
/// The archive directory holding the document's assets (see [`Document::assets`]).
pub const ASSET_DIR: &str = "assets/";

/// Why a project could not be read or written.
#[derive(Debug, thiserror::Error)]
pub enum ProjectError {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("not a Cadette project: {0}")]
    Archive(#[from] zip::result::ZipError),
    #[error("not a Cadette project: no {DOCUMENT_ENTRY} entry")]
    NoDocument,
    #[error("document.json is not valid: {0}")]
    Json(#[from] serde_json::Error),
    #[error("project format version {found} is newer than this build supports ({supported})")]
    TooNew { found: u64, supported: u64 },
    #[error("project format version {found} is older than this build can read ({oldest})")]
    TooOld { found: u64, oldest: u64 },
    #[error("the history is not valid: {}", .0.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; "))]
    InvalidHistory(Vec<DependencyError>),
}

/// A project as read from, or about to be written to, a `.cdt` archive.
#[derive(Clone, Debug, Default)]
pub struct ProjectFile {
    pub document: Document,
    /// Every archive entry other than the document and its assets, by name
    /// (e.g. `view.json`). Names under `assets/` are reserved for
    /// [`Document::assets`] and are not written.
    pub extras: BTreeMap<String, Vec<u8>>,
}

impl ProjectFile {
    pub fn new(document: Document) -> Self {
        Self { document, extras: BTreeMap::new() }
    }

    /// Attach (or replace) an extra entry serialized as pretty JSON.
    pub fn set_json_extra<T: Serialize>(&mut self, name: &str, value: &T) -> Result<(), ProjectError> {
        self.extras.insert(name.to_string(), serde_json::to_vec_pretty(value)?);
        Ok(())
    }

    /// Parse an extra entry as JSON, leniently: an absent or malformed entry
    /// is `None` (extras must never fail a load).
    pub fn json_extra<T: for<'de> Deserialize<'de>>(&self, name: &str) -> Option<T> {
        self.extras.get(name).and_then(|bytes| serde_json::from_slice(bytes).ok())
    }
}

/// The `document.json` envelope.
#[derive(Serialize, Deserialize)]
struct Envelope {
    format_version: u64,
    document: serde_json::Value,
}

/// Serialize a document into the `document.json` envelope (pretty-printed).
pub fn document_to_json(document: &Document) -> Result<Vec<u8>, ProjectError> {
    let envelope = Envelope {
        format_version: FORMAT_VERSION,
        document: serde_json::to_value(document)?,
    };
    Ok(serde_json::to_vec_pretty(&envelope)?)
}

/// Parse a `document.json` envelope: check the version, migrate if needed,
/// deserialize, repair what is safely repairable, and refuse an invalid
/// history.
pub fn document_from_json(bytes: &[u8]) -> Result<Document, ProjectError> {
    let envelope: Envelope = serde_json::from_slice(bytes)?;
    let version = envelope.format_version;
    if version > FORMAT_VERSION {
        return Err(ProjectError::TooNew { found: version, supported: FORMAT_VERSION });
    }
    if version < MIN_SUPPORTED_VERSION {
        return Err(ProjectError::TooOld { found: version, oldest: MIN_SUPPORTED_VERSION });
    }
    let value = migrate(envelope.document, version);
    let mut document: Document = serde_json::from_value(value)?;
    // Defensive: repair invariants a corrupt/hand-edited file may violate (id
    // allocator behind existing ids, rollback bar out of range). Repair is
    // safe, so a repaired document just loads.
    document.sanitize();
    document.history.validate().map_err(ProjectError::InvalidHistory)?;
    Ok(document)
}

/// Bring a document written at envelope version `from` up to
/// [`FORMAT_VERSION`], one step at a time. Every version so far reads as-is;
/// when a breaking reshape lands, its upgrade goes here (`from == old` →
/// rewrite the JSON) and `MIN_SUPPORTED_VERSION` stays where it is, so old
/// files keep opening.
pub fn migrate(document: serde_json::Value, from: u64) -> serde_json::Value {
    match from {
        _ => document,
    }
}

/// Read a project from any seekable byte source (a file, a `Cursor`).
pub fn read<R: Read + Seek>(reader: R) -> Result<ProjectFile, ProjectError> {
    let mut archive = zip::ZipArchive::new(reader)?;
    let mut document = None;
    let mut extras = BTreeMap::new();
    let mut assets = Vec::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        if entry.is_dir() {
            continue;
        }
        let mut bytes = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut bytes)?;
        if entry.name() == DOCUMENT_ENTRY {
            document = Some(document_from_json(&bytes)?);
        } else if let Some(name) = entry.name().strip_prefix(ASSET_DIR) {
            // Only bare file names: a hand-made `assets/../x` is dropped,
            // and the feature that names it reports its file missing.
            if cdt_core::Assets::is_clean_name(name) {
                assets.push((name.to_string(), bytes));
            }
        } else {
            extras.insert(entry.name().to_string(), bytes);
        }
    }
    let mut document = document.ok_or(ProjectError::NoDocument)?;
    for (name, bytes) in assets {
        document.assets.set(&name, bytes);
    }
    Ok(ProjectFile { document, extras })
}

/// Read a project from a `.cdt` file.
pub fn read_path(path: impl AsRef<Path>) -> Result<ProjectFile, ProjectError> {
    read(std::fs::File::open(path)?)
}

/// Write a project to any seekable sink.
pub fn write<W: Write + Seek>(writer: W, project: &ProjectFile) -> Result<(), ProjectError> {
    let mut zip = zip::ZipWriter::new(writer);
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    zip.start_file(DOCUMENT_ENTRY, opts)?;
    zip.write_all(&document_to_json(&project.document)?)?;
    for (name, bytes) in &project.document.assets {
        if project.document.references_asset(name) && cdt_core::Assets::is_clean_name(name) {
            zip.start_file(format!("{ASSET_DIR}{name}"), opts)?;
            zip.write_all(bytes)?;
        }
    }
    for (name, bytes) in &project.extras {
        if name.starts_with(ASSET_DIR) {
            continue; // reserved for the document's assets
        }
        zip.start_file(name, opts)?;
        zip.write_all(bytes)?;
    }
    zip.finish()?;
    Ok(())
}

/// Write a project to a `.cdt` file (created or truncated).
pub fn write_path(path: impl AsRef<Path>, project: &ProjectFile) -> Result<(), ProjectError> {
    write(std::fs::File::create(path)?, project)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cdt_core::{FeatureKind, DVec3};
    use std::io::Cursor;

    fn sample() -> Document {
        let mut doc = Document::new("part");
        let b = doc.add("Box", FeatureKind::Box { size: DVec3::new(10.0, 20.0, 30.0) });
        doc.add("Round", FeatureKind::FilletAll { source: b, radius: 2.0 });
        doc
    }

    #[test]
    fn assets_ride_in_the_archive_while_a_feature_still_uses_them() {
        let mut doc = sample();
        let kept = doc.assets.insert("servo.step", b"ISO-10303-21;".to_vec());
        let orphan = doc.assets.insert("old.step", b"ISO-10303-21; old".to_vec());
        doc.add("servo", FeatureKind::Import { asset: kept.clone() });
        let mut buf = Cursor::new(Vec::new());
        write(&mut buf, &ProjectFile::new(doc)).unwrap();

        // The archive holds exactly the referenced asset, under assets/.
        let mut archive = zip::ZipArchive::new(Cursor::new(buf.get_ref().clone())).unwrap();
        let names: Vec<String> = (0..archive.len()).map(|i| archive.by_index(i).unwrap().name().to_string()).collect();
        assert!(names.contains(&format!("assets/{kept}")), "{names:?}");
        assert!(!names.iter().any(|n| n.contains(&orphan)), "{names:?}");

        let back = read(Cursor::new(buf.into_inner())).unwrap();
        assert_eq!(back.document.assets.get(&kept).map(|b| b.as_ref()), Some(&b"ISO-10303-21;"[..]));
        assert!(back.document.assets.get(&orphan).is_none());
        assert!(back.extras.is_empty(), "assets are not extras: {:?}", back.extras.keys());
    }

    #[test]
    fn asset_entries_with_a_directory_part_are_dropped_and_extras_never_shadow_assets() {
        // A hand-made archive: good and bad asset names, an extra under assets/.
        let mut buf = Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut buf);
            let opts = zip::write::SimpleFileOptions::default();
            let mut doc = sample();
            doc.add("ok", FeatureKind::Import { asset: "ok.step".into() });
            doc.add("evil", FeatureKind::Import { asset: "../evil.step".into() });
            zip.start_file(DOCUMENT_ENTRY, opts).unwrap();
            zip.write_all(&document_to_json(&doc).unwrap()).unwrap();
            for name in ["assets/ok.step", "assets/../evil.step", "assets/sub/x.step", "assets//abs.step"] {
                zip.start_file(name, opts).unwrap();
                zip.write_all(b"ISO-10303-21;").unwrap();
            }
            zip.finish().unwrap();
        }
        let back = read(Cursor::new(buf.into_inner())).unwrap();
        let names: Vec<&str> = back.document.assets.names().collect();
        assert_eq!(names, ["ok.step"]);
        assert!(back.extras.is_empty());

        // Writing: an extra under assets/ is skipped rather than colliding.
        let mut project = back;
        project.extras.insert("assets/ok.step".into(), b"shadow".to_vec());
        project.extras.insert("view.json".into(), b"{}".to_vec());
        let mut buf = Cursor::new(Vec::new());
        write(&mut buf, &project).unwrap();
        let again = read(Cursor::new(buf.into_inner())).unwrap();
        assert_eq!(again.document.assets.get("ok.step").map(|b| b.as_ref()), Some(&b"ISO-10303-21;"[..]));
        assert_eq!(again.extras.keys().collect::<Vec<_>>(), ["view.json"]);
    }

    #[test]
    fn round_trips_document_and_extras_through_a_zip() {
        let mut project = ProjectFile::new(sample());
        project.set_json_extra("view.json", &serde_json::json!({ "yaw": 1.5 })).unwrap();
        project.extras.insert("resources/note.txt".into(), b"hello".to_vec());
        let mut buf = Cursor::new(Vec::new());
        write(&mut buf, &project).unwrap();
        buf.set_position(0);
        let back = read(buf).unwrap();
        assert_eq!(back.document.history.features(), project.document.history.features());
        assert_eq!(back.document.rollback(), 2);
        assert_eq!(back.extras.len(), 2);
        let view: serde_json::Value = back.json_extra("view.json").unwrap();
        assert_eq!(view["yaw"], 1.5);
        assert_eq!(back.extras["resources/note.txt"], b"hello");
        assert!(back.json_extra::<serde_json::Value>("missing.json").is_none());
    }

    #[test]
    fn version_checks() {
        let doc = sample();
        let bytes = serde_json::to_vec(&Envelope {
            format_version: FORMAT_VERSION + 1,
            document: serde_json::to_value(&doc).unwrap(),
        })
        .unwrap();
        assert!(matches!(document_from_json(&bytes), Err(ProjectError::TooNew { .. })));
        let bytes = serde_json::to_vec(&Envelope {
            format_version: 0,
            document: serde_json::to_value(&doc).unwrap(),
        })
        .unwrap();
        assert!(matches!(document_from_json(&bytes), Err(ProjectError::TooOld { .. })));
    }

    #[test]
    fn a_forward_reference_is_refused_at_load() {
        let doc = sample();
        let mut value = serde_json::to_value(&doc).unwrap();
        let features = value["history"]["features"].as_array_mut().unwrap();
        features.swap(0, 1); // the fillet now precedes the box it rounds
        let bytes = serde_json::to_vec(&Envelope { format_version: FORMAT_VERSION, document: value }).unwrap();
        assert!(matches!(document_from_json(&bytes), Err(ProjectError::InvalidHistory(_))));
    }

    #[test]
    fn an_archive_without_a_document_is_not_a_project() {
        let mut buf = Cursor::new(Vec::new());
        let mut zip = zip::ZipWriter::new(&mut buf);
        zip.start_file("readme.txt", zip::write::SimpleFileOptions::default()).unwrap();
        zip.write_all(b"nope").unwrap();
        zip.finish().unwrap();
        buf.set_position(0);
        assert!(matches!(read(buf), Err(ProjectError::NoDocument)));
        assert!(matches!(read(Cursor::new(b"not a zip".to_vec())), Err(ProjectError::Archive(_))));
    }
}
