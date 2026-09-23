//! The control link: how a running Cadette window is driven from outside —
//! by `cdt live ...`, an MCP server, or any process that can open a socket.
//!
//! **Transport.** A Unix domain socket per running instance (named pipes on
//! Windows are not wired yet), newline-delimited JSON: one [`LinkRequest`]
//! per line in, one [`LinkResponse`] per line out. The app handles requests
//! on its UI thread between frames, so an edit from outside is an ordinary
//! undo step and the viewport updates live.
//!
//! **Discovery.** Each instance writes an [`Instance`] record into the
//! registry directory (`<config dir>/Cadette/instances/<pid>.json`) naming
//! its socket and the file it has open, and removes it on exit. A client
//! lists the registry, drops entries whose socket no longer answers, and
//! picks the instance with the wanted file open (or the only/most recent
//! one).

use std::path::{Path, PathBuf};

use cdt_core::FeatureId;
use serde::{Deserialize, Serialize};

use crate::EditBatch;

/// A request to a running instance.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum LinkRequest {
    /// Liveness + which file is open.
    Ping,
    /// The session report (+ face/edge detail for these bodies).
    Info {
        #[serde(default)]
        detail: Vec<FeatureId>,
    },
    /// What the user has selected, the visible bodies, the camera, and
    /// whether the app is mid-interaction (edits are refused then).
    State,
    /// Apply an edit batch as one undo step. Refused while the user is
    /// dragging or sketching. `save` writes the file afterwards (only if the
    /// project is bound to one).
    Apply {
        #[serde(flatten)]
        batch: EditBatch,
        #[serde(default)]
        save: bool,
    },
    Undo,
    Redo,
    /// Save to the project's file (error if it has none).
    Save,
    /// Render what the user sees — current camera, theme and selection — to
    /// a PNG at `path`, model only (no chrome).
    Render {
        path: PathBuf,
        #[serde(default = "default_width")]
        width: u32,
        #[serde(default = "default_height")]
        height: u32,
    },
}

fn default_width() -> u32 {
    1280
}
fn default_height() -> u32 {
    800
}

/// The reply: `{"ok": true, "data": ...}` or `{"ok": false, "error": "..."}`.
/// The payload is nested, not flattened, so a report's own `ok` field (did
/// the model build?) never collides with the link's (was the request
/// handled?).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinkResponse {
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub data: serde_json::Value,
}

impl LinkResponse {
    pub fn ok(data: serde_json::Value) -> Self {
        Self { ok: true, error: None, data }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self { ok: false, error: Some(message.into()), data: serde_json::Value::Null }
    }
}

/// A running instance, as recorded in the registry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Instance {
    pub pid: u32,
    pub socket: PathBuf,
    /// The project file the window has open, if it is bound to one.
    #[serde(default)]
    pub file: Option<PathBuf>,
    /// Seconds since the Unix epoch when the record was written.
    pub updated: u64,
}

/// The per-user config directory Cadette uses (`~/.config/Cadette`,
/// `~/Library/Application Support/Cadette`, `%APPDATA%\Cadette`).
pub fn config_dir() -> Option<PathBuf> {
    let base = if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
    } else if cfg!(windows) {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
    };
    Some(base?.join("Cadette"))
}

/// Where instance records live.
pub fn registry_dir() -> Option<PathBuf> {
    config_dir().map(|d| d.join("instances"))
}

/// The socket path an instance with this pid binds.
pub fn socket_path(pid: u32) -> PathBuf {
    std::env::temp_dir().join(format!("cadette-{pid}.sock"))
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Instance {
    pub fn record_path(pid: u32) -> Option<PathBuf> {
        registry_dir().map(|d| d.join(format!("{pid}.json")))
    }

    /// Write (or refresh) this instance's record.
    pub fn announce(pid: u32, socket: &Path, file: Option<&Path>) -> std::io::Result<()> {
        let Some(path) = Self::record_path(pid) else {
            return Ok(());
        };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let record = Instance {
            pid,
            socket: socket.to_path_buf(),
            file: file.map(|f| std::fs::canonicalize(f).unwrap_or_else(|_| f.to_path_buf())),
            updated: now(),
        };
        std::fs::write(path, serde_json::to_vec_pretty(&record)?)
    }

    /// Remove this instance's record (on exit).
    pub fn withdraw(pid: u32) {
        if let Some(path) = Self::record_path(pid) {
            let _ = std::fs::remove_file(path);
        }
    }

    /// Every recorded instance whose socket still answers a ping; stale
    /// records are removed. Most recently updated first.
    pub fn live() -> Vec<Instance> {
        let Some(dir) = registry_dir() else {
            return Vec::new();
        };
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(text) = std::fs::read(&path) else { continue };
            let Ok(record) = serde_json::from_slice::<Instance>(&text) else {
                let _ = std::fs::remove_file(&path);
                continue;
            };
            match send(&record.socket, &LinkRequest::Ping) {
                Ok(resp) if resp.ok => out.push(record),
                _ => {
                    // The window is gone (a killed process never ran its
                    // cleanup): drop the record and its socket file.
                    let _ = std::fs::remove_file(&path);
                    let _ = std::fs::remove_file(&record.socket);
                }
            }
        }
        out.sort_by(|a, b| b.updated.cmp(&a.updated));
        out
    }

    /// The live instance with `file` open, if any.
    pub fn with_file(file: &Path) -> Option<Instance> {
        let wanted = std::fs::canonicalize(file).unwrap_or_else(|_| file.to_path_buf());
        Self::live().into_iter().find(|i| i.file.as_deref() == Some(wanted.as_path()))
    }
}

/// Send one request to an instance's socket and read its reply.
#[cfg(unix)]
pub fn send(socket: &Path, request: &LinkRequest) -> std::io::Result<LinkResponse> {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;
    let mut stream = UnixStream::connect(socket)?;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(60)))?;
    stream.set_write_timeout(Some(std::time::Duration::from_secs(10)))?;
    let mut line = serde_json::to_string(request)?;
    line.push('\n');
    stream.write_all(line.as_bytes())?;
    stream.flush()?;
    let mut reader = BufReader::new(stream);
    let mut reply = String::new();
    reader.read_line(&mut reply)?;
    if reply.trim().is_empty() {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "the instance closed the link without replying"));
    }
    Ok(serde_json::from_str(reply.trim())?)
}

#[cfg(not(unix))]
pub fn send(_socket: &Path, _request: &LinkRequest) -> std::io::Result<LinkResponse> {
    Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "the control link is only available on Unix so far"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_and_responses_round_trip_as_tagged_json() {
        let req = LinkRequest::Apply { batch: EditBatch { label: Some("x".into()), edits: vec![] }, save: true };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains(r#""cmd":"apply""#) && json.contains(r#""save":true"#), "{json}");
        let back: LinkRequest = serde_json::from_str(r#"{"cmd":"render","path":"/tmp/x.png"}"#).unwrap();
        assert!(matches!(back, LinkRequest::Render { width: 1280, height: 800, .. }));
        let resp = LinkResponse::ok(serde_json::json!({ "ok": false, "file": "a.cdt" }));
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains(r#""ok":true,"data":{"#) && json.contains(r#""file":"a.cdt""#) && !json.contains("error"), "{json}");
        let back: LinkResponse = serde_json::from_str(&json).unwrap();
        assert!(back.ok && back.data["ok"] == false, "the payload's own ok survives");
        let err: LinkResponse = serde_json::from_str(r#"{"ok":false,"error":"busy"}"#).unwrap();
        assert_eq!(err.error.as_deref(), Some("busy"));
    }
}
