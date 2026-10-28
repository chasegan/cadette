//! The app side of the control link (see `cdt_session::link`): a socket the
//! running window listens on so `cdt live ...`, an MCP server, or an AI agent
//! can read and edit the open project. Requests are handled on the UI thread
//! between frames — the kernel isn't thread-safe, and this way an outside
//! edit is an ordinary undo step that the viewport shows immediately.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, SyncSender};

use cdt_core::{DocEdit, FeatureId};
use cdt_render::{Controller, HeadlessRenderer, Pick, RenderOptions};
use cdt_session::link::{socket_path, Instance, LinkRequest, LinkResponse};
use cdt_session::{body_detail, build_report};
use cdt_solver::LmSolver;

use crate::Modeler;

type Pending = (LinkRequest, SyncSender<LinkResponse>);

/// The listening socket plus the queue of requests waiting for the UI thread.
pub(crate) struct Link {
    pid: u32,
    socket: PathBuf,
    rx: Receiver<Pending>,
}

impl Link {
    /// Bind this process's socket and start accepting. `None` (with a log
    /// line) if the socket can't be bound, or on platforms without one yet.
    #[cfg(unix)]
    pub(crate) fn start() -> Option<Link> {
        use std::io::{BufRead, BufReader, Write};
        use std::os::unix::net::UnixListener;

        let pid = std::process::id();
        let socket = socket_path(pid);
        let _ = std::fs::remove_file(&socket);
        let listener = match UnixListener::bind(&socket) {
            Ok(l) => l,
            Err(e) => {
                log::warn!("control link unavailable: could not bind {}: {e}", socket.display());
                return None;
            }
        };
        let (tx, rx) = mpsc::channel::<Pending>();
        std::thread::Builder::new()
            .name("cdt-link".into())
            .spawn(move || {
                for stream in listener.incoming() {
                    let Ok(stream) = stream else { continue };
                    let tx = tx.clone();
                    std::thread::spawn(move || {
                        let mut writer = match stream.try_clone() {
                            Ok(w) => w,
                            Err(_) => return,
                        };
                        let reader = BufReader::new(stream);
                        for line in reader.lines() {
                            let Ok(line) = line else { break };
                            if line.trim().is_empty() {
                                continue;
                            }
                            let response = match serde_json::from_str::<LinkRequest>(&line) {
                                Err(e) => LinkResponse::error(format!("malformed request: {e}")),
                                Ok(request) => {
                                    let wait = match &request {
                                        LinkRequest::Apply { batch, .. }
                                            if batch.edits.iter().any(|e| matches!(e, DocEdit::Import { .. })) =>
                                        {
                                            600
                                        }
                                        _ => 30,
                                    };
                                    let (reply_tx, reply_rx) = mpsc::sync_channel(1);
                                    if tx.send((request, reply_tx)).is_err() {
                                        LinkResponse::error("the window is closing")
                                    } else {
                                        // The UI thread handles it on its next frame; a
                                        // hidden window may not be drawing, so wait a while.
                                        // An import parses its STEP twice (check, then
                                        // regen), which can take minutes on an assembly.
                                        match reply_rx.recv_timeout(std::time::Duration::from_secs(wait)) {
                                            Ok(r) => r,
                                            Err(_) => LinkResponse::error(
                                                "the window did not answer (is it hidden or busy?)",
                                            ),
                                        }
                                    }
                                }
                            };
                            let mut text = serde_json::to_string(&response).unwrap_or_default();
                            text.push('\n');
                            if writer.write_all(text.as_bytes()).is_err() {
                                break;
                            }
                        }
                    });
                }
            })
            .ok()?;
        log::info!("control link listening on {}", socket.display());
        Some(Link { pid, socket, rx })
    }

    #[cfg(not(unix))]
    pub(crate) fn start() -> Option<Link> {
        log::info!("control link: not available on this platform yet");
        None
    }

    /// Record (or refresh) this instance in the registry, naming its file.
    pub(crate) fn announce(&self, file: Option<&Path>) {
        if let Err(e) = Instance::announce(self.pid, &self.socket, file) {
            log::warn!("could not record this instance for `cdt live`: {e}");
        }
    }

    fn pending(&self) -> Vec<Pending> {
        let mut out = Vec::new();
        while let Ok(item) = self.rx.try_recv() {
            out.push(item);
        }
        out
    }
}

impl Drop for Link {
    fn drop(&mut self) {
        Instance::withdraw(self.pid);
        let _ = std::fs::remove_file(&self.socket);
    }
}

impl Modeler {
    /// Refresh the registry record after the bound file changed.
    pub(crate) fn announce_link(&self) {
        if let Some(link) = &self.link {
            link.announce(self.current_path.as_deref());
        }
    }

    /// Handle every request that arrived since the last frame. Returns true
    /// if the document changed (the caller regenerates).
    pub(crate) fn poll_link(&mut self) -> bool {
        let Some(link) = &self.link else {
            return false;
        };
        let pending = link.pending();
        let mut changed = false;
        for (request, reply) in pending {
            let (response, did_change) = self.handle_link(request);
            changed |= did_change;
            let _ = reply.send(response);
        }
        changed
    }

    fn busy(&self) -> bool {
        self.interacting() || self.sketch_session.is_some()
    }

    fn link_file(&self) -> serde_json::Value {
        serde_json::json!({
            "file": self.current_path,
            "dirty": self.dirty,
            "pid": std::process::id(),
            "version": env!("CARGO_PKG_VERSION"),
        })
    }

    /// Rebuild (cheap when cached) and describe the model.
    fn link_report(&mut self, detail: &[FeatureId]) -> Result<serde_json::Value, String> {
        let regen = self.regen_cache.regenerate_solved(&mut self.doc, &mut self.backend, &mut LmSolver);
        let report = build_report(&self.doc, &regen);
        let mut details = Vec::new();
        for &id in detail {
            let body = regen.body(id).ok_or_else(|| format!("no built body for feature {}", id.0))?;
            details.push(body_detail(id, &body.solid).map_err(|e| e.to_string())?);
        }
        let mut value = serde_json::to_value(&report).map_err(|e| e.to_string())?;
        if !details.is_empty() {
            value["details"] = serde_json::to_value(&details).map_err(|e| e.to_string())?;
        }
        Ok(value)
    }

    fn handle_link(&mut self, request: LinkRequest) -> (LinkResponse, bool) {
        match request {
            LinkRequest::Ping => (LinkResponse::ok(self.link_file()), false),
            LinkRequest::Info { detail } => match self.link_report(&detail) {
                Ok(v) => (LinkResponse::ok(v), false),
                Err(e) => (LinkResponse::error(e), false),
            },
            LinkRequest::State => {
                let picks: Vec<serde_json::Value> = self
                    .selection
                    .selected()
                    .iter()
                    .filter_map(|p| match *p {
                        Pick::Face(g) => self.locate_face(g).map(|(i, local)| {
                            serde_json::json!({ "kind": "face", "body": self.ui.visible.get(i), "index": local, "global": g })
                        }),
                        Pick::Edge(g) => self.locate_edge(g).map(|(i, local)| {
                            serde_json::json!({ "kind": "edge", "body": self.ui.visible.get(i), "index": local, "global": g })
                        }),
                    })
                    .collect();
                let mut v = self.link_file();
                v["busy"] = self.busy().into();
                v["sketching"] = self.sketch_session.is_some().into();
                v["selected_feature"] = serde_json::to_value(self.ui.selected).unwrap_or_default();
                v["picks"] = picks.into();
                v["visible"] = serde_json::to_value(&self.ui.visible).unwrap_or_default();
                v["errors"] = serde_json::to_value(&self.ui.errors).unwrap_or_default();
                v["camera"] = self
                    .camera_pose
                    .map(|c| serde_json::json!({ "target": c.target, "yaw": c.yaw, "pitch": c.pitch, "distance": c.distance }))
                    .unwrap_or(serde_json::Value::Null);
                (LinkResponse::ok(v), false)
            }
            LinkRequest::Apply { batch, save } => {
                if self.busy() {
                    return (LinkResponse::error("the user is mid-edit (dragging or sketching); try again shortly"), false);
                }
                if let Err(e) = cdt_session::check_imports(&batch.edits) {
                    return (LinkResponse::error(e.to_string()), false);
                }
                let before = self.doc.clone();
                let outcomes = match self.doc.apply_edits(&batch.edits) {
                    Ok(o) => o,
                    Err(e) => return (LinkResponse::error(e.to_string()), false),
                };
                self.record_undo(before);
                self.ui.selected = outcomes.iter().rev().find_map(|o| o.created()).or(self.ui.selected);
                self.selection.clear();
                self.status = Some(match &batch.label {
                    Some(label) => format!("Agent: {label}"),
                    None => format!("Agent applied {} edit(s)", batch.edits.len()),
                });
                let mut value = match self.link_report(&[]) {
                    Ok(v) => v,
                    Err(e) => return (LinkResponse::error(e), true),
                };
                value["outcomes"] = serde_json::to_value(&outcomes).unwrap_or_default();
                if save {
                    match self.current_path.clone() {
                        Some(path) => match self.save_project_to(&path.to_string_lossy()) {
                            Ok(()) => {
                                self.mark_saved();
                                value["written"] = serde_json::to_value(&path).unwrap_or_default();
                            }
                            Err(e) => value["save_error"] = e.into(),
                        },
                        None => value["save_error"] = "the project is not bound to a file yet".into(),
                    }
                }
                (LinkResponse::ok(value), true)
            }
            LinkRequest::Undo => {
                if self.busy() {
                    return (LinkResponse::error("the user is mid-edit; try again shortly"), false);
                }
                let done = self.undo();
                (LinkResponse::ok(serde_json::json!({ "undone": done })), done)
            }
            LinkRequest::Redo => {
                if self.busy() {
                    return (LinkResponse::error("the user is mid-edit; try again shortly"), false);
                }
                let done = self.redo();
                (LinkResponse::ok(serde_json::json!({ "redone": done })), done)
            }
            LinkRequest::Save => match self.current_path.clone() {
                Some(path) => match self.save_project_to(&path.to_string_lossy()) {
                    Ok(()) => {
                        self.mark_saved();
                        (LinkResponse::ok(serde_json::json!({ "written": path })), false)
                    }
                    Err(e) => (LinkResponse::error(e), false),
                },
                None => (LinkResponse::error("the project is not bound to a file yet; save it in the app first"), false),
            },
            LinkRequest::Render { path, width, height } => {
                let mesh = self.mesh();
                let opts = RenderOptions {
                    width,
                    height,
                    theme: self.viewport_theme(),
                    camera: self.camera_pose,
                    grid: self.grid(),
                    highlights: self.highlights(),
                };
                let result = HeadlessRenderer::new()
                    .and_then(|mut r| r.render(&mesh, &opts))
                    .and_then(|img| img.save(&path).map_err(|e| e.into()));
                match result {
                    Ok(()) => (LinkResponse::ok(serde_json::json!({ "written": path, "width": width, "height": height })), false),
                    Err(e) => (LinkResponse::error(format!("render failed: {e:#}")), false),
                }
            }
        }
    }
}
