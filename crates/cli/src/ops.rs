//! The operations behind both the `cdt` subcommands and the MCP tools: each
//! takes plain arguments, works on a file (a headless [`Session`]) or on a
//! running window (through the control link), and returns JSON plus an exit
//! code — no printing here.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context};
use cdt_core::{DocEdit, Document, FeatureId};
use cdt_render::{CameraPose, GridSpec, HeadlessRenderer, Highlights, RenderOptions, ViewportTheme};
use cdt_session::link::{self, Instance, LinkRequest, LinkResponse};
use cdt_session::{EditBatch, Session, EXPORT_DEFLECTION_MM, PREVIEW_DEFLECTION_MM};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Exit codes: 0 ok · 1 usage or I/O error · 2 the model has regeneration
/// errors · 3 an edit was refused.
pub const EXIT_MODEL_ERRORS: u8 = 2;
pub const EXIT_REFUSED: u8 = 3;

/// A JSON result plus the exit code it implies.
pub struct Outcome {
    pub value: Value,
    pub code: u8,
    /// A rendered image (PNG bytes), for `render`.
    pub image_png: Option<Vec<u8>>,
}

impl Outcome {
    fn ok(value: Value) -> Self {
        Self { value, code: 0, image_png: None }
    }
}

/// Where an operation should act: a file (possibly via the window that has
/// it open), or a window picked by pid / most recent.
#[derive(Clone, Debug, Default)]
pub struct Target {
    pub file: Option<PathBuf>,
    /// Work on the file directly even if a window has it open.
    pub offline: bool,
    /// A specific window (only meaningful without `file`, or to disambiguate).
    pub pid: Option<u32>,
}

enum Resolved {
    File(PathBuf),
    Window(Instance),
}

impl Target {
    pub fn file(file: impl Into<PathBuf>, offline: bool) -> Self {
        Self { file: Some(file.into()), offline, pid: None }
    }

    pub fn window(pid: Option<u32>) -> Self {
        Self { file: None, offline: false, pid }
    }

    fn resolve(&self) -> anyhow::Result<Resolved> {
        match &self.file {
            Some(file) => {
                if !self.offline {
                    if let Some(instance) = Instance::with_file(file) {
                        return Ok(Resolved::Window(instance));
                    }
                }
                Ok(Resolved::File(file.clone()))
            }
            None => Ok(Resolved::Window(pick_instance(self.pid)?)),
        }
    }

    /// Only a window will do (state, undo, redo, save, what-the-user-sees).
    fn window_only(&self) -> anyhow::Result<Instance> {
        match self.resolve()? {
            Resolved::Window(i) => Ok(i),
            Resolved::File(f) => Err(anyhow!(
                "no running Cadette window has {} open (this needs a window; open the file in Cadette first)",
                f.display()
            )),
        }
    }
}

pub fn pick_instance(pid: Option<u32>) -> anyhow::Result<Instance> {
    let live = Instance::live();
    match pid {
        Some(pid) => live.into_iter().find(|i| i.pid == pid).ok_or_else(|| anyhow!("no running Cadette window with pid {pid}")),
        None => live
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("no running Cadette window found (open one, or give a file to work on directly)")),
    }
}

/// Send a request to a window; the reply's payload with a `via` stamp, and
/// the exit code (`refused_code` when the window said no; 2 when the payload
/// reports a model that didn't build).
fn relay(instance: &Instance, request: LinkRequest, refused_code: u8) -> anyhow::Result<Outcome> {
    let response: LinkResponse = link::send(&instance.socket, &request)
        .with_context(|| format!("talking to the Cadette window (pid {})", instance.pid))?;
    let mut value = match response.data {
        Value::Object(_) => response.data,
        Value::Null => json!({}),
        other => json!({ "result": other }),
    };
    value["via"] = json!({ "pid": instance.pid, "file": instance.file });
    if !response.ok {
        value["ok"] = false.into();
        value["error"] = response.error.into();
        return Ok(Outcome { value, code: refused_code, image_png: None });
    }
    let model_ok = value.get("ok").and_then(|v| v.as_bool()).unwrap_or(true);
    Ok(Outcome { value, code: if model_ok { 0 } else { EXIT_MODEL_ERRORS }, image_png: None })
}

fn open(file: &Path) -> anyhow::Result<Session> {
    Session::open(file).map_err(|e| anyhow!("{e}")).with_context(|| format!("opening {}", file.display()))
}

/// Which bodies to detail: feature ids, or every visible body.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Detail {
    #[default]
    None,
    Ids(Vec<FeatureId>),
    All,
}

impl Detail {
    /// From the CLI's `--detail a,b,c` / `--detail all`.
    pub fn parse(items: &[String]) -> anyhow::Result<Self> {
        if items.is_empty() {
            return Ok(Detail::None);
        }
        if items.iter().any(|d| d == "all") {
            return Ok(Detail::All);
        }
        items
            .iter()
            .map(|d| d.parse::<u64>().map(FeatureId).map_err(|_| anyhow!("--detail takes feature ids or `all`, not {d:?}")))
            .collect::<Result<Vec<_>, _>>()
            .map(Detail::Ids)
    }
}

pub fn new_project(file: &Path, name: Option<String>) -> anyhow::Result<Outcome> {
    let name = name.unwrap_or_else(|| {
        file.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "untitled".into())
    });
    Session::new(Document::new(name)).save_as(file).with_context(|| format!("writing {}", file.display()))?;
    Ok(Outcome::ok(json!({ "created": file })))
}

pub fn info(target: &Target, detail: Detail) -> anyhow::Result<Outcome> {
    match target.resolve()? {
        Resolved::Window(instance) => {
            let ids = match &detail {
                Detail::None => vec![],
                Detail::Ids(ids) => ids.clone(),
                Detail::All => {
                    // Ask for the report first to learn the visible bodies.
                    let first = relay(&instance, LinkRequest::Info { detail: vec![] }, 1)?;
                    first.value["bodies"]
                        .as_array()
                        .map(|b| b.iter().filter_map(|x| x["feature"].as_u64().map(FeatureId)).collect())
                        .unwrap_or_default()
                }
            };
            relay(&instance, LinkRequest::Info { detail: ids }, 1)
        }
        Resolved::File(file) => {
            let mut session = open(&file)?;
            let report = session.report();
            let ids: Vec<FeatureId> = match detail {
                Detail::None => vec![],
                Detail::Ids(ids) => ids,
                Detail::All => report.bodies.iter().map(|b| b.feature).collect(),
            };
            let details: Vec<_> = ids
                .iter()
                .map(|&id| session.body_detail(id).map_err(|e| anyhow!("{e}")))
                .collect::<Result<_, _>>()?;
            let mut value = serde_json::to_value(&report)?;
            if !details.is_empty() {
                value["details"] = serde_json::to_value(&details)?;
            }
            Ok(Outcome { value, code: if report.ok { 0 } else { EXIT_MODEL_ERRORS }, image_png: None })
        }
    }
}

pub fn check(target: &Target) -> anyhow::Result<Outcome> {
    let full = info(target, Detail::None)?;
    let ok = full.value["ok"].as_bool().unwrap_or(false);
    let mut value = json!({
        "ok": ok,
        "errors": full.value["errors"].clone(),
        "bodies": full.value["bodies"].as_array().map(|b| b.len()).unwrap_or(0),
    });
    if let Some(via) = full.value.get("via") {
        value["via"] = via.clone();
    }
    Ok(Outcome { value, code: if ok { 0 } else { EXIT_MODEL_ERRORS }, image_png: None })
}

/// Apply a batch. On a file: save back (or to `out`) unless `dry_run`. On a
/// window: the edit lands there as an undo step, saved if `save`.
pub fn apply(target: &Target, batch: EditBatch, out: Option<PathBuf>, dry_run: bool, save: bool) -> anyhow::Result<Outcome> {
    let resolved = match (&out, dry_run) {
        // A different output file or a dry run never goes through a window.
        (Some(_), _) | (_, true) => match &target.file {
            Some(f) => Resolved::File(f.clone()),
            None => return Err(anyhow!("--out / --dry-run need a file to work on")),
        },
        _ => target.resolve()?,
    };
    match resolved {
        Resolved::Window(instance) => relay(&instance, LinkRequest::Apply { batch, save }, EXIT_REFUSED),
        Resolved::File(file) => {
            let mut session = open(&file)?;
            let outcomes = match session.apply(&batch.edits) {
                Ok(o) => o,
                Err(e) => {
                    return Ok(Outcome {
                        value: json!({ "applied": false, "error": e.to_string(), "edit": e.index }),
                        code: EXIT_REFUSED,
                        image_png: None,
                    })
                }
            };
            let report = session.report();
            let written = if dry_run {
                None
            } else {
                let path = out.unwrap_or(file);
                session.save_as(&path).with_context(|| format!("writing {}", path.display()))?;
                Some(path)
            };
            let mut value = serde_json::to_value(&report)?;
            value["applied"] = true.into();
            value["label"] = batch.label.into();
            value["outcomes"] = serde_json::to_value(&outcomes)?;
            value["written"] = serde_json::to_value(written)?;
            Ok(Outcome { value, code: if report.ok { 0 } else { EXIT_MODEL_ERRORS }, image_png: None })
        }
    }
}

/// The edit that imports `step` (a STEP file on disk): its bytes, base64,
/// under the file's own name.
pub fn import_edit(step: &Path, name: Option<String>) -> anyhow::Result<DocEdit> {
    use base64::Engine as _;
    let bytes = std::fs::read(step).with_context(|| format!("reading {}", step.display()))?;
    let file = step
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .ok_or_else(|| anyhow!("{} has no file name", step.display()))?;
    Ok(DocEdit::ImportStep { name, file, data: base64::engine::general_purpose::STANDARD.encode(bytes) })
}

/// Import a STEP file as a new feature — through the window if one has the
/// project open (one undoable step), else on the file.
pub fn import_step(target: &Target, step: &Path, name: Option<String>, out: Option<PathBuf>, save: bool) -> anyhow::Result<Outcome> {
    let edit = import_edit(step, name)?;
    let label = match &edit {
        DocEdit::ImportStep { file, .. } => format!("Import {file}"),
        _ => unreachable!(),
    };
    apply(target, EditBatch { label: Some(label), edits: vec![edit] }, out, false, save)
}

pub fn export_step(file: &Path, out: &Path) -> anyhow::Result<Outcome> {
    let mut session = open(file)?;
    session.export_step(out).map_err(|e| anyhow!("{e}"))?;
    let bytes = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    Ok(Outcome::ok(json!({ "written": out, "bytes": bytes })))
}

pub fn export_stl(file: &Path, out: &Path, deflection: Option<f64>) -> anyhow::Result<Outcome> {
    let deflection = deflection.unwrap_or(EXPORT_DEFLECTION_MM);
    let mut session = open(file)?;
    session.export_stl(out, deflection).map_err(|e| anyhow!("{e}"))?;
    let bytes = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    Ok(Outcome::ok(json!({ "written": out, "bytes": bytes, "deflection_mm": deflection })))
}

/// A named view for [`render`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum View {
    #[default]
    Iso,
    Top,
    Bottom,
    Front,
    Back,
    Left,
    Right,
}

impl View {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        serde_json::from_value(Value::String(s.to_lowercase())).map_err(|_| anyhow!("unknown view {s:?}"))
    }

    /// Orbit angles (yaw, pitch) in radians; `None` keeps the automatic framing's.
    fn angles(self) -> Option<(f32, f32)> {
        use std::f32::consts::{FRAC_PI_2, PI};
        Some(match self {
            View::Iso => return None,
            View::Top => (-FRAC_PI_2, 1.55),
            View::Bottom => (-FRAC_PI_2, -1.55),
            View::Front => (-FRAC_PI_2, 0.0),
            View::Back => (FRAC_PI_2, 0.0),
            View::Right => (0.0, 0.0),
            View::Left => (PI, 0.0),
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct RenderArgs {
    pub view: View,
    pub yaw: Option<f32>,
    pub pitch: Option<f32>,
    pub width: u32,
    pub height: u32,
    pub dark: bool,
    /// Tint one body (by the feature id that produced it).
    pub highlight: Option<FeatureId>,
    /// Draw the XY ground grid at this spacing (mm).
    pub grid: Option<f32>,
    /// Through a window: what the user sees (their camera, theme, selection)
    /// instead of a named view.
    pub as_seen: bool,
}

impl RenderArgs {
    pub fn parse_size(s: &str) -> anyhow::Result<(u32, u32)> {
        let (w, h) = s.split_once('x').ok_or_else(|| anyhow!("size takes WIDTHxHEIGHT, not {s:?}"))?;
        Ok((w.parse()?, h.parse()?))
    }
}

/// Render to `out` (a PNG path; a temp file if `None`) and also return the
/// PNG bytes.
pub fn render(target: &Target, out: Option<&Path>, args: &RenderArgs) -> anyhow::Result<Outcome> {
    let (width, height) = (args.width.max(1), args.height.max(1));
    let temp;
    let out: &Path = match out {
        Some(p) => p,
        None => {
            temp = std::env::temp_dir().join(format!("cdt-render-{}.png", std::process::id()));
            &temp
        }
    };
    if args.as_seen {
        let instance = target.window_only()?;
        let path = std::path::absolute(out).unwrap_or_else(|_| out.to_path_buf());
        let mut outcome = relay(&instance, LinkRequest::Render { path: path.clone(), width, height }, 1)?;
        if outcome.code == 0 {
            outcome.image_png = Some(std::fs::read(&path).with_context(|| format!("reading back {}", path.display()))?);
        }
        return Ok(outcome);
    }
    let file = match &target.file {
        Some(f) => f.clone(),
        None => {
            // A window but no file: render the file it has open (saved
            // state); for the live view use `as_seen`.
            let instance = pick_instance(target.pid)?;
            instance.file.clone().ok_or_else(|| anyhow!(
                "the window (pid {}) has an unsaved project; render it with as_seen (what the user sees) or save it first",
                instance.pid
            ))?
        }
    };
    let mut session = open(&file)?;
    let scene = session.scene_mesh(PREVIEW_DEFLECTION_MM);
    let framed = cdt_render::frame_camera(&scene.mesh);
    let (mut yaw, mut pitch) = args.view.angles().unwrap_or((framed.yaw, framed.pitch));
    if let Some(y) = args.yaw {
        yaw = y;
    }
    if let Some(p) = args.pitch {
        pitch = p;
    }
    let camera = CameraPose { target: framed.target.to_array(), yaw, pitch, distance: framed.distance };
    let highlights = match args.highlight {
        Some(id) => {
            let range = scene
                .bodies
                .iter()
                .find(|b| b.feature == id)
                .ok_or_else(|| anyhow!("feature {} is not a visible body", id.0))?;
            Highlights { hover_body: Some((range.face_start, range.face_end)), ..Default::default() }
        }
        None => Highlights::default(),
    };
    let grid = match args.grid {
        Some(spacing) => {
            let extent = scene
                .mesh
                .vertices
                .iter()
                .map(|v| v.position[0].abs().max(v.position[1].abs()))
                .fold(spacing * 10.0, f32::max);
            GridSpec { spacing, half_extent: extent * 1.5, visible: true, ..Default::default() }
        }
        None => GridSpec::default(),
    };
    let opts = RenderOptions {
        width,
        height,
        theme: if args.dark { ViewportTheme::DARK } else { ViewportTheme::LIGHT },
        camera: Some(camera),
        grid,
        highlights,
    };
    let image = HeadlessRenderer::new()?.render(&scene.mesh, &opts)?;
    image.save(out).with_context(|| format!("writing {}", out.display()))?;
    let png = std::fs::read(out)?;
    Ok(Outcome {
        value: json!({
            "written": out, "width": width, "height": height,
            "bodies": scene.bodies.iter().map(|b| b.feature).collect::<Vec<_>>(),
            "tessellation_failures": scene.failures.iter().map(|(id, m)| json!({ "feature": id, "message": m })).collect::<Vec<_>>(),
            "camera": { "view": args.view, "yaw": yaw, "pitch": pitch, "distance": framed.distance, "target": framed.target.to_array() },
        }),
        code: 0,
        image_png: Some(png),
    })
}

// --- window-only operations -------------------------------------------------

pub fn windows() -> Outcome {
    Outcome::ok(json!({ "instances": Instance::live() }))
}

pub fn ping(target: &Target) -> anyhow::Result<Outcome> {
    relay(&target.window_only()?, LinkRequest::Ping, 1)
}

pub fn state(target: &Target) -> anyhow::Result<Outcome> {
    relay(&target.window_only()?, LinkRequest::State, 1)
}

pub fn undo(target: &Target) -> anyhow::Result<Outcome> {
    relay(&target.window_only()?, LinkRequest::Undo, 1)
}

pub fn redo(target: &Target) -> anyhow::Result<Outcome> {
    relay(&target.window_only()?, LinkRequest::Redo, 1)
}

pub fn save(target: &Target) -> anyhow::Result<Outcome> {
    relay(&target.window_only()?, LinkRequest::Save, 1)
}

/// A batch file: `{"label": ..., "edits": [...]}` or a bare array of edits.
pub fn parse_batch(value: Value) -> anyhow::Result<EditBatch> {
    if value.is_array() {
        let edits: Vec<DocEdit> = serde_json::from_value(value).context("an edit in the batch is malformed")?;
        Ok(EditBatch { label: None, edits })
    } else {
        serde_json::from_value(value).context("the batch must be an array of edits or {\"label\", \"edits\"}")
    }
}
