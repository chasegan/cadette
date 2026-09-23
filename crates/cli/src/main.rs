//! `cdt` — Cadette from the command line, for scripts and AI agents.
//!
//! Every command reads a `.cdt` file, does one thing, and prints JSON to
//! stdout (errors go to stderr, with a non-zero exit code). It is stateless:
//! `apply` writes the file back (or to `--out`), so the next call sees the
//! result. Edits are `DocEdit` batches — see `cdt apply --help` for the shape.
//!
//! When a running Cadette window has the file open, `info`, `check` and
//! `apply` go through that window instead (the edit lands as an undo step
//! and the viewport updates); `--offline` forces the file path. `cdt live`
//! talks to a window directly, file or no file — "help me with what I have
//! open".
//!
//! Exit codes: 0 ok · 1 usage or I/O error · 2 the model has regeneration
//! errors (`check`, and `apply` after saving) · 3 an edit was refused.

use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{anyhow, Context};
use cdt_core::{DocEdit, Document, FeatureId};
use cdt_render::{CameraPose, GridSpec, Highlights, HeadlessRenderer, RenderOptions, ViewportTheme};
use cdt_session::link::{self, Instance, LinkRequest, LinkResponse};
use cdt_session::{EditBatch, Session, EXPORT_DEFLECTION_MM, PREVIEW_DEFLECTION_MM};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::Serialize;

#[derive(Parser)]
#[command(name = "cdt", version, about = "Cadette from the command line: inspect, edit, check, render and export .cdt projects.")]
struct Cli {
    /// Print compact JSON (one line) instead of pretty-printed.
    #[arg(long, global = true)]
    compact: bool,
    /// Work on the file directly even if a running window has it open.
    #[arg(long, global = true)]
    offline: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create an empty project file.
    New {
        file: PathBuf,
        /// The document name (defaults to the file stem).
        #[arg(long)]
        name: Option<String>,
    },
    /// Report every feature's status and every visible body's measurements.
    Info {
        file: PathBuf,
        /// Also list the faces and edges of these bodies (feature ids), or
        /// `all` for every visible body.
        #[arg(long, value_delimiter = ',')]
        detail: Vec<String>,
    },
    /// Regenerate and list the errors; exit 2 if there are any.
    Check { file: PathBuf },
    /// Apply a batch of edits (a JSON file, or `-` for stdin), regenerate,
    /// save, and report. The batch is `{"label": "...", "edits": [...]}` or a
    /// bare array of edits; each edit is `{"op": "add", "kind": {"Box": {"size": [40, 13, 26]}}}`,
    /// `{"op": "replace", "id": 12, "kind": ...}`, `{"op": "suppress", "id": 9, "suppressed": true}`,
    /// `{"op": "remove", "id": 9}`, `{"op": "insert_transform", "kind": {"Translate": {"source": 12, "offset": [10, 0, 0]}}}`,
    /// `{"op": "rename" ...}`, `set_color`, `ungroup`, `reorder`, `duplicate`, `set_rollback`, `rollback_to_tip`, `tidy`.
    Apply {
        file: PathBuf,
        /// The edit batch (JSON file), or `-` to read it from stdin.
        #[arg(default_value = "-")]
        edits: String,
        /// Write the result here instead of back to `file`.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Regenerate and report, but don't write anything.
        #[arg(long)]
        dry_run: bool,
    },
    /// Write the visible bodies as a binary STL.
    ExportStl {
        file: PathBuf,
        #[arg(short, long)]
        out: PathBuf,
        /// Mesh chord deviation in mm (smaller = finer).
        #[arg(long, default_value_t = EXPORT_DEFLECTION_MM)]
        deflection: f64,
    },
    /// Render the model to a PNG (no window needed).
    Render {
        file: PathBuf,
        #[arg(short, long)]
        out: PathBuf,
        #[command(flatten)]
        view: ViewArgs,
    },
    /// Talk to a running Cadette window (the most recent one, or --pid).
    Live {
        /// Which window, when several are open.
        #[arg(long, global = true)]
        pid: Option<u32>,
        #[command(subcommand)]
        command: LiveCommand,
    },
}

#[derive(Subcommand)]
enum LiveCommand {
    /// The running windows and the files they have open.
    List,
    /// Liveness and the open file.
    Ping,
    /// The report for the open project (+ face/edge detail for these ids).
    Info {
        #[arg(long, value_delimiter = ',')]
        detail: Vec<u64>,
    },
    /// The user's selection, the visible bodies, the camera, and whether
    /// the window is mid-edit.
    State,
    /// Apply an edit batch (JSON file or `-`) as one undo step.
    Apply {
        #[arg(default_value = "-")]
        edits: String,
        /// Also save the project to its file afterwards.
        #[arg(long)]
        save: bool,
    },
    Undo,
    Redo,
    /// Save the project to its file.
    Save,
    /// Render what the user sees (camera, theme, selection) to a PNG.
    Render {
        #[arg(short, long)]
        out: PathBuf,
        #[arg(long, default_value = "1280x800")]
        size: String,
    },
}

#[derive(Args)]
struct ViewArgs {
    /// A named view; `iso` frames the whole model from three-quarters.
    #[arg(long, default_value = "iso")]
    view: View,
    /// Override the view's orbit angles (radians).
    #[arg(long)]
    yaw: Option<f32>,
    #[arg(long)]
    pitch: Option<f32>,
    /// Image size as WIDTHxHEIGHT.
    #[arg(long, default_value = "1280x800")]
    size: String,
    #[arg(long, default_value = "light")]
    theme: Theme,
    /// Tint one body (by the feature id that produced it).
    #[arg(long)]
    highlight: Option<u64>,
    /// Draw the XY ground grid at this spacing (mm).
    #[arg(long)]
    grid: Option<f32>,
}

#[derive(Clone, Copy, ValueEnum)]
enum View {
    Iso,
    Top,
    Bottom,
    Front,
    Back,
    Left,
    Right,
}

#[derive(Clone, Copy, ValueEnum)]
enum Theme {
    Light,
    Dark,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli.command, cli.compact, cli.offline) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}

fn print<T: Serialize>(value: &T, compact: bool) -> anyhow::Result<()> {
    let text = if compact { serde_json::to_string(value)? } else { serde_json::to_string_pretty(value)? };
    println!("{text}");
    Ok(())
}

/// Send a request to a window and print its reply. Exit 1 on a link error
/// or a refused request; `bad_model` maps a `"ok": false` report to exit 2.
fn relay(instance: &Instance, request: LinkRequest, compact: bool, refused_code: u8) -> anyhow::Result<ExitCode> {
    let response: LinkResponse = link::send(&instance.socket, &request)
        .with_context(|| format!("talking to the Cadette window (pid {})", instance.pid))?;
    let mut value = match response.data {
        serde_json::Value::Object(_) => response.data,
        serde_json::Value::Null => serde_json::json!({}),
        other => serde_json::json!({ "result": other }),
    };
    value["via"] = serde_json::json!({ "pid": instance.pid, "file": instance.file });
    if !response.ok {
        value["ok"] = false.into();
        value["error"] = response.error.clone().into();
        print(&value, compact)?;
        return Ok(ExitCode::from(refused_code));
    }
    let model_ok = value.get("ok").and_then(|v| v.as_bool()).unwrap_or(true);
    print(&value, compact)?;
    Ok(if model_ok { ExitCode::SUCCESS } else { ExitCode::from(2) })
}

fn pick_instance(pid: Option<u32>) -> anyhow::Result<Instance> {
    let live = Instance::live();
    match pid {
        Some(pid) => live.into_iter().find(|i| i.pid == pid).ok_or_else(|| anyhow!("no running Cadette window with pid {pid}")),
        None => live.into_iter().next().ok_or_else(|| anyhow!("no running Cadette window found (open one, or work on a file directly)")),
    }
}

fn run(command: Command, compact: bool, offline: bool) -> anyhow::Result<ExitCode> {
    // A window with the file open answers for it: the edit becomes an undo
    // step there and the viewport shows it.
    let window_for = |file: &PathBuf| if offline { None } else { Instance::with_file(file) };
    match command {
        Command::Live { pid, command } => {
            if let LiveCommand::List = command {
                let live = Instance::live();
                print(&serde_json::json!({ "instances": live }), compact)?;
                return Ok(ExitCode::SUCCESS);
            }
            let instance = pick_instance(pid)?;
            let (request, refused) = match command {
                LiveCommand::List => unreachable!(),
                LiveCommand::Ping => (LinkRequest::Ping, 1),
                LiveCommand::Info { detail } => (LinkRequest::Info { detail: detail.into_iter().map(FeatureId).collect() }, 1),
                LiveCommand::State => (LinkRequest::State, 1),
                LiveCommand::Apply { edits, save } => (LinkRequest::Apply { batch: read_batch(&edits)?, save }, 3),
                LiveCommand::Undo => (LinkRequest::Undo, 1),
                LiveCommand::Redo => (LinkRequest::Redo, 1),
                LiveCommand::Save => (LinkRequest::Save, 1),
                LiveCommand::Render { out, size } => {
                    let (width, height) = parse_size(&size)?;
                    let path = std::path::absolute(&out).unwrap_or(out);
                    (LinkRequest::Render { path, width, height }, 1)
                }
            };
            relay(&instance, request, compact, refused)
        }
        Command::New { file, name } => {
            let name = name.unwrap_or_else(|| {
                file.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "untitled".into())
            });
            let session = Session::new(Document::new(name));
            session.save_as(&file).with_context(|| format!("writing {}", file.display()))?;
            print(&serde_json::json!({ "created": file }), compact)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Info { file, detail } => {
            if let Some(instance) = window_for(&file) {
                let detail = detail.iter().filter_map(|d| d.parse::<u64>().ok().map(FeatureId)).collect();
                return relay(&instance, LinkRequest::Info { detail }, compact, 1);
            }
            let mut session = open(&file)?;
            let report = session.report();
            let wanted: Vec<FeatureId> = if detail.iter().any(|d| d == "all") {
                report.bodies.iter().map(|b| b.feature).collect()
            } else {
                detail
                    .iter()
                    .map(|d| d.parse::<u64>().map(FeatureId).map_err(|_| anyhow!("--detail takes feature ids or `all`, not {d:?}")))
                    .collect::<Result<_, _>>()?
            };
            let details: Vec<_> = wanted
                .iter()
                .map(|&id| session.body_detail(id).map_err(|e| anyhow!("{e}")))
                .collect::<Result<_, _>>()?;
            #[derive(Serialize)]
            struct Out<'a> {
                #[serde(flatten)]
                report: &'a cdt_session::Report,
                #[serde(skip_serializing_if = "Vec::is_empty")]
                details: Vec<cdt_session::BodyDetail>,
            }
            print(&Out { report: &report, details }, compact)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Check { file } => {
            if let Some(instance) = window_for(&file) {
                let response = link::send(&instance.socket, &LinkRequest::Info { detail: vec![] })?;
                if !response.ok {
                    return Err(anyhow!("the window refused: {}", response.error.unwrap_or_default()));
                }
                let ok = response.data["ok"].as_bool().unwrap_or(false);
                let errors = response.data["errors"].clone();
                let bodies = response.data["bodies"].as_array().map(|b| b.len()).unwrap_or(0);
                print(&serde_json::json!({ "ok": ok, "errors": errors, "bodies": bodies, "via": { "pid": instance.pid } }), compact)?;
                return Ok(if ok { ExitCode::SUCCESS } else { ExitCode::from(2) });
            }
            let mut session = open(&file)?;
            let report = session.report();
            print(&serde_json::json!({ "ok": report.ok, "errors": report.errors, "bodies": report.bodies.len() }), compact)?;
            Ok(if report.ok { ExitCode::SUCCESS } else { ExitCode::from(2) })
        }
        Command::Apply { file, edits, out, dry_run } => {
            let batch = read_batch(&edits)?;
            if out.is_none() && !dry_run {
                if let Some(instance) = window_for(&file) {
                    return relay(&instance, LinkRequest::Apply { batch, save: true }, compact, 3);
                }
            }
            let mut session = open(&file)?;
            let applied = match session.apply(&batch.edits) {
                Ok(outcomes) => outcomes,
                Err(e) => {
                    print(&serde_json::json!({ "applied": false, "error": e.to_string(), "edit": e.index }), compact)?;
                    return Ok(ExitCode::from(3));
                }
            };
            let report = session.report();
            let written = if dry_run {
                None
            } else {
                let target = out.unwrap_or(file);
                session.save_as(&target).with_context(|| format!("writing {}", target.display()))?;
                Some(target)
            };
            #[derive(Serialize)]
            struct Out<'a> {
                applied: bool,
                label: Option<&'a str>,
                outcomes: &'a [cdt_core::EditOutcome],
                written: Option<PathBuf>,
                #[serde(flatten)]
                report: &'a cdt_session::Report,
            }
            print(
                &Out { applied: true, label: batch.label.as_deref(), outcomes: &applied, written, report: &report },
                compact,
            )?;
            Ok(if report.ok { ExitCode::SUCCESS } else { ExitCode::from(2) })
        }
        Command::ExportStl { file, out, deflection } => {
            let mut session = open(&file)?;
            session.export_stl(&out, deflection).map_err(|e| anyhow!("{e}"))?;
            let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
            print(&serde_json::json!({ "written": out, "bytes": bytes, "deflection_mm": deflection }), compact)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Render { file, out, view } => {
            let mut session = open(&file)?;
            let scene = session.scene_mesh(PREVIEW_DEFLECTION_MM);
            let (width, height) = parse_size(&view.size)?;
            let framed = cdt_render::frame_camera(&scene.mesh);
            let (mut yaw, mut pitch) = match view.view {
                View::Iso => (framed.yaw, framed.pitch),
                View::Top => (-std::f32::consts::FRAC_PI_2, 1.55),
                View::Bottom => (-std::f32::consts::FRAC_PI_2, -1.55),
                View::Front => (-std::f32::consts::FRAC_PI_2, 0.0),
                View::Back => (std::f32::consts::FRAC_PI_2, 0.0),
                View::Right => (0.0, 0.0),
                View::Left => (std::f32::consts::PI, 0.0),
            };
            if let Some(y) = view.yaw {
                yaw = y;
            }
            if let Some(p) = view.pitch {
                pitch = p;
            }
            let camera = CameraPose { target: framed.target.to_array(), yaw, pitch, distance: framed.distance };
            let highlights = match view.highlight {
                Some(id) => {
                    let range = scene
                        .bodies
                        .iter()
                        .find(|b| b.feature == FeatureId(id))
                        .ok_or_else(|| anyhow!("feature {id} is not a visible body"))?;
                    Highlights { hover_body: Some((range.face_start, range.face_end)), ..Default::default() }
                }
                None => Highlights::default(),
            };
            let grid = match view.grid {
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
                theme: match view.theme {
                    Theme::Light => ViewportTheme::LIGHT,
                    Theme::Dark => ViewportTheme::DARK,
                },
                camera: Some(camera),
                grid,
                highlights,
            };
            let image = HeadlessRenderer::new()?.render(&scene.mesh, &opts)?;
            image.save(&out).with_context(|| format!("writing {}", out.display()))?;
            print(
                &serde_json::json!({
                    "written": out, "width": width, "height": height,
                    "bodies": scene.bodies.iter().map(|b| b.feature).collect::<Vec<_>>(),
                    "tessellation_failures": scene.failures.iter().map(|(id, m)| serde_json::json!({"feature": id, "message": m})).collect::<Vec<_>>(),
                    "camera": { "yaw": yaw, "pitch": pitch, "distance": framed.distance, "target": framed.target.to_array() },
                }),
                compact,
            )?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn open(file: &PathBuf) -> anyhow::Result<Session> {
    Session::open(file).map_err(|e| anyhow!("{e}")).with_context(|| format!("opening {}", file.display()))
}

/// A batch file: `{"label": ..., "edits": [...]}` or a bare array of edits.
fn read_batch(source: &str) -> anyhow::Result<EditBatch> {
    let text = if source == "-" {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s).context("reading edits from stdin")?;
        s
    } else {
        std::fs::read_to_string(source).with_context(|| format!("reading {source}"))?
    };
    let value: serde_json::Value = serde_json::from_str(&text).context("the edit batch is not valid JSON")?;
    if value.is_array() {
        let edits: Vec<DocEdit> = serde_json::from_value(value).context("an edit in the batch is malformed")?;
        Ok(EditBatch { label: None, edits })
    } else {
        serde_json::from_value(value).context("the batch must be an array of edits or {\"label\", \"edits\"}")
    }
}

fn parse_size(s: &str) -> anyhow::Result<(u32, u32)> {
    let (w, h) = s.split_once('x').ok_or_else(|| anyhow!("--size takes WIDTHxHEIGHT, not {s:?}"))?;
    Ok((w.parse()?, h.parse()?))
}
