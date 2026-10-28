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
//! open". `cdt mcp` serves all of this to an AI agent over the Model Context
//! Protocol (stdio).
//!
//! Exit codes: 0 ok · 1 usage or I/O error · 2 the model has regeneration
//! errors (`check`, and `apply` after saving) · 3 an edit was refused.

mod mcp;
mod ops;

use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Context;
use cdt_session::{EditBatch, EXPORT_DEFLECTION_MM};
use clap::{Args, Parser, Subcommand, ValueEnum};
use ops::{Detail, Outcome, RenderArgs, Target, View};

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
    /// Import a STEP file as a new feature (the file is stored inside the
    /// project). Goes through the window if one has the project open.
    Import {
        file: PathBuf,
        /// The STEP file (.step / .stp) to import.
        step: PathBuf,
        /// The feature's name (default: the STEP file's stem).
        #[arg(long)]
        name: Option<String>,
        /// Write the result here instead of back to `file`.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Write the visible bodies as a STEP file (exact B-rep, in millimetres).
    ExportStep {
        file: PathBuf,
        /// Where to write the STEP file.
        #[arg(short, long)]
        out: PathBuf,
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
    /// Serve the Model Context Protocol over stdio, for AI agents. Register
    /// as `{"command": "cdt", "args": ["mcp"]}`.
    Mcp,
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
        detail: Vec<String>,
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
    /// Import a STEP file into the open project as one undo step.
    Import {
        /// The STEP file (.step / .stp) to import.
        step: PathBuf,
        /// The feature's name (default: the STEP file's stem).
        #[arg(long)]
        name: Option<String>,
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
    view: ViewName,
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
enum ViewName {
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
    if let Command::Mcp = cli.command {
        return match mcp::serve() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e:#}");
                ExitCode::from(1)
            }
        };
    }
    match run(cli.command, cli.offline) {
        Ok(outcome) => {
            let text = if cli.compact {
                serde_json::to_string(&outcome.value)
            } else {
                serde_json::to_string_pretty(&outcome.value)
            };
            match text {
                Ok(t) => println!("{t}"),
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(1);
                }
            }
            ExitCode::from(outcome.code)
        }
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}

fn run(command: Command, offline: bool) -> anyhow::Result<Outcome> {
    match command {
        Command::Mcp => unreachable!("handled in main"),
        Command::New { file, name } => ops::new_project(&file, name),
        Command::Info { file, detail } => ops::info(&Target::file(file, offline), Detail::parse(&detail)?),
        Command::Check { file } => ops::check(&Target::file(file, offline)),
        Command::Apply { file, edits, out, dry_run } => {
            ops::apply(&Target::file(file, offline), read_batch(&edits)?, out, dry_run, true)
        }
        Command::Import { file, step, name, out } => {
            ops::import_step(&Target::file(file, offline), &step, name, out, true)
        }
        Command::ExportStep { file, out } => ops::export_step(&file, &out),
        Command::ExportStl { file, out, deflection } => ops::export_stl(&file, &out, Some(deflection)),
        Command::Render { file, out, view } => {
            let (width, height) = RenderArgs::parse_size(&view.size)?;
            let args = RenderArgs {
                view: match view.view {
                    ViewName::Iso => View::Iso,
                    ViewName::Top => View::Top,
                    ViewName::Bottom => View::Bottom,
                    ViewName::Front => View::Front,
                    ViewName::Back => View::Back,
                    ViewName::Left => View::Left,
                    ViewName::Right => View::Right,
                },
                yaw: view.yaw,
                pitch: view.pitch,
                width,
                height,
                dark: matches!(view.theme, Theme::Dark),
                highlight: view.highlight.map(cdt_core::FeatureId),
                grid: view.grid,
                as_seen: false,
            };
            let mut outcome = ops::render(&Target::file(file, true), Some(&out), &args)?;
            outcome.image_png = None;
            Ok(outcome)
        }
        Command::Live { pid, command } => {
            let target = Target::window(pid);
            match command {
                LiveCommand::List => Ok(ops::windows()),
                LiveCommand::Ping => ops::ping(&target),
                LiveCommand::Info { detail } => ops::info(&target, Detail::parse(&detail)?),
                LiveCommand::State => ops::state(&target),
                LiveCommand::Apply { edits, save } => ops::apply(&target, read_batch(&edits)?, None, false, save),
                LiveCommand::Import { step, name, save } => ops::import_step(&target, &step, name, None, save),
                LiveCommand::Undo => ops::undo(&target),
                LiveCommand::Redo => ops::redo(&target),
                LiveCommand::Save => ops::save(&target),
                LiveCommand::Render { out, size } => {
                    let (width, height) = RenderArgs::parse_size(&size)?;
                    let args = RenderArgs { width, height, as_seen: true, ..Default::default() };
                    let mut outcome = ops::render(&target, Some(&out), &args)?;
                    outcome.image_png = None;
                    Ok(outcome)
                }
            }
        }
    }
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
    ops::parse_batch(value)
}
