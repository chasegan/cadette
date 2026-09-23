//! `cdt mcp`: a Model Context Protocol server over stdio, exposing the
//! operations in [`crate::ops`] as tools and the agent reference as a
//! resource. Hand-rolled JSON-RPC (newline-delimited) — the protocol surface
//! we need is small: initialize, tools/list, tools/call, resources/list,
//! resources/read, ping.

use std::io::{BufRead, Write};
use std::path::PathBuf;

use anyhow::anyhow;
use base64::Engine;
use cdt_core::FeatureId;
use serde_json::{json, Value};

use crate::ops::{self, Detail, RenderArgs, Target, View};

const SUPPORTED_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];
const HELP: &str = include_str!("help.md");
const HELP_URI: &str = "cadette://guide";

pub fn serve() -> anyhow::Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let message: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                write(&mut stdout, &error_response(Value::Null, -32700, &format!("parse error: {e}")))?;
                continue;
            }
        };
        let id = message.get("id").cloned().unwrap_or(Value::Null);
        let method = message["method"].as_str().unwrap_or("");
        let params = message.get("params").cloned().unwrap_or(json!({}));
        if id.is_null() {
            // A notification (initialized, cancelled, ...): nothing to answer.
            continue;
        }
        let response = match handle(method, &params) {
            Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            Err(Rpc::MethodNotFound) => error_response(id, -32601, &format!("method not found: {method}")),
            Err(Rpc::Invalid(msg)) => error_response(id, -32602, &msg),
            Err(Rpc::Internal(msg)) => error_response(id, -32603, &msg),
        };
        write(&mut stdout, &response)?;
    }
    Ok(())
}

fn write(out: &mut impl Write, value: &Value) -> anyhow::Result<()> {
    let mut text = serde_json::to_string(value)?;
    text.push('\n');
    out.write_all(text.as_bytes())?;
    out.flush()?;
    Ok(())
}

fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

enum Rpc {
    MethodNotFound,
    Invalid(String),
    Internal(String),
}

fn handle(method: &str, params: &Value) -> Result<Value, Rpc> {
    match method {
        "initialize" => {
            let requested = params["protocolVersion"].as_str().unwrap_or("");
            let version = if SUPPORTED_VERSIONS.contains(&requested) { requested } else { SUPPORTED_VERSIONS[0] };
            Ok(json!({
                "protocolVersion": version,
                "capabilities": { "tools": {}, "resources": {} },
                "serverInfo": { "name": "cadette", "version": env!("CARGO_PKG_VERSION") },
                "instructions": "Cadette parametric CAD. Start with the cadette://guide resource (or the cadette_help tool) for the edit and feature-kind JSON shapes. Work on a .cdt file by path; if a Cadette window has it open, edits land there live.",
            }))
        }
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tools() })),
        "tools/call" => {
            let name = params["name"].as_str().ok_or_else(|| Rpc::Invalid("tools/call needs a name".into()))?;
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            Ok(call_tool(name, &args))
        }
        "resources/list" => Ok(json!({ "resources": [{
            "uri": HELP_URI, "name": "Cadette agent guide", "mimeType": "text/markdown",
            "description": "How the history works, the edit ops, and every feature kind's JSON shape."
        }] })),
        "resources/read" => {
            let uri = params["uri"].as_str().unwrap_or("");
            if uri != HELP_URI {
                return Err(Rpc::Invalid(format!("unknown resource {uri}")));
            }
            Ok(json!({ "contents": [{ "uri": HELP_URI, "mimeType": "text/markdown", "text": HELP }] }))
        }
        "prompts/list" => Ok(json!({ "prompts": [] })),
        "logging/setLevel" => Ok(json!({})),
        _ => Err(Rpc::MethodNotFound),
    }
}

fn tools() -> Vec<Value> {
    let file = json!({ "type": "string", "description": "Path to the .cdt project file." });
    let file_opt = json!({ "type": "string", "description": "Path to the .cdt project file. Omit to target the most recently used running Cadette window." });
    let offline = json!({ "type": "boolean", "description": "Work on the file directly even if a Cadette window has it open (default false: go through the window so the user sees the edit)." });
    let pid = json!({ "type": "integer", "description": "A specific Cadette window (from cadette_windows), when several are open." });
    vec![
        tool("cadette_help", "The agent guide: how the history works, the edit ops, and every feature kind's JSON shape. Read it once before editing.", json!({ "type": "object", "properties": {} })),
        tool("cadette_windows", "List running Cadette windows (pid, socket, open file).", json!({ "type": "object", "properties": {} })),
        tool("cadette_new", "Create an empty project file.", json!({ "type": "object", "properties": { "file": file, "name": { "type": "string" } }, "required": ["file"] })),
        tool("cadette_info", "The report: every feature (id, name, kind, status, error, visible) and every visible body's exact bbox, volume, surface area, centroid, face/edge counts and validity. `detail` adds each listed body's faces (surface, area, centroid, outward normal, radius) and edges (curve, length, endpoints, on-curve midpoint = a fillet anchor).", json!({ "type": "object", "properties": { "file": file_opt, "offline": offline, "pid": pid, "detail": { "description": "Feature ids of bodies to detail, or \"all\".", "oneOf": [ { "type": "array", "items": { "type": "integer" } }, { "type": "string", "enum": ["all"] } ] } } })),
        tool("cadette_check", "Regenerate and return ok + the errors (which step failed and why).", json!({ "type": "object", "properties": { "file": file_opt, "offline": offline, "pid": pid } })),
        tool("cadette_apply", "Apply a batch of edits (see cadette_help) all-or-nothing, regenerate, and return the outcomes (new ids) plus the fresh report. On a file: saved back (or to `out`). Through a window: one undoable step, saved if `save`.", json!({ "type": "object", "properties": { "file": file_opt, "offline": offline, "pid": pid, "edits": { "type": "array", "items": { "type": "object" }, "description": "The edits, e.g. [{\"op\":\"add\",\"kind\":{\"Box\":{\"size\":[40,13,26]}}}]." }, "label": { "type": "string", "description": "Shown to the user as the name of this step." }, "out": { "type": "string", "description": "Write the result to this file instead of back to `file`." }, "dry_run": { "type": "boolean" }, "save": { "type": "boolean", "description": "Through a window: also save the project to its file." } }, "required": ["edits"] })),
        tool("cadette_render", "Render the model to a PNG and return the image. Named views (iso/top/bottom/front/back/left/right) with optional yaw/pitch, size, dark theme, a highlighted body and a ground grid; or `as_seen` for the user's own camera/selection in a running window.", json!({ "type": "object", "properties": { "file": file_opt, "pid": pid, "out": { "type": "string", "description": "Where to write the PNG (a temp file if omitted)." }, "view": { "type": "string", "enum": ["iso", "top", "bottom", "front", "back", "left", "right"] }, "yaw": { "type": "number" }, "pitch": { "type": "number" }, "width": { "type": "integer" }, "height": { "type": "integer" }, "dark": { "type": "boolean" }, "highlight": { "type": "integer", "description": "Feature id of a visible body to tint." }, "grid": { "type": "number", "description": "Ground grid spacing in mm." }, "as_seen": { "type": "boolean", "description": "What the user sees in their window (camera, theme, selection)." } } })),
        tool("cadette_export_stl", "Write the visible bodies as a binary STL.", json!({ "type": "object", "properties": { "file": file, "out": { "type": "string" }, "deflection": { "type": "number", "description": "Mesh chord deviation in mm (default 0.05)." } }, "required": ["file", "out"] })),
        tool("cadette_state", "A running window's state: the open file, dirty flag, the user's selection (`selected_feature`, `picks` with body feature id + local face/edge index), visible bodies, camera, errors, and `busy` (mid-drag or sketching: edits are refused).", json!({ "type": "object", "properties": { "file": file_opt, "pid": pid } })),
        tool("cadette_undo", "Undo the last step in a running window.", json!({ "type": "object", "properties": { "file": file_opt, "pid": pid } })),
        tool("cadette_redo", "Redo in a running window.", json!({ "type": "object", "properties": { "file": file_opt, "pid": pid } })),
        tool("cadette_save", "Save a running window's project to its file.", json!({ "type": "object", "properties": { "file": file_opt, "pid": pid } })),
    ]
}

fn tool(name: &str, description: &str, schema: Value) -> Value {
    json!({ "name": name, "description": description, "inputSchema": schema })
}

fn target(args: &Value) -> Target {
    Target {
        file: args["file"].as_str().map(PathBuf::from),
        offline: args["offline"].as_bool().unwrap_or(false),
        pid: args["pid"].as_u64().map(|p| p as u32),
    }
}

fn call_tool(name: &str, args: &Value) -> Value {
    match run_tool(name, args) {
        Ok(outcome) => {
            let mut content = vec![json!({ "type": "text", "text": serde_json::to_string_pretty(&outcome.value).unwrap_or_default() })];
            if let Some(png) = outcome.image_png {
                content.insert(0, json!({ "type": "image", "data": base64::engine::general_purpose::STANDARD.encode(png), "mimeType": "image/png" }));
            }
            // A refused edit or an I/O failure is an error the agent must notice; a
            // model that regenerated with failing steps is a normal result whose
            // report says `ok: false`.
            json!({ "content": content, "isError": outcome.code == 1 || outcome.code == ops::EXIT_REFUSED, "structuredContent": outcome.value })
        }
        Err(e) => json!({ "content": [{ "type": "text", "text": format!("error: {e:#}") }], "isError": true }),
    }
}

fn run_tool(name: &str, args: &Value) -> anyhow::Result<ops::Outcome> {
    match name {
        "cadette_help" => Ok(ops::Outcome { value: json!({ "guide": HELP }), code: 0, image_png: None }),
        "cadette_windows" => Ok(ops::windows()),
        "cadette_new" => {
            let file = args["file"].as_str().ok_or_else(|| anyhow!("file is required"))?;
            ops::new_project(std::path::Path::new(file), args["name"].as_str().map(String::from))
        }
        "cadette_info" => {
            let detail = match &args["detail"] {
                Value::Null => Detail::None,
                Value::String(s) if s == "all" => Detail::All,
                Value::Array(a) => Detail::Ids(a.iter().filter_map(|v| v.as_u64().map(FeatureId)).collect()),
                other => return Err(anyhow!("detail must be a list of ids or \"all\", not {other}")),
            };
            ops::info(&target(args), detail)
        }
        "cadette_check" => ops::check(&target(args)),
        "cadette_apply" => {
            let edits = args.get("edits").cloned().ok_or_else(|| anyhow!("edits is required"))?;
            let mut batch = ops::parse_batch(edits)?;
            if let Some(label) = args["label"].as_str() {
                batch.label = Some(label.to_string());
            }
            ops::apply(
                &target(args),
                batch,
                args["out"].as_str().map(PathBuf::from),
                args["dry_run"].as_bool().unwrap_or(false),
                args["save"].as_bool().unwrap_or(false),
            )
        }
        "cadette_render" => {
            let render = RenderArgs {
                view: match args["view"].as_str() {
                    Some(v) => View::parse(v)?,
                    None => View::Iso,
                },
                yaw: args["yaw"].as_f64().map(|v| v as f32),
                pitch: args["pitch"].as_f64().map(|v| v as f32),
                width: args["width"].as_u64().unwrap_or(1280) as u32,
                height: args["height"].as_u64().unwrap_or(800) as u32,
                dark: args["dark"].as_bool().unwrap_or(false),
                highlight: args["highlight"].as_u64().map(FeatureId),
                grid: args["grid"].as_f64().map(|v| v as f32),
                as_seen: args["as_seen"].as_bool().unwrap_or(false),
            };
            let out = args["out"].as_str().map(PathBuf::from);
            ops::render(&target(args), out.as_deref(), &render)
        }
        "cadette_export_stl" => {
            let file = args["file"].as_str().ok_or_else(|| anyhow!("file is required"))?;
            let out = args["out"].as_str().ok_or_else(|| anyhow!("out is required"))?;
            ops::export_stl(std::path::Path::new(file), std::path::Path::new(out), args["deflection"].as_f64())
        }
        "cadette_state" => ops::state(&target(args)),
        "cadette_undo" => ops::undo(&target(args)),
        "cadette_redo" => ops::redo(&target(args)),
        "cadette_save" => ops::save(&target(args)),
        other => Err(anyhow!("unknown tool {other}")),
    }
}
