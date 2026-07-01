//! Persistent application preferences: a small JSON file in the platform's
//! per-user config directory. Currently just the light/dark theme; kept
//! deliberately tiny and best-effort (a missing or unreadable file falls back
//! to defaults, and save failures are logged, never fatal).

use std::path::PathBuf;

use cdt_ui::Theme;
use serde::{Deserialize, Serialize};

/// User preferences, serialized to `prefs.json`. `#[serde(default)]` so adding
/// fields later stays backward-compatible with older files.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Prefs {
    /// The active look-and-feel.
    pub theme: Theme,
}

/// The per-user config directory for Cadette, creating it if needed. `None` if
/// no home/config location could be determined.
fn config_dir() -> Option<PathBuf> {
    let base = if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
    } else if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
    };
    let dir = base?.join("Cadette");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log::warn!("could not create config dir {}: {e}", dir.display());
        return None;
    }
    Some(dir)
}

fn prefs_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("prefs.json"))
}

/// Load preferences, falling back to defaults if the file is absent or invalid.
pub fn load() -> Prefs {
    let Some(path) = prefs_path() else {
        return Prefs::default();
    };
    match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
            log::warn!("ignoring malformed prefs {}: {e}", path.display());
            Prefs::default()
        }),
        Err(_) => Prefs::default(), // first run: no file yet
    }
}

/// Persist preferences. Best-effort: failures are logged, not propagated.
pub fn save(prefs: &Prefs) {
    let Some(path) = prefs_path() else {
        return;
    };
    match serde_json::to_string_pretty(prefs) {
        Ok(text) => {
            if let Err(e) = std::fs::write(&path, text) {
                log::warn!("could not write prefs {}: {e}", path.display());
            }
        }
        Err(e) => log::warn!("could not serialize prefs: {e}"),
    }
}
