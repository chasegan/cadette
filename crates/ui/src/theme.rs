//! Look-and-feel: the original **dark** theme and an on-brand **light** theme
//! built from the Cadette brand palette.
//!
//! Only the egui chrome (the panels around the viewport) is themed. The 3D view
//! renders behind a transparent `CentralPanel`, so switching themes never
//! touches the scene — the viewport keeps its dark studio look in both modes.

use egui::{Color32, Stroke, Visuals};
use serde::{Deserialize, Serialize};

// --- Cadette brand palette (from the brand guide) ------------------------

/// Eggshell — primary background.
pub const EGGSHELL: Color32 = Color32::from_rgb(0xF3, 0xEC, 0xDC);
/// Cloud — cards & surfaces.
pub const CLOUD: Color32 = Color32::from_rgb(0xFC, 0xF8, 0xEF);
/// Deep Space — ink & dark fields.
pub const DEEP_SPACE: Color32 = Color32::from_rgb(0x18, 0x22, 0x41);
/// Atomic Tangerine — primary accent.
pub const ATOMIC_TANGERINE: Color32 = Color32::from_rgb(0xE8, 0x63, 0x3C);
/// Aqua Signal — positive / links.
pub const AQUA_SIGNAL: Color32 = Color32::from_rgb(0x2B, 0xA3, 0x9C);
/// Beacon Gold — stars & highlights.
pub const BEACON_GOLD: Color32 = Color32::from_rgb(0xE6, 0xA9, 0x2E);
/// Eggshell, nudged darker — zebra striping / faint fills against `EGGSHELL`.
const EGGSHELL_DIM: Color32 = Color32::from_rgb(0xEA, 0xE1, 0xCC);
/// The brand's error red (`Signal Red`), for problems on a light background.
const SIGNAL_RED: Color32 = Color32::from_rgb(0xC0, 0x39, 0x2B);

/// The active look-and-feel. Persisted in app preferences.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Theme {
    /// The original dark studio look (egui's default dark visuals).
    Dark,
    /// On-brand light look built from the Cadette palette.
    Light,
}

impl Default for Theme {
    fn default() -> Self {
        Theme::Light
    }
}

impl Theme {
    /// The other theme (for a toggle).
    pub fn toggled(self) -> Self {
        match self {
            Theme::Dark => Theme::Light,
            Theme::Light => Theme::Dark,
        }
    }

    pub fn is_dark(self) -> bool {
        matches!(self, Theme::Dark)
    }

    /// The egui visuals for this theme. `Dark` is egui's stock dark visuals —
    /// the app's original look, unchanged.
    pub fn visuals(self) -> Visuals {
        match self {
            Theme::Dark => Visuals::dark(),
            Theme::Light => light_visuals(),
        }
    }

    /// Accent colors for panel-drawn text/marks that must read against this
    /// theme's background (errors, positive states, highlights). Widget chrome
    /// itself comes from [`Self::visuals`]; this is only for our explicit
    /// `RichText::color(..)` callers.
    pub fn palette(self) -> Palette {
        match self {
            Theme::Dark => Palette {
                positive: Color32::from_rgb(120, 200, 120),
                warn: Color32::from_rgb(230, 180, 60),
                error: Color32::from_rgb(232, 92, 92),
            },
            Theme::Light => Palette {
                positive: AQUA_SIGNAL,
                warn: BEACON_GOLD,
                error: SIGNAL_RED,
            },
        }
    }
}

/// Theme-resolved accent colors for panel content (see [`Theme::palette`]).
#[derive(Clone, Copy)]
pub struct Palette {
    /// Positive / success (e.g. "fully constrained").
    pub positive: Color32,
    /// Warnings / highlights.
    pub warn: Color32,
    /// Errors / problems.
    pub error: Color32,
}

/// The on-brand light visuals: Eggshell panels, Cloud surfaces, Deep Space ink,
/// Atomic Tangerine accents.
fn light_visuals() -> Visuals {
    let mut v = Visuals::light();
    let ink = DEEP_SPACE;

    // Surfaces.
    v.panel_fill = EGGSHELL;
    v.window_fill = CLOUD;
    v.extreme_bg_color = CLOUD; // text-edit / slider troughs
    v.faint_bg_color = EGGSHELL_DIM; // zebra striping
    v.code_bg_color = CLOUD;
    v.window_stroke = Stroke::new(1.0, ink.gamma_multiply(0.15));

    // Accents.
    v.hyperlink_color = ATOMIC_TANGERINE;
    v.selection.bg_fill = ATOMIC_TANGERINE.gamma_multiply(0.35);
    v.selection.stroke = Stroke::new(1.0, ink);

    let hairline = |a: f32| Stroke::new(1.0, ink.gamma_multiply(a));

    // Non-interactive: labels, headings, separators — ink on Eggshell.
    v.widgets.noninteractive.bg_fill = EGGSHELL;
    v.widgets.noninteractive.weak_bg_fill = EGGSHELL;
    v.widgets.noninteractive.bg_stroke = hairline(0.12);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, ink);

    // Inactive: resting buttons — Cloud cards with ink glyphs.
    v.widgets.inactive.bg_fill = CLOUD;
    v.widgets.inactive.weak_bg_fill = CLOUD;
    v.widgets.inactive.bg_stroke = hairline(0.12);
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, ink);

    // Hovered: a subtle tangerine lift.
    v.widgets.hovered.bg_fill = ATOMIC_TANGERINE.gamma_multiply(0.18);
    v.widgets.hovered.weak_bg_fill = ATOMIC_TANGERINE.gamma_multiply(0.18);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, ATOMIC_TANGERINE.gamma_multiply(0.6));
    v.widgets.hovered.fg_stroke = Stroke::new(1.0, ink);

    // Active / pressed: solid tangerine. `fg_stroke` doubles as egui's
    // `strong_text_color` (used by every `.strong()` label), so it must stay
    // ink — a light glyph here would turn all headings near-invisible on the
    // Eggshell panels. Ink reads fine on the tangerine press state too.
    v.widgets.active.bg_fill = ATOMIC_TANGERINE;
    v.widgets.active.weak_bg_fill = ATOMIC_TANGERINE;
    v.widgets.active.bg_stroke = Stroke::new(1.0, ATOMIC_TANGERINE);
    v.widgets.active.fg_stroke = Stroke::new(1.0, ink);

    // Open combo/menu mirrors hovered.
    v.widgets.open = v.widgets.hovered;

    v
}
