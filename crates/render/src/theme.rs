//! The viewport's color scheme, so the 3D view can follow the app's theme.

/// Colors that drive the 3D viewport's look, so it can follow the app's
/// light/dark theme. Supplied per frame by [`Controller::viewport_theme`];
/// everything downstream (background gradient, material/edge shading, grid,
/// ViewCube) reads from here rather than baked-in constants.
///
/// [`Controller::viewport_theme`]: crate::Controller::viewport_theme
#[derive(Clone, Copy, Debug)]
pub struct ViewportTheme {
    /// Background gradient: top and bottom colors (linear RGB, 0..1).
    pub bg_top: [f32; 3],
    pub bg_bottom: [f32; 3],
    /// The solid material's base (unlit) color.
    pub face_base: [f32; 3],
    /// Feature-edge line color.
    pub edge: [f32; 3],
    /// Grid line colors, RGBA (alpha-blended over the background).
    pub grid_minor: [f32; 4],
    pub grid_major: [f32; 4],
    /// ViewCube face fill, hovered-face fill, edge, and label colors (sRGB u8).
    pub view_cube_face: [u8; 3],
    pub view_cube_face_hover: [u8; 3],
    pub view_cube_edge: [u8; 3],
    pub view_cube_label: [u8; 3],
}

impl ViewportTheme {
    /// The original dark studio look (unchanged from before theming).
    pub const DARK: Self = Self {
        bg_top: [0.09, 0.10, 0.12],
        bg_bottom: [0.09, 0.10, 0.12],
        face_base: [0.62, 0.66, 0.72],
        edge: [0.13, 0.15, 0.18],
        grid_minor: [0.58, 0.60, 0.66, 0.16],
        grid_major: [0.62, 0.64, 0.70, 0.38],
        view_cube_face: [206, 209, 214],
        view_cube_face_hover: [120, 165, 235],
        view_cube_edge: [90, 90, 90],
        view_cube_label: [40, 40, 40],
    };

    /// On-brand light look: a warm Cloud→Eggshell gradient, the same neutral-gray
    /// material, and Deep Space-tinted grid/ViewCube so they read on the light
    /// background. RGB channels are LINEAR (the sRGB target re-encodes on write),
    /// so the brand hexes are converted from sRGB here.
    pub const LIGHT: Self = Self {
        bg_top: [0.973, 0.939, 0.863],    // Cloud #FCF8EF (linear)
        bg_bottom: [0.776, 0.701, 0.552], // dimmed Eggshell ~#E4DAC4 (linear)
        face_base: [0.62, 0.66, 0.72],    // keep the neutral material
        edge: [0.13, 0.15, 0.18],         // dark edges read on light too
        grid_minor: [0.009, 0.016, 0.054, 0.20], // Deep Space (linear)
        grid_major: [0.009, 0.016, 0.054, 0.42],
        view_cube_face: [200, 192, 172],     // warm gray (sRGB u8 for egui)
        view_cube_face_hover: [232, 99, 60], // Atomic Tangerine
        view_cube_edge: [24, 34, 65],        // Deep Space
        view_cube_label: [24, 34, 65],
    };
}
