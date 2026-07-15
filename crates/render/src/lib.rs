//! # cdt-render
//!
//! The wgpu viewport: render graph, mesh buffers, camera, ray-cast picking,
//! gizmos, grid, and measurement overlays. Milestone B brings up a live window
//! ([`run`]) that displays kernel meshes with an orbit camera; later phases
//! split this into the scene/picking/overlay structure from the outline.

pub mod camera;
mod controller;
mod gizmo;
mod grid;
mod math;
mod picking;
mod scene;
mod screenshot;
mod theme;
mod view;
mod view_cube;
mod window;

// Re-export egui so `cdt-ui` and the app share exactly this version.
pub use egui;

pub use camera::{CameraPose, OrbitCamera};
pub use controller::{
    Axis3, Controller, Gizmo, GizmoHandle, GridSpec, Highlights, MeshData, Pick, Plane3,
    ResizeBox, TransformDelta,
};
pub use grid::min_grid_spacing;
pub use theme::ViewportTheme;
pub use view::ViewContext;
pub use screenshot::screenshot;
pub use window::run;

use bytemuck::{Pod, Zeroable};

/// A single render vertex: world-space position + normal + source face id.
/// Matches the flat arrays produced by `cdt_kernel::Mesh`. The `face_id` drives
/// GPU picking and face highlighting.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub face_id: u32,
    /// Per-body display colour (linear RGB). A negative red channel is a
    /// sentinel meaning "no colour set" — the shader then uses the theme's
    /// default material, so un-coloured bodies look exactly as before.
    pub color: [f32; 3],
}

/// Sentinel `color` marking a vertex with no user colour (use the theme default).
pub const NO_COLOR: [f32; 3] = [-1.0, -1.0, -1.0];

/// Convert an sRGB u8 colour (e.g. a Build Graph lane colour) to the linear
/// RGB floats the shader works in.
pub fn srgb_to_linear(c: [u8; 3]) -> [f32; 3] {
    let f = |v: u8| {
        let s = v as f32 / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    [f(c[0]), f(c[1]), f(c[2])]
}

/// Interleave the kernel's parallel position/normal/face-id arrays into GPU
/// vertices, tagging every vertex of this body with `color` (use [`NO_COLOR`]
/// for the theme default).
pub fn interleave(
    positions: &[f32],
    normals: &[f32],
    face_ids: &[u32],
    color: [f32; 3],
) -> Vec<Vertex> {
    debug_assert_eq!(positions.len(), normals.len());
    debug_assert_eq!(positions.len() / 3, face_ids.len());
    positions
        .chunks_exact(3)
        .zip(normals.chunks_exact(3))
        .zip(face_ids)
        .map(|((p, n), &face_id)| Vertex {
            position: [p[0], p[1], p[2]],
            normal: [n[0], n[1], n[2]],
            face_id,
            color,
        })
        .collect()
}

/// A crisp edge-line vertex: position + source edge id (for edge picking).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct EdgeVertex {
    pub position: [f32; 3],
    pub edge_id: u32,
}

/// A transform-gizmo vertex: world-space position + RGBA color. The gizmo is
/// drawn as colored triangles on top of the model; alpha lets the rotation
/// protractor wedge be translucent.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GizmoVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

/// Interleave the kernel's edge position/id arrays into edge-line vertices.
pub fn interleave_edges(positions: &[f32], edge_ids: &[u32]) -> Vec<EdgeVertex> {
    debug_assert_eq!(positions.len() / 3, edge_ids.len());
    positions
        .chunks_exact(3)
        .zip(edge_ids)
        .map(|(p, &edge_id)| EdgeVertex {
            position: [p[0], p[1], p[2]],
            edge_id,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleave_tags_every_vertex_with_the_body_colour() {
        let positions = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let normals = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let ids = [0u32, 0];
        let verts = interleave(&positions, &normals, &ids, [0.2, 0.4, 0.6]);
        assert_eq!(verts.len(), 2);
        assert!(verts.iter().all(|v| v.color == [0.2, 0.4, 0.6]));
        // The sentinel marks "no user colour" for the theme-default path.
        let plain = interleave(&positions, &normals, &ids, NO_COLOR);
        assert!(plain[0].color[0] < 0.0);
    }
}
