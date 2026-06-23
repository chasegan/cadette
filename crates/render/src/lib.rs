//! # rmf-render
//!
//! The wgpu viewport: render graph, mesh buffers, camera, ray-cast picking,
//! gizmos, grid, and measurement overlays. Milestone B brings up a live window
//! ([`run`]) that displays kernel meshes with an orbit camera; later phases
//! split this into the scene/picking/overlay structure from the outline.

pub mod camera;
mod view;
mod viewer;

// Re-export egui so `rmf-ui` and the app share exactly this version.
pub use egui;

pub use camera::OrbitCamera;
pub use view::ViewContext;
pub use viewer::{run, screenshot, Controller, MeshData};

use bytemuck::{Pod, Zeroable};

/// A single render vertex: world-space position + normal. Matches the flat
/// arrays produced by `rmf_kernel::Mesh` (interleaved here for GPU upload).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// Interleave the kernel's parallel position/normal arrays into GPU vertices.
pub fn interleave(positions: &[f32], normals: &[f32]) -> Vec<Vertex> {
    debug_assert_eq!(positions.len(), normals.len());
    positions
        .chunks_exact(3)
        .zip(normals.chunks_exact(3))
        .map(|(p, n)| Vertex {
            position: [p[0], p[1], p[2]],
            normal: [n[0], n[1], n[2]],
        })
        .collect()
}
