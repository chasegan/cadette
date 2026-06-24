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

/// A single render vertex: world-space position + normal + source face id.
/// Matches the flat arrays produced by `rmf_kernel::Mesh`. The `face_id` drives
/// GPU picking and face highlighting.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub face_id: u32,
}

/// Interleave the kernel's parallel position/normal/face-id arrays into GPU
/// vertices.
pub fn interleave(positions: &[f32], normals: &[f32], face_ids: &[u32]) -> Vec<Vertex> {
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
