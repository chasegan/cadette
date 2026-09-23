//! Display meshes for the session's visible bodies, with the Build Graph
//! lane colours the app uses, and the per-body face/edge id ranges that let a
//! caller highlight one body's faces.

use cdt_core::{FeatureId, LANE_PALETTE};
use cdt_render::{interleave, interleave_edges, srgb_to_linear, MeshData};

use crate::Session;

/// Where one body's faces and edges sit in the combined mesh's id space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BodyRange {
    pub feature: FeatureId,
    /// Global face ids `[face_start, face_end)`.
    pub face_start: u32,
    pub face_end: u32,
    /// Global edge ids `[edge_start, edge_end)`.
    pub edge_start: u32,
    pub edge_end: u32,
}

/// The combined display mesh plus its body ranges.
#[derive(Clone, Default)]
pub struct SceneMesh {
    pub mesh: MeshData,
    pub bodies: Vec<BodyRange>,
    /// Bodies that could not be tessellated, with the kernel's message.
    pub failures: Vec<(FeatureId, String)>,
}

impl Session {
    /// Tessellate every visible body at `deflection` into one display mesh
    /// (face/edge ids offset to stay unique per body), coloured by the
    /// feature's colour or its Build Graph lane.
    pub fn scene_mesh(&mut self, deflection: f64) -> SceneMesh {
        let lane_of: std::collections::HashMap<FeatureId, usize> = self
            .document()
            .history
            .build_graph()
            .rows
            .iter()
            .map(|r| (r.id, r.lane))
            .collect();
        let visible: Vec<FeatureId> = self.regenerate().visible().to_vec();
        let doc = self.document();
        let colors: Vec<(FeatureId, [f32; 3])> = visible
            .iter()
            .map(|&id| {
                let color = doc.history.resolve_color(id).unwrap_or_else(|| {
                    let lane = lane_of.get(&id).copied().unwrap_or(0);
                    srgb_to_linear(LANE_PALETTE[lane % LANE_PALETTE.len()])
                });
                (id, color)
            })
            .collect();
        let mut out = SceneMesh::default();
        let (mut face_offset, mut edge_offset) = (0u32, 0u32);
        for (id, color) in colors {
            let Some(body) = self.body(id) else { continue };
            match body.tessellate(deflection) {
                Ok(part) => {
                    let mesh = &mut out.mesh;
                    let base = mesh.vertices.len() as u32;
                    let face_ids: Vec<u32> = part.face_ids.iter().map(|f| f + face_offset).collect();
                    mesh.vertices.extend(interleave(&part.positions, &part.normals, &face_ids, color));
                    mesh.indices.extend(part.indices.iter().map(|i| i + base));
                    let faces = part.face_ids.iter().copied().max().map_or(0, |m| m + 1);
                    let edge_base = mesh.edge_vertices.len() as u32;
                    let edge_ids: Vec<u32> = part.edge_ids.iter().map(|e| e + edge_offset).collect();
                    mesh.edge_vertices.extend(interleave_edges(&part.edge_positions, &edge_ids));
                    mesh.edge_indices.extend(part.edge_indices.iter().map(|i| i + edge_base));
                    let edges = part.edge_ids.iter().copied().max().map_or(0, |m| m + 1);
                    out.bodies.push(BodyRange {
                        feature: id,
                        face_start: face_offset,
                        face_end: face_offset + faces,
                        edge_start: edge_offset,
                        edge_end: edge_offset + edges,
                    });
                    face_offset += faces;
                    edge_offset += edges;
                }
                Err(e) => out.failures.push((id, e.to_string())),
            }
        }
        out
    }
}
