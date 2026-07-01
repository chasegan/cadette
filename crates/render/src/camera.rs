//! Orbit camera for the viewport.
//!
//! The model space is **Z-up** (matching the kernel: a cube sits in `0..40` on
//! Z). The camera orbits a `target` point at a given `distance`, parameterized
//! by `yaw`/`pitch`. Left-drag orbits, middle/right-drag pans, scroll dollies.

use glam::{Mat4, Vec3};

/// Pitch is clamped just shy of the poles to keep `look_at` well-defined when
/// the view direction would otherwise align with the Z-up vector.
const PITCH_LIMIT: f32 = 1.55; // ~88.8°

#[derive(Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub fov_y: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl OrbitCamera {
    /// Frame a model given its bounding sphere (`center`, `radius`) from a
    /// pleasant three-quarter view.
    pub fn framing(center: Vec3, radius: f32) -> Self {
        let radius = radius.max(1e-3);
        Self {
            target: center,
            yaw: -0.9,
            pitch: 0.6,
            distance: radius * 3.0,
            fov_y: 45f32.to_radians(),
            znear: radius * 0.01,
            zfar: radius * 100.0,
        }
    }

    /// World-space camera position.
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let dir = Vec3::new(cp * cy, cp * sy, sp);
        self.target + dir * self.distance
    }

    /// Combined view-projection matrix for the given viewport aspect ratio.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Z);
        // Depth range adapts to the orbit distance (rather than the wide
        // framing range) to keep good depth-buffer precision — important for
        // z-fighting and for reconstructing world points from depth.
        let znear = (self.distance * 0.05).max(1e-3);
        let zfar = self.distance * 10.0;
        let proj = Mat4::perspective_rh(self.fov_y, aspect.max(1e-3), znear, zfar);
        proj * view
    }

    /// Orbit by mouse deltas (radians-ish; scaled by the caller).
    pub fn orbit(&mut self, dyaw: f32, dpitch: f32) {
        self.yaw -= dyaw;
        self.pitch = (self.pitch + dpitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Point the camera to look ALONG `-axis` at the target — i.e. the eye sits
    /// on the `axis` side (a ViewCube face-click snaps to an orthographic view).
    /// `axis` should be a unit world direction; at the Z poles the yaw is kept.
    pub fn look_along(&mut self, axis: Vec3) {
        let a = axis.normalize_or_zero();
        self.pitch = a.z.asin().clamp(-PITCH_LIMIT, PITCH_LIMIT);
        if a.x.hypot(a.y) > 1e-4 {
            self.yaw = a.y.atan2(a.x);
        }
    }

    /// Dolly toward/away from the target. Positive `amount` zooms in.
    pub fn dolly(&mut self, amount: f32) {
        let factor = (1.0 - amount * 0.1).clamp(0.2, 5.0);
        self.distance = (self.distance * factor).clamp(self.znear * 2.0, self.zfar * 0.5);
    }

    /// Pan the target in the camera's screen plane. Deltas are in pixels;
    /// scaled by distance so panning feels consistent at any zoom.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let forward = (self.target - self.eye()).normalize_or_zero();
        let right = forward.cross(Vec3::Z).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        let scale = self.distance * 0.0015;
        self.target += (-dx * right + dy * up) * scale;
    }
}
