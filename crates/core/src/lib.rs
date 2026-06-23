//! # rmf-core
//!
//! Pure domain logic with **no** dependency on the geometry kernel or GPU.
//! Features are described as *data* (e.g. `Extrude { profile, distance }`) so
//! the history engine can replay them against the kernel to regenerate
//! geometry. Keeping this crate kernel-free is what makes the model testable.
//!
//! Modules here are intentionally thin seeds; they grow through Phase 1.

pub mod units;
