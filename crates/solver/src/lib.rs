//! # cdt-solver
//!
//! The 2D sketch constraint solver. A pure-Rust Levenberg-Marquardt
//! least-squares core ([`lm`]) drives constraint residuals to zero;
//! [`solve_sketch`] compiles a [`cdt_core::Sketch2d`] into that system and
//! writes the solved geometry back.
//!
//! Pure Rust, no FFI — the whole solver is unit-testable without the kernel.
//! The brief allows wrapping FreeCAD's planegcs later; this hand-rolled core is
//! the MVP, kept behind [`solve_sketch`] so the rest of the app never sees the
//! numerics.

pub mod lm;
mod sketch_solve;

pub use lm::{solve, Residual, SolveOptions, SolveReport};
pub use sketch_solve::{solve_sketch, SketchSolution};
