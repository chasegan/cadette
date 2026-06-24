//! # rmf-interaction
//!
//! The interaction model from the project outline: a tool state machine driven
//! by selection context, plus snapping/inference. This is where Riemanifold's
//! "feel" lives. It depends only on `rmf-core`, so the selection/tool logic
//! stays pure and unit-testable, free of GPU or windowing concerns.
//!
//! Built up incrementally: this first piece is the [`Selection`] model. The
//! tool state machine and sketch-interaction logic migrate here next.

pub mod selection;

pub use selection::Selection;
