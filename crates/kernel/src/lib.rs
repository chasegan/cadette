//! # rmf-kernel
//!
//! The OpenCASCADE (OCCT) boundary for Riemanifold and the **only** crate that
//! crosses into C++. Downstream code never sees raw FFI: it works with the safe
//! [`Solid`] type and plain Rust [`Mesh`] data.
//!
//! - [`ffi`] — the raw `#[cxx::bridge]`. Treat as private plumbing.
//! - [`solids`] — safe, owned wrappers and the public modeling API.

pub mod ffi;
pub mod solids;

pub use solids::{Mesh, Solid};

/// Error raised when an OCCT operation fails. Wraps the message OCCT threw
/// across the FFI boundary (see `guard` in `bridge.cpp`).
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct KernelError(pub String);

impl From<cxx::Exception> for KernelError {
    fn from(e: cxx::Exception) -> Self {
        KernelError(e.what().to_string())
    }
}

/// Convenience alias for fallible kernel operations.
pub type Result<T> = std::result::Result<T, KernelError>;
