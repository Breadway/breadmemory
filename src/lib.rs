//! breadmemory: local-first ChatGPT archive and current-context memory.
//!
//! The binary in `main.rs` is a thin CLI over these modules.

pub mod chunk;
pub mod context;
pub mod db;
pub mod export;
pub mod importer;
pub mod mcp;
pub mod render;
pub mod search;
pub mod text;
pub mod util;
