use wasm_bindgen::prelude::*;

/// Gauss Core version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
