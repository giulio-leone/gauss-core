use crate::registry::{py_err, HandleRegistry};
use gauss_core::patterns::{CoercionStrategy, ToolValidator as RustToolValidator};
use gauss_core::plugin;
use gauss_core::tool::{ToolExample as RustToolExample, ToolRegistry as RustToolRegistry};
use pyo3::prelude::*;
use std::sync::Arc;

// ============ Plugin System ============

static PLUGIN_REGISTRIES: HandleRegistry<plugin::PluginRegistry> = HandleRegistry::new();

#[pyfunction]
pub fn create_plugin_registry() -> u32 {
    PLUGIN_REGISTRIES.insert(plugin::PluginRegistry::new())
}

#[pyfunction]
pub fn plugin_registry_add_telemetry(handle: u32) -> PyResult<()> {
    PLUGIN_REGISTRIES.with_mut(handle, |registry| {
        registry.register(Arc::new(plugin::TelemetryPlugin));
        Ok(())
    })
}

#[pyfunction]
pub fn plugin_registry_add_memory(handle: u32) -> PyResult<()> {
    PLUGIN_REGISTRIES.with_mut(handle, |registry| {
        registry.register(Arc::new(plugin::MemoryPlugin));
        Ok(())
    })
}

#[pyfunction]
pub fn plugin_registry_list(handle: u32) -> PyResult<Vec<String>> {
    PLUGIN_REGISTRIES.get(handle, |registry| {
        registry.list().into_iter().map(String::from).collect()
    })
}

#[pyfunction]
pub fn plugin_registry_emit(handle: u32, event_json: String) -> PyResult<()> {
    let event: plugin::GaussEvent = serde_json::from_str(&event_json)
        .map_err(|e| py_err(format!("Invalid event JSON: {e}")))?;
    PLUGIN_REGISTRIES.get(handle, |registry| {
        registry.emit(&event);
    })?;
    Ok(())
}

#[pyfunction]
pub fn destroy_plugin_registry(handle: u32) -> PyResult<()> {
    PLUGIN_REGISTRIES.remove(handle)
}

// ============ Tool Validator ============

static TOOL_VALIDATORS: HandleRegistry<RustToolValidator> = HandleRegistry::new();

#[pyfunction]
#[pyo3(signature = (strategies=None))]
pub fn create_tool_validator(strategies: Option<Vec<String>>) -> u32 {
    let validator = match strategies {
        Some(strats) => {
            let parsed: Vec<CoercionStrategy> = strats
                .iter()
                .filter_map(|s| match s.as_str() {
                    "null_to_default" => Some(CoercionStrategy::NullToDefault),
                    "type_cast" => Some(CoercionStrategy::TypeCast),
                    "json_parse" => Some(CoercionStrategy::JsonParse),
                    "strip_null" => Some(CoercionStrategy::StripNull),
                    _ => None,
                })
                .collect();
            RustToolValidator::with_strategies(parsed)
        }
        None => RustToolValidator::new(),
    };
    TOOL_VALIDATORS.insert(validator)
}

#[pyfunction]
pub fn tool_validator_validate(handle: u32, input: String, schema: String) -> PyResult<String> {
    TOOL_VALIDATORS.get(handle, |validator| {
        let input_val: serde_json::Value =
            serde_json::from_str(&input).map_err(|e| py_err(format!("{e}")))?;
        let schema_val: serde_json::Value =
            serde_json::from_str(&schema).map_err(|e| py_err(format!("{e}")))?;
        let result = validator
            .validate(input_val, &schema_val)
            .map_err(|e| py_err(format!("{e}")))?;
        serde_json::to_string(&result).map_err(|e| py_err(format!("{e}")))
    })?
}

#[pyfunction]
pub fn destroy_tool_validator(handle: u32) -> PyResult<()> {
    TOOL_VALIDATORS.remove(handle)
}

// ============ Tool Registry ============

static TOOL_REGISTRIES: HandleRegistry<RustToolRegistry> = HandleRegistry::new();

#[pyfunction]
pub fn create_tool_registry() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(1);
    let handle = COUNTER.fetch_add(1, Ordering::Relaxed);
    TOOL_REGISTRIES
        .raw()
        .lock()
        .unwrap()
        .insert(handle, RustToolRegistry::new());
    handle
}

#[pyfunction]
pub fn tool_registry_add(handle: u32, tool_json: String) -> PyResult<()> {
    let v: serde_json::Value = serde_json::from_str(&tool_json).map_err(py_err)?;
    let name = v["name"]
        .as_str()
        .ok_or_else(|| py_err("Missing 'name'"))?
        .to_string();
    let description = v["description"]
        .as_str()
        .ok_or_else(|| py_err("Missing 'description'"))?
        .to_string();
    let tags: Vec<String> = v["tags"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|t| t.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let examples: Vec<RustToolExample> = v["examples"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    Some(RustToolExample {
                        description: e["description"].as_str()?.to_string(),
                        input: e["input"].clone(),
                        expected_output: e.get("expectedOutput").cloned(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let mut builder = gauss_core::tool::Tool::builder(&name, &description).tags(tags);
    for ex in examples {
        builder = builder.example(ex);
    }
    let tool = builder.build();
    TOOL_REGISTRIES.with_mut(handle, |registry| {
        registry.register(tool);
        Ok(())
    })
}

#[pyfunction]
pub fn tool_registry_search(handle: u32, query: String) -> PyResult<String> {
    let results = TOOL_REGISTRIES.get(handle, |registry| {
        registry
            .search(&query)
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "tags": t.tags,
                })
            })
            .collect::<Vec<serde_json::Value>>()
    })?;
    serde_json::to_string(&results).map_err(py_err)
}

#[pyfunction]
pub fn tool_registry_by_tag(handle: u32, tag: String) -> PyResult<String> {
    let results = TOOL_REGISTRIES.get(handle, |registry| {
        registry
            .by_tag(&tag)
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "tags": t.tags,
                })
            })
            .collect::<Vec<serde_json::Value>>()
    })?;
    serde_json::to_string(&results).map_err(py_err)
}

#[pyfunction]
pub fn tool_registry_list(handle: u32) -> PyResult<String> {
    let results = TOOL_REGISTRIES.get(handle, |registry| {
        registry
            .list()
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "tags": t.tags,
                    "examples": t.examples,
                })
            })
            .collect::<Vec<serde_json::Value>>()
    })?;
    serde_json::to_string(&results).map_err(py_err)
}

#[pyfunction]
pub fn destroy_tool_registry(handle: u32) -> PyResult<()> {
    TOOL_REGISTRIES.remove(handle)
}
