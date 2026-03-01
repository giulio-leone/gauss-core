use crate::registry::HandleRegistry;
use gauss_core::patterns::{CoercionStrategy, ToolValidator as RustToolValidator};
use gauss_core::plugin;
use gauss_core::tool::{Tool as RustTool, ToolExample as RustToolExample, ToolRegistry as RustToolRegistry};
use napi::bindgen_prelude::*;
use std::sync::Arc;

// ============ Plugin System ============

static PLUGIN_REGISTRIES: HandleRegistry<plugin::PluginRegistry> = HandleRegistry::new();

#[napi]
pub fn create_plugin_registry() -> u32 {
    PLUGIN_REGISTRIES.insert(plugin::PluginRegistry::new())
}

#[napi]
pub fn plugin_registry_add_telemetry(handle: u32) -> Result<()> {
    PLUGIN_REGISTRIES.with_mut(handle, |registry| {
        registry.register(Arc::new(plugin::TelemetryPlugin));
        Ok(())
    })
}

#[napi]
pub fn plugin_registry_add_memory(handle: u32) -> Result<()> {
    PLUGIN_REGISTRIES.with_mut(handle, |registry| {
        registry.register(Arc::new(plugin::MemoryPlugin));
        Ok(())
    })
}

#[napi]
pub fn plugin_registry_list(handle: u32) -> Result<Vec<String>> {
    PLUGIN_REGISTRIES.get(handle, |registry| {
        registry.list().into_iter().map(String::from).collect()
    })
}

#[napi]
pub fn plugin_registry_emit(handle: u32, event_json: String) -> Result<()> {
    let event: plugin::GaussEvent = serde_json::from_str(&event_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid event JSON: {e}")))?;
    PLUGIN_REGISTRIES.get(handle, |registry| {
        registry.emit(&event);
    })?;
    Ok(())
}

#[napi]
pub fn destroy_plugin_registry(handle: u32) -> Result<()> {
    PLUGIN_REGISTRIES.remove(handle)?;
    Ok(())
}

// ============ Tool Validator ============

static TOOL_VALIDATORS: HandleRegistry<RustToolValidator> = HandleRegistry::new();

#[napi]
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

#[napi]
pub fn tool_validator_validate(handle: u32, input: String, schema: String) -> Result<String> {
    TOOL_VALIDATORS.get(handle, |validator| {
        let input_val: serde_json::Value = serde_json::from_str(&input)
            .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
        let schema_val: serde_json::Value = serde_json::from_str(&schema)
            .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
        let result = validator
            .validate(input_val, &schema_val)
            .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
        serde_json::to_string(&result).map_err(|e| napi::Error::from_reason(format!("{e}")))
    })?
}

#[napi]
pub fn destroy_tool_validator(handle: u32) -> Result<()> {
    TOOL_VALIDATORS.remove(handle)?;
    Ok(())
}

// ============ Tool Registry ============

static TOOL_REGISTRIES: HandleRegistry<RustToolRegistry> = HandleRegistry::new();

#[napi]
pub fn create_tool_registry() -> u32 {
    TOOL_REGISTRIES.insert(RustToolRegistry::new())
}

#[napi]
pub fn tool_registry_add(handle: u32, tool_json: String) -> Result<()> {
    let v: serde_json::Value = serde_json::from_str(&tool_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid tool JSON: {e}")))?;
    let name = v["name"]
        .as_str()
        .ok_or_else(|| napi::Error::from_reason("Missing 'name'"))?
        .to_string();
    let description = v["description"]
        .as_str()
        .ok_or_else(|| napi::Error::from_reason("Missing 'description'"))?
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
    let mut builder = RustTool::builder(&name, &description).tags(tags);
    for ex in examples {
        builder = builder.example(ex);
    }
    let tool = builder.build();
    TOOL_REGISTRIES.with_mut(handle, |registry| {
        registry.register(tool);
        Ok(())
    })
}

#[napi]
pub fn tool_registry_search(handle: u32, query: String) -> Result<serde_json::Value> {
    TOOL_REGISTRIES.get(handle, |registry| {
        let results: Vec<serde_json::Value> = registry
            .search(&query)
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "tags": t.tags,
                })
            })
            .collect();
        serde_json::Value::Array(results)
    })
}

#[napi]
pub fn tool_registry_by_tag(handle: u32, tag: String) -> Result<serde_json::Value> {
    TOOL_REGISTRIES.get(handle, |registry| {
        let results: Vec<serde_json::Value> = registry
            .by_tag(&tag)
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "tags": t.tags,
                })
            })
            .collect();
        serde_json::Value::Array(results)
    })
}

#[napi]
pub fn tool_registry_list(handle: u32) -> Result<serde_json::Value> {
    TOOL_REGISTRIES.get(handle, |registry| {
        let results: Vec<serde_json::Value> = registry
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
            .collect();
        serde_json::Value::Array(results)
    })
}

#[napi]
pub fn destroy_tool_registry(handle: u32) -> Result<()> {
    TOOL_REGISTRIES.remove(handle)?;
    Ok(())
}
