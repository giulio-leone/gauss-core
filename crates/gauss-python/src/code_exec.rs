use crate::provider::get_provider;
use crate::types::{parse_messages, parse_reasoning_effort};
use gauss_core::provider::GenerateOptions;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde_json::json;

/// Call generate. Returns JSON string.
#[pyfunction]
#[pyo3(signature = (provider_handle, messages_json, temperature=None, max_tokens=None, thinking_budget=None, reasoning_effort=None, cache_control=None))]
pub fn generate(
    py: Python<'_>,
    provider_handle: u32,
    messages_json: String,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    thinking_budget: Option<u32>,
    reasoning_effort: Option<String>,
    cache_control: Option<bool>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let provider = get_provider(provider_handle)?;
        let rust_msgs = parse_messages(&messages_json)?;

        let parsed_effort = reasoning_effort.as_deref().and_then(parse_reasoning_effort);

        let opts = GenerateOptions {
            temperature,
            max_tokens,
            thinking_budget,
            reasoning_effort: parsed_effort,
            cache_control: cache_control.unwrap_or(false),
            ..GenerateOptions::default()
        };

        let result = provider
            .generate(&rust_msgs, &[], &opts)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Generate error: {e}")))?;

        let text = result.text().unwrap_or("").to_string();

        let citations_json: Vec<serde_json::Value> = result
            .citations
            .iter()
            .map(|c| {
                json!({
                    "type": c.citation_type,
                    "cited_text": c.cited_text,
                    "document_title": c.document_title,
                    "start": c.start,
                    "end": c.end,
                })
            })
            .collect();

        let output = json!({
            "text": text,
            "thinking": result.thinking,
            "citations": citations_json,
            "usage": {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "cache_read_tokens": result.usage.cache_read_tokens,
                "cache_creation_tokens": result.usage.cache_creation_tokens,
            },
            "finish_reason": format!("{:?}", result.finish_reason),
        });

        serde_json::to_string(&output)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
    })
}

/// Stream generate. Returns JSON array of StreamEvent objects.
#[pyfunction]
#[pyo3(signature = (provider_handle, messages_json, temperature=None, max_tokens=None))]
pub fn stream_generate(
    py: Python<'_>,
    provider_handle: u32,
    messages_json: String,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        use futures::StreamExt;
        let provider = get_provider(provider_handle)?;
        let rust_msgs = parse_messages(&messages_json)?;

        let opts = GenerateOptions {
            temperature,
            max_tokens,
            ..GenerateOptions::default()
        };

        let mut stream = provider
            .stream(&rust_msgs, &[], &opts)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Stream error: {e}")))?;

        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            match event {
                Ok(e) => {
                    let json = serde_json::to_string(&e)
                        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
                    events.push(json);
                }
                Err(e) => {
                    return Err(PyRuntimeError::new_err(format!("Stream event error: {e}")));
                }
            }
        }

        let output = format!("[{}]", events.join(","));
        Ok(output)
    })
}

/// Execute code in a specific language runtime.
#[pyfunction]
#[pyo3(signature = (language, code, timeout_secs=None, working_dir=None, sandbox=None))]
pub fn execute_code(
    py: Python<'_>,
    language: String,
    code: String,
    timeout_secs: Option<u64>,
    working_dir: Option<String>,
    sandbox: Option<String>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let sandbox_config = match sandbox.as_deref() {
            Some("strict") => gauss_core::code_execution::SandboxConfig::strict(),
            Some("permissive") => gauss_core::code_execution::SandboxConfig::permissive(),
            _ => gauss_core::code_execution::SandboxConfig::default(),
        };

        let config = gauss_core::code_execution::CodeExecutionConfig {
            python: language == "python",
            javascript: language == "javascript",
            bash: language == "bash",
            timeout: std::time::Duration::from_secs(timeout_secs.unwrap_or(30)),
            working_dir,
            env: Vec::new(),
            sandbox: sandbox_config,
            interpreters: std::collections::HashMap::new(),
        };

        let orch = gauss_core::code_execution::CodeExecutionOrchestrator::new(config);
        let result = orch
            .execute(&language, &code)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Code execution error: {e}")))?;

        let result_json = json!({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "runtime": result.runtime,
            "success": result.success(),
        });

        serde_json::to_string(&result_json)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
    })
}

/// Check which code runtimes are available.
#[pyfunction]
pub fn available_runtimes(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let config = gauss_core::code_execution::CodeExecutionConfig::all();
        let orch = gauss_core::code_execution::CodeExecutionOrchestrator::new(config);
        let runtimes = orch.available_runtimes().await;
        serde_json::to_string(&runtimes)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
    })
}

/// Generate images using a provider's image generation API.
#[pyfunction]
#[pyo3(signature = (provider_handle, prompt, model=None, size=None, quality=None, style=None, aspect_ratio=None, n=None, response_format=None))]
pub fn generate_image(
    py: Python<'_>,
    provider_handle: u32,
    prompt: String,
    model: Option<String>,
    size: Option<String>,
    quality: Option<String>,
    style: Option<String>,
    aspect_ratio: Option<String>,
    n: Option<u32>,
    response_format: Option<String>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let provider = get_provider(provider_handle)?;

        let config = gauss_core::ImageGenerationConfig {
            model,
            size,
            quality,
            style,
            aspect_ratio,
            n,
            response_format,
        };

        let result = provider
            .generate_image(&prompt, &config)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Image generation error: {e}")))?;

        serde_json::to_string(&result)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
    })
}
