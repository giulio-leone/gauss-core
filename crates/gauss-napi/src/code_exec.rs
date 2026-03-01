use crate::provider::get_provider;
use crate::types::*;
use gauss_core::code_execution::{CodeExecutionConfig, CodeExecutionOrchestrator, SandboxConfig};
use gauss_core::message::Message as RustMessage;
use gauss_core::provider::GenerateOptions;
use gauss_core::tool::Tool as RustTool;
use napi::bindgen_prelude::*;
use serde_json::json;

/// Execute code in a specific language runtime.
#[napi]
pub async fn execute_code(
    language: String,
    code: String,
    timeout_secs: Option<u32>,
    working_dir: Option<String>,
    sandbox: Option<String>,
) -> Result<serde_json::Value> {
    let sandbox_config = match sandbox.as_deref() {
        Some("strict") => SandboxConfig::strict(),
        Some("permissive") => SandboxConfig::permissive(),
        _ => SandboxConfig::default(),
    };

    let config = CodeExecutionConfig {
        python: language == "python",
        javascript: language == "javascript",
        bash: language == "bash",
        timeout: std::time::Duration::from_secs(timeout_secs.unwrap_or(30) as u64),
        working_dir,
        env: Vec::new(),
        sandbox: sandbox_config,
        interpreters: std::collections::HashMap::new(),
    };

    let orch = CodeExecutionOrchestrator::new(config);
    let result = orch
        .execute(&language, &code)
        .await
        .map_err(|e| napi::Error::from_reason(format!("Code execution error: {e}")))?;

    Ok(json!({
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exitCode": result.exit_code,
        "timedOut": result.timed_out,
        "runtime": result.runtime,
        "success": result.success(),
    }))
}

/// Check which code runtimes are available on this system.
#[napi]
pub async fn available_runtimes() -> Result<Vec<String>> {
    let config = CodeExecutionConfig::all();
    let orch = CodeExecutionOrchestrator::new(config);
    Ok(orch.available_runtimes().await)
}

/// Generate an image using the provider's image generation API.
#[napi]
pub async fn generate_image(
    provider_handle: u32,
    prompt: String,
    model: Option<String>,
    size: Option<String>,
    quality: Option<String>,
    style: Option<String>,
    aspect_ratio: Option<String>,
    n: Option<u32>,
    response_format: Option<String>,
) -> Result<serde_json::Value> {
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
        .map_err(|e| napi::Error::from_reason(format!("Image generation error: {e}")))?;

    let images: Vec<serde_json::Value> = result
        .images
        .iter()
        .map(|img| {
            json!({
                "url": img.url,
                "base64": img.base64,
                "mimeType": img.mime_type,
            })
        })
        .collect();

    Ok(json!({
        "images": images,
        "revisedPrompt": result.revised_prompt,
    }))
}

/// Call a provider directly (without agent loop).
#[napi]
pub async fn generate(
    provider_handle: u32,
    messages: Vec<JsMessage>,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    thinking_budget: Option<u32>,
    reasoning_effort: Option<String>,
    cache_control: Option<bool>,
) -> Result<serde_json::Value> {
    let provider = get_provider(provider_handle)?;
    let rust_msgs: Vec<RustMessage> = messages.iter().map(js_message_to_rust).collect();

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
        .map_err(|e| napi::Error::from_reason(format!("Generate error: {e}")))?;

    let text = result.text().unwrap_or("").to_string();

    let citations_json: Vec<serde_json::Value> = result.citations.iter().map(|c| json!({
        "type": c.citation_type,
        "citedText": c.cited_text,
        "documentTitle": c.document_title,
        "start": c.start,
        "end": c.end,
    })).collect();

    Ok(json!({
        "text": text,
        "thinking": result.thinking,
        "citations": citations_json,
        "groundingMetadata": result.grounding_metadata,
        "usage": {
            "inputTokens": result.usage.input_tokens,
            "outputTokens": result.usage.output_tokens,
            "cacheReadTokens": result.usage.cache_read_tokens,
            "cacheCreationTokens": result.usage.cache_creation_tokens,
        },
        "finishReason": format!("{:?}", result.finish_reason),
    }))
}

/// Call a provider with tool definitions.
#[napi]
pub async fn generate_with_tools(
    provider_handle: u32,
    messages: Vec<JsMessage>,
    tools: Vec<ToolDef>,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    thinking_budget: Option<u32>,
    reasoning_effort: Option<String>,
) -> Result<serde_json::Value> {
    let provider = get_provider(provider_handle)?;
    let rust_msgs: Vec<RustMessage> = messages.iter().map(js_message_to_rust).collect();

    let rust_tools: Vec<RustTool> = tools
        .iter()
        .map(|td| {
            let mut tb = RustTool::builder(&td.name, &td.description);
            if let Some(ref params) = td.parameters {
                tb = tb.parameters_json(params.clone());
            }
            tb.build()
        })
        .collect();

    let parsed_effort = reasoning_effort.as_deref().and_then(parse_reasoning_effort);

    let opts = GenerateOptions {
        temperature,
        max_tokens,
        thinking_budget,
        reasoning_effort: parsed_effort,
        ..GenerateOptions::default()
    };

    let result = provider
        .generate(&rust_msgs, &rust_tools, &opts)
        .await
        .map_err(|e| napi::Error::from_reason(format!("Generate error: {e}")))?;

    let text = result.text().unwrap_or("").to_string();
    let tool_calls: Vec<serde_json::Value> = result
        .message
        .tool_calls()
        .into_iter()
        .map(|(id, name, args)| {
            json!({
                "id": id,
                "name": name,
                "args": args,
            })
        })
        .collect();

    Ok(json!({
        "text": text,
        "toolCalls": tool_calls,
        "usage": {
            "inputTokens": result.usage.input_tokens,
            "outputTokens": result.usage.output_tokens,
        },
        "finishReason": format!("{:?}", result.finish_reason),
    }))
}
