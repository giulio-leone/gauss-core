use gauss_core::agent::AgentOutput as RustAgentOutput;
use gauss_core::code_execution::{CodeExecutionConfig, SandboxConfig};
use gauss_core::message::Message as RustMessage;
use gauss_core::provider::ReasoningEffort;
use serde_json::json;

#[napi(object)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Option<serde_json::Value>,
}

#[napi(object)]
pub struct JsMessage {
    pub role: String,
    pub content: String,
}

#[napi(object)]
pub struct AgentOptions {
    pub instructions: Option<String>,
    pub max_steps: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
    pub seed: Option<f64>,
    pub stop_on_tool: Option<String>,
    pub output_schema: Option<serde_json::Value>,
    pub thinking_budget: Option<u32>,
    pub reasoning_effort: Option<String>,
    pub cache_control: Option<bool>,
    pub code_execution: Option<CodeExecutionOptions>,
    pub grounding: Option<bool>,
    pub native_code_execution: Option<bool>,
    pub response_modalities: Option<Vec<String>>,
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            instructions: None,
            max_steps: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            seed: None,
            stop_on_tool: None,
            output_schema: None,
            thinking_budget: None,
            reasoning_effort: None,
            cache_control: None,
            code_execution: None,
            grounding: None,
            native_code_execution: None,
            response_modalities: None,
        }
    }
}

#[napi(object)]
pub struct CodeExecutionOptions {
    pub python: Option<bool>,
    pub javascript: Option<bool>,
    pub bash: Option<bool>,
    pub timeout_secs: Option<u32>,
    pub working_dir: Option<String>,
    pub sandbox: Option<String>,
    pub unified: Option<bool>,
}

#[napi(object)]
pub struct NapiCitation {
    pub citation_type: String,
    pub cited_text: Option<String>,
    pub document_title: Option<String>,
    pub start: Option<u32>,
    pub end: Option<u32>,
}

#[napi(object)]
pub struct AgentResult {
    pub text: String,
    pub steps: u32,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub structured_output: Option<serde_json::Value>,
    pub thinking: Option<String>,
    pub citations: Vec<NapiCitation>,
    pub grounding_metadata: Option<serde_json::Value>,
}

pub fn js_message_to_rust(msg: &JsMessage) -> RustMessage {
    match msg.role.as_str() {
        "system" => RustMessage::system(&msg.content),
        "assistant" => RustMessage::assistant(&msg.content),
        "user" => RustMessage::user(&msg.content),
        _ => RustMessage::user(&msg.content),
    }
}

pub fn rust_citations_to_js(citations: &[gauss_core::Citation]) -> Vec<NapiCitation> {
    citations
        .iter()
        .map(|c| NapiCitation {
            citation_type: c.citation_type.clone(),
            cited_text: Some(c.cited_text.clone()),
            document_title: c.document_title.clone(),
            start: c.start.map(|v| v as u32),
            end: c.end.map(|v| v as u32),
        })
        .collect()
}

pub fn rust_output_to_js(output: RustAgentOutput) -> AgentResult {
    let citations = rust_citations_to_js(&output.citations);
    let grounding = if output.grounding_metadata.is_empty() {
        None
    } else {
        Some(serde_json::to_value(&output.grounding_metadata).unwrap_or(json!([])))
    };
    AgentResult {
        text: output.text,
        steps: output.steps as u32,
        input_tokens: output.usage.input_tokens as u32,
        output_tokens: output.usage.output_tokens as u32,
        structured_output: output.structured_output,
        thinking: output.thinking,
        citations,
        grounding_metadata: grounding,
    }
}

pub fn napi_code_exec_to_config(opts: &CodeExecutionOptions) -> CodeExecutionConfig {
    let sandbox = match opts.sandbox.as_deref() {
        Some("strict") => SandboxConfig::strict(),
        Some("permissive") => SandboxConfig::permissive(),
        _ => SandboxConfig::default(),
    };
    CodeExecutionConfig {
        python: opts.python.unwrap_or(true),
        javascript: opts.javascript.unwrap_or(true),
        bash: opts.bash.unwrap_or(true),
        timeout: std::time::Duration::from_secs(opts.timeout_secs.unwrap_or(30) as u64),
        working_dir: opts.working_dir.clone(),
        env: Vec::new(),
        sandbox,
        interpreters: std::collections::HashMap::new(),
    }
}

pub fn parse_reasoning_effort(s: &str) -> Option<ReasoningEffort> {
    match s.to_lowercase().as_str() {
        "low" => Some(ReasoningEffort::Low),
        "medium" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        _ => None,
    }
}

pub fn apply_code_execution(
    mut builder: gauss_core::agent::AgentBuilder,
    ce: &CodeExecutionOptions,
) -> gauss_core::agent::AgentBuilder {
    let config = napi_code_exec_to_config(ce);
    if ce.unified.unwrap_or(false) {
        builder = builder.code_execution_unified(config);
    } else {
        builder = builder.code_execution(config);
    }
    builder
}
