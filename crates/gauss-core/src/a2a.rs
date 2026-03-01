//! Google A2A (Agent-to-Agent) Protocol — data model and types.
//!
//! Implements the core data structures for A2A protocol v0.3.0:
//! AgentCard, Task, Message, Part, Artifact, and all related types.

use serde::{Deserialize, Serialize};

// ── A2A Error Codes ──────────────────────────────────────────────────────────

pub const TASK_NOT_FOUND: i32 = -32001;
pub const CONTENT_TYPE_NOT_SUPPORTED: i32 = -32002;
pub const UNSUPPORTED_OPERATION: i32 = -32003;

// ── Agent Card ───────────────────────────────────────────────────────────────

/// Agent's "business card" served at `/.well-known/agent.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCard {
    pub name: String,
    pub description: String,
    pub url: String,
    pub version: String,
    #[serde(default = "default_protocol_version")]
    pub protocol_version: String,
    pub capabilities: AgentCapabilities,
    pub skills: Vec<AgentSkill>,
    pub default_input_modes: Vec<String>,
    pub default_output_modes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authentication: Option<AgentAuthentication>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Vec<Extension>>,
}

fn default_protocol_version() -> String {
    "0.3.0".to_string()
}

/// What the agent supports.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCapabilities {
    pub streaming: bool,
    pub push_notifications: bool,
    pub state_transition_history: bool,
}

/// A capability the agent offers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSkill {
    pub id: String,
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    pub examples: Vec<String>,
}

/// Authentication schemes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentAuthentication {
    pub schemes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials: Option<String>,
}

// ── Task ─────────────────────────────────────────────────────────────────────

/// Unit of work with lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Task {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_id: Option<String>,
    pub status: TaskStatus,
    pub messages: Vec<A2aMessage>,
    pub artifacts: Vec<Artifact>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Current status of a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskStatus {
    pub state: TaskState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<A2aMessage>,
    pub timestamp: String,
}

/// Lifecycle state of a task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskState {
    Submitted,
    Working,
    #[serde(rename = "input-required")]
    InputRequired,
    Completed,
    Canceled,
    Failed,
    Rejected,
}

// ── Message ──────────────────────────────────────────────────────────────────

/// Role in an A2A conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum A2aMessageRole {
    User,
    Agent,
}

/// A message exchanged between user and agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct A2aMessage {
    pub role: A2aMessageRole,
    pub parts: Vec<Part>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

// ── Part ─────────────────────────────────────────────────────────────────────

/// A content part within a message or artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Part {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "file")]
    File { file: FileContent },
    #[serde(rename = "data")]
    Data { data: serde_json::Value },
}

/// File content (inline bytes or URI reference).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
}

// ── Artifact ─────────────────────────────────────────────────────────────────

/// An output artifact produced by a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Artifact {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parts: Vec<Part>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub append: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_chunk: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

// ── Extension ────────────────────────────────────────────────────────────────

/// Protocol extension declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Extension {
    pub uri: String,
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

// ── JSON-RPC 2.0 ─────────────────────────────────────────────────────────────

/// JSON-RPC 2.0 request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    pub method: String,
    pub params: serde_json::Value,
}

/// JSON-RPC 2.0 response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

// ── Send Message ─────────────────────────────────────────────────────────────

/// Parameters for `message/send`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SendMessageRequest {
    pub message: A2aMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub configuration: Option<MessageSendConfiguration>,
}

/// Configuration for sending a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageSendConfiguration {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_output_modes: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocking: Option<bool>,
}

// ── Streaming Events ─────────────────────────────────────────────────────────

/// Streaming event for task status changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskStatusUpdateEvent {
    pub id: String,
    pub status: TaskStatus,
    #[serde(rename = "final")]
    pub final_: bool,
}

/// Streaming event for artifact updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskArtifactUpdateEvent {
    pub id: String,
    pub artifact: Artifact,
}

// ── Helper Implementations ───────────────────────────────────────────────────

impl JsonRpcResponse {
    /// Create a successful JSON-RPC response.
    pub fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Create an error JSON-RPC response.
    pub fn error(id: serde_json::Value, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }

    /// Create a "task not found" error response.
    pub fn task_not_found(id: serde_json::Value) -> Self {
        Self::error(id, TASK_NOT_FOUND, "Task not found")
    }

    /// Create an "unsupported operation" error response.
    pub fn unsupported_operation(id: serde_json::Value) -> Self {
        Self::error(id, UNSUPPORTED_OPERATION, "Unsupported operation")
    }
}

impl JsonRpcRequest {
    /// Create a new JSON-RPC request.
    pub fn new(id: serde_json::Value, method: impl Into<String>, params: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.into(),
            params,
        }
    }
}

impl A2aMessage {
    /// Create a text message from the user.
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: A2aMessageRole::User,
            parts: vec![Part::Text { text: text.into() }],
            metadata: None,
        }
    }

    /// Create a text message from the agent.
    pub fn agent_text(text: impl Into<String>) -> Self {
        Self {
            role: A2aMessageRole::Agent,
            parts: vec![Part::Text { text: text.into() }],
            metadata: None,
        }
    }
}

impl TaskStatus {
    /// Create a new TaskStatus with current timestamp placeholder.
    pub fn new(state: TaskState, timestamp: impl Into<String>) -> Self {
        Self {
            state,
            message: None,
            timestamp: timestamp.into(),
        }
    }
}

impl AgentCard {
    /// Create a minimal AgentCard with required fields.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        url: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            url: url.into(),
            version: version.into(),
            protocol_version: default_protocol_version(),
            capabilities: AgentCapabilities {
                streaming: false,
                push_notifications: false,
                state_transition_history: false,
            },
            skills: Vec::new(),
            default_input_modes: vec!["text".to_string()],
            default_output_modes: vec!["text".to_string()],
            authentication: None,
            extensions: None,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn roundtrip<T: Serialize + for<'de> Deserialize<'de>>(val: &T) -> T {
        let json = serde_json::to_string(val).expect("serialize");
        serde_json::from_str(&json).expect("deserialize")
    }

    #[test]
    fn test_task_state_serde() {
        assert_eq!(serde_json::to_string(&TaskState::Submitted).unwrap(), "\"submitted\"");
        assert_eq!(serde_json::to_string(&TaskState::Working).unwrap(), "\"working\"");
        assert_eq!(serde_json::to_string(&TaskState::InputRequired).unwrap(), "\"input-required\"");
        assert_eq!(serde_json::to_string(&TaskState::Completed).unwrap(), "\"completed\"");
        assert_eq!(serde_json::to_string(&TaskState::Canceled).unwrap(), "\"canceled\"");
        assert_eq!(serde_json::to_string(&TaskState::Failed).unwrap(), "\"failed\"");
        assert_eq!(serde_json::to_string(&TaskState::Rejected).unwrap(), "\"rejected\"");
    }

    #[test]
    fn test_task_state_deserialize() {
        let state: TaskState = serde_json::from_str("\"input-required\"").unwrap();
        assert_eq!(state, TaskState::InputRequired);
    }

    #[test]
    fn test_message_role_serde() {
        assert_eq!(serde_json::to_string(&A2aMessageRole::User).unwrap(), "\"user\"");
        assert_eq!(serde_json::to_string(&A2aMessageRole::Agent).unwrap(), "\"agent\"");
        let role: A2aMessageRole = serde_json::from_str("\"agent\"").unwrap();
        assert_eq!(role, A2aMessageRole::Agent);
    }

    #[test]
    fn test_part_text_serde() {
        let part = Part::Text { text: "hello".into() };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "hello");
        let back: Part = serde_json::from_value(json).unwrap();
        match back {
            Part::Text { text } => assert_eq!(text, "hello"),
            _ => panic!("expected Text part"),
        }
    }

    #[test]
    fn test_part_file_serde() {
        let part = Part::File {
            file: FileContent {
                name: Some("test.txt".into()),
                mime_type: Some("text/plain".into()),
                bytes: Some("aGVsbG8=".into()),
                uri: None,
            },
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "file");
        assert_eq!(json["file"]["name"], "test.txt");
        assert_eq!(json["file"]["mimeType"], "text/plain");
        let back: Part = serde_json::from_value(json).unwrap();
        match back {
            Part::File { file } => {
                assert_eq!(file.name.unwrap(), "test.txt");
                assert_eq!(file.mime_type.unwrap(), "text/plain");
            }
            _ => panic!("expected File part"),
        }
    }

    #[test]
    fn test_part_data_serde() {
        let part = Part::Data { data: json!({"key": "value"}) };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "data");
        assert_eq!(json["data"]["key"], "value");
    }

    #[test]
    fn test_message_roundtrip() {
        let msg = A2aMessage::user_text("Hello, agent!");
        let back = roundtrip(&msg);
        assert_eq!(back.role, A2aMessageRole::User);
        assert_eq!(back.parts.len(), 1);
    }

    #[test]
    fn test_agent_message_roundtrip() {
        let msg = A2aMessage::agent_text("I can help with that.");
        let back = roundtrip(&msg);
        assert_eq!(back.role, A2aMessageRole::Agent);
    }

    #[test]
    fn test_task_roundtrip() {
        let task = Task {
            id: "task-1".into(),
            context_id: Some("ctx-1".into()),
            status: TaskStatus::new(TaskState::Working, "2025-01-01T00:00:00Z"),
            messages: vec![A2aMessage::user_text("Do something")],
            artifacts: vec![],
            metadata: None,
        };
        let json = serde_json::to_value(&task).unwrap();
        assert_eq!(json["contextId"], "ctx-1");
        assert_eq!(json["status"]["state"], "working");
        let back: Task = serde_json::from_value(json).unwrap();
        assert_eq!(back.id, "task-1");
        assert_eq!(back.status.state, TaskState::Working);
    }

    #[test]
    fn test_artifact_roundtrip() {
        let artifact = Artifact {
            name: Some("result.txt".into()),
            description: Some("Output file".into()),
            parts: vec![Part::Text { text: "content".into() }],
            index: Some(0),
            append: Some(false),
            last_chunk: Some(true),
            metadata: None,
        };
        let json = serde_json::to_value(&artifact).unwrap();
        assert_eq!(json["lastChunk"], true);
        let back: Artifact = serde_json::from_value(json).unwrap();
        assert_eq!(back.name.unwrap(), "result.txt");
        assert_eq!(back.last_chunk, Some(true));
    }

    #[test]
    fn test_agent_card_roundtrip() {
        let card = AgentCard::new("TestAgent", "A test agent", "https://example.com/agent", "1.0.0");
        let json = serde_json::to_value(&card).unwrap();
        assert_eq!(json["protocolVersion"], "0.3.0");
        assert_eq!(json["defaultInputModes"][0], "text");
        let back: AgentCard = serde_json::from_value(json).unwrap();
        assert_eq!(back.name, "TestAgent");
        assert_eq!(back.protocol_version, "0.3.0");
    }

    #[test]
    fn test_agent_card_with_skills() {
        let mut card = AgentCard::new("SkillAgent", "desc", "https://a.com", "1.0");
        card.skills.push(AgentSkill {
            id: "translate".into(),
            name: "Translation".into(),
            description: "Translates text".into(),
            tags: vec!["nlp".into()],
            examples: vec!["Translate hello to French".into()],
        });
        card.capabilities.streaming = true;
        let back = roundtrip(&card);
        assert_eq!(back.skills.len(), 1);
        assert_eq!(back.skills[0].id, "translate");
        assert!(back.capabilities.streaming);
    }

    #[test]
    fn test_json_rpc_request_roundtrip() {
        let req = JsonRpcRequest::new(json!(1), "message/send", json!({"text": "hi"}));
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["jsonrpc"], "2.0");
        assert_eq!(json["method"], "message/send");
        let back: JsonRpcRequest = serde_json::from_value(json).unwrap();
        assert_eq!(back.method, "message/send");
    }

    #[test]
    fn test_json_rpc_success_response() {
        let resp = JsonRpcResponse::success(json!(1), json!({"status": "ok"}));
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["jsonrpc"], "2.0");
        assert!(json["result"].is_object());
        assert!(json.get("error").is_none());
    }

    #[test]
    fn test_json_rpc_error_response() {
        let resp = JsonRpcResponse::task_not_found(json!(42));
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["error"]["code"], TASK_NOT_FOUND);
        assert_eq!(json["error"]["message"], "Task not found");
        assert!(json.get("result").is_none());
    }

    #[test]
    fn test_send_message_request_roundtrip() {
        let req = SendMessageRequest {
            message: A2aMessage::user_text("test"),
            configuration: Some(MessageSendConfiguration {
                accepted_output_modes: Some(vec!["text".into()]),
                history_length: Some(10),
                blocking: Some(true),
            }),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["configuration"]["historyLength"], 10);
        assert_eq!(json["configuration"]["blocking"], true);
        let back: SendMessageRequest = serde_json::from_value(json).unwrap();
        assert_eq!(back.configuration.unwrap().history_length, Some(10));
    }

    #[test]
    fn test_task_status_update_event() {
        let evt = TaskStatusUpdateEvent {
            id: "task-1".into(),
            status: TaskStatus::new(TaskState::Completed, "2025-01-01T12:00:00Z"),
            final_: true,
        };
        let json = serde_json::to_value(&evt).unwrap();
        assert_eq!(json["final"], true);
        assert_eq!(json["status"]["state"], "completed");
        let back: TaskStatusUpdateEvent = serde_json::from_value(json).unwrap();
        assert!(back.final_);
    }

    #[test]
    fn test_task_artifact_update_event() {
        let evt = TaskArtifactUpdateEvent {
            id: "task-1".into(),
            artifact: Artifact {
                name: Some("chunk".into()),
                description: None,
                parts: vec![Part::Text { text: "partial".into() }],
                index: Some(0),
                append: Some(true),
                last_chunk: Some(false),
                metadata: None,
            },
        };
        let json = serde_json::to_value(&evt).unwrap();
        assert_eq!(json["artifact"]["append"], true);
        assert_eq!(json["artifact"]["lastChunk"], false);
        let back: TaskArtifactUpdateEvent = serde_json::from_value(json).unwrap();
        assert_eq!(back.artifact.append, Some(true));
    }

    #[test]
    fn test_extension_roundtrip() {
        let ext = Extension {
            uri: "https://example.com/ext/v1".into(),
            required: true,
            metadata: Some(json!({"version": "1.0"})),
        };
        let back = roundtrip(&ext);
        assert_eq!(back.uri, "https://example.com/ext/v1");
        assert!(back.required);
    }

    #[test]
    fn test_authentication_roundtrip() {
        let auth = AgentAuthentication {
            schemes: vec!["bearer".into(), "oauth2".into()],
            credentials: None,
        };
        let back = roundtrip(&auth);
        assert_eq!(back.schemes.len(), 2);
        assert!(back.credentials.is_none());
    }

    #[test]
    fn test_file_content_skip_none_fields() {
        let fc = FileContent {
            name: Some("doc.pdf".into()),
            mime_type: None,
            bytes: None,
            uri: Some("https://example.com/doc.pdf".into()),
        };
        let json = serde_json::to_value(&fc).unwrap();
        assert!(json.get("mimeType").is_none());
        assert!(json.get("bytes").is_none());
        assert_eq!(json["uri"], "https://example.com/doc.pdf");
    }

    #[test]
    fn test_error_codes() {
        assert_eq!(TASK_NOT_FOUND, -32001);
        assert_eq!(CONTENT_TYPE_NOT_SUPPORTED, -32002);
        assert_eq!(UNSUPPORTED_OPERATION, -32003);
    }
}
