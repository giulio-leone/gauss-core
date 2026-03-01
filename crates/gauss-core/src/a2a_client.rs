//! A2A (Agent-to-Agent) client — connects to remote A2A agents via JSON-RPC 2.0 over HTTP.

use std::pin::Pin;
use std::time::Duration;

use futures::Stream;
use serde_json::json;

use crate::a2a::*;
use crate::a2a_server::{A2aStreamEvent, SendMessageResponse};
use crate::error::{GaussError, Result};

// ── Client ───────────────────────────────────────────────────────────────────

/// HTTP client for communicating with remote A2A agents.
pub struct A2aClient {
    http: reqwest::Client,
    base_url: String,
    auth_token: Option<String>,
}

impl A2aClient {
    /// Create a new client pointing at `base_url` (e.g. `"https://agent.example.com"`).
    pub fn new(base_url: &str) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            auth_token: None,
        }
    }

    /// Attach a bearer token for authenticated requests.
    pub fn with_auth_token(mut self, token: &str) -> Self {
        self.auth_token = Some(token.to_string());
        self
    }

    /// Override the default HTTP timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.http = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        self
    }

    // ── Discovery ────────────────────────────────────────────────────────

    /// Fetch the remote agent's [`AgentCard`] from `/.well-known/agent.json`.
    pub async fn discover(&self) -> Result<AgentCard> {
        let url = format!("{}/.well-known/agent.json", self.base_url);
        let mut req = self.http.get(&url);
        if let Some(token) = &self.auth_token {
            req = req.bearer_auth(token);
        }
        let resp = req.send().await.map_err(|e| {
            GaussError::provider("a2a", format!("Discovery request failed: {e}"))
        })?;
        if !resp.status().is_success() {
            return Err(GaussError::provider(
                "a2a",
                format!("Discovery failed with status {}", resp.status()),
            ));
        }
        resp.json::<AgentCard>().await.map_err(|e| {
            GaussError::provider("a2a", format!("Failed to parse AgentCard: {e}"))
        })
    }

    // ── Core operations ──────────────────────────────────────────────────

    /// Send a message to the remote agent (`message/send`).
    pub async fn send_message(
        &self,
        message: A2aMessage,
        config: Option<MessageSendConfiguration>,
    ) -> Result<SendMessageResult> {
        let params = serde_json::to_value(SendMessageRequest {
            message,
            configuration: config,
        })
        .map_err(|e| GaussError::internal(format!("Serialize params: {e}")))?;

        let result = self.jsonrpc_call("message/send", params).await?;
        let resp: SendMessageResponse = serde_json::from_value(result)
            .map_err(|e| GaussError::provider("a2a", format!("Invalid response: {e}")))?;

        Ok(match resp {
            SendMessageResponse::Task(t) => SendMessageResult::Task(t),
            SendMessageResponse::Message(m) => SendMessageResult::Message(m),
        })
    }

    /// Send a message and stream results (`message/stream`).
    /// Returns a stream of [`A2aStreamEvent`].
    pub async fn stream_message(
        &self,
        message: A2aMessage,
        config: Option<MessageSendConfiguration>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<A2aStreamEvent>> + Send>>> {
        let params = serde_json::to_value(SendMessageRequest {
            message,
            configuration: config,
        })
        .map_err(|e| GaussError::internal(format!("Serialize params: {e}")))?;

        let body = JsonRpcRequest::new(
            serde_json::Value::String(uuid::Uuid::new_v4().to_string()),
            "message/stream",
            params,
        );

        let mut req = self.http.post(&self.base_url).json(&body);
        if let Some(token) = &self.auth_token {
            req = req.bearer_auth(token);
        }

        let resp = req.send().await.map_err(|e| {
            GaussError::provider("a2a", format!("Stream request failed: {e}"))
        })?;
        if !resp.status().is_success() {
            return Err(GaussError::provider(
                "a2a",
                format!("Stream request failed with status {}", resp.status()),
            ));
        }

        let stream = async_stream::try_stream! {
            use futures::StreamExt;
            let mut byte_stream = resp.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk) = byte_stream.next().await {
                let chunk = chunk.map_err(|e| {
                    GaussError::provider("a2a", format!("Stream read error: {e}"))
                })?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete SSE events (terminated by double newline).
                while let Some(pos) = buffer.find("\n\n") {
                    let event_block = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    for line in event_block.lines() {
                        let line = line.trim();
                        if let Some(data) = line.strip_prefix("data:") {
                            let data = data.trim();
                            if data.is_empty() || data == "[DONE]" {
                                continue;
                            }
                            // The SSE data is a full JSON-RPC response envelope.
                            let rpc_resp: JsonRpcResponse = serde_json::from_str(data)
                                .map_err(|e| {
                                    GaussError::provider("a2a", format!("Invalid SSE JSON-RPC: {e}"))
                                })?;
                            if let Some(err) = rpc_resp.error {
                                Err(GaussError::provider(
                                    "a2a",
                                    format!("Remote error {}: {}", err.code, err.message),
                                ))?;
                            }
                            if let Some(result) = rpc_resp.result {
                                let evt: A2aStreamEvent = serde_json::from_value(result)
                                    .map_err(|e| {
                                        GaussError::provider("a2a", format!("Invalid stream event: {e}"))
                                    })?;
                                yield evt;
                            }
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }

    /// Get a task by ID (`tasks/get`).
    pub async fn get_task(&self, task_id: &str, history_length: Option<u32>) -> Result<Task> {
        let mut params = json!({ "id": task_id });
        if let Some(hl) = history_length {
            params["historyLength"] = json!(hl);
        }
        let result = self.jsonrpc_call("tasks/get", params).await?;
        serde_json::from_value(result)
            .map_err(|e| GaussError::provider("a2a", format!("Invalid task response: {e}")))
    }

    /// List tasks, optionally filtered by context (`tasks/list`).
    pub async fn list_tasks(&self, context_id: Option<&str>) -> Result<Vec<Task>> {
        let params = match context_id {
            Some(id) => json!({ "contextId": id }),
            None => json!({}),
        };
        let result = self.jsonrpc_call("tasks/list", params).await?;
        serde_json::from_value(result)
            .map_err(|e| GaussError::provider("a2a", format!("Invalid list response: {e}")))
    }

    /// Cancel a task (`tasks/cancel`).
    pub async fn cancel_task(&self, task_id: &str) -> Result<Task> {
        let params = json!({ "id": task_id });
        let result = self.jsonrpc_call("tasks/cancel", params).await?;
        serde_json::from_value(result)
            .map_err(|e| GaussError::provider("a2a", format!("Invalid cancel response: {e}")))
    }

    // ── Convenience ──────────────────────────────────────────────────────

    /// Quick helper: send a text message and wait for completion.
    ///
    /// Creates a user text message, sends it, and polls until the task
    /// completes (or returns the message text directly).
    pub async fn ask(&self, text: &str) -> Result<String> {
        let message = A2aMessage::user_text(text);
        let result = self.send_message(message, None).await?;

        match result {
            SendMessageResult::Message(msg) => extract_text(&msg),
            SendMessageResult::Task(mut task) => {
                // Poll until completed / failed / canceled.
                while !matches!(
                    task.status.state,
                    TaskState::Completed | TaskState::Failed | TaskState::Canceled
                ) {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    task = self.get_task(&task.id, None).await?;
                }
                if task.status.state == TaskState::Failed {
                    return Err(GaussError::provider("a2a", "Remote task failed"));
                }
                if task.status.state == TaskState::Canceled {
                    return Err(GaussError::provider("a2a", "Remote task was canceled"));
                }
                // Prefer the last agent message.
                if let Some(msg) = task.messages.iter().rev().find(|m| m.role == A2aMessageRole::Agent) {
                    return extract_text(msg);
                }
                // Fall back to artifact text.
                for artifact in task.artifacts.iter().rev() {
                    for part in &artifact.parts {
                        if let Part::Text { text } = part {
                            return Ok(text.clone());
                        }
                    }
                }
                Err(GaussError::provider("a2a", "No text in completed task"))
            }
        }
    }

    // ── Internal ─────────────────────────────────────────────────────────

    /// Generic JSON-RPC 2.0 call over HTTP POST.
    async fn jsonrpc_call(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let body = JsonRpcRequest::new(
            serde_json::Value::String(uuid::Uuid::new_v4().to_string()),
            method,
            params,
        );

        let mut req = self.http.post(&self.base_url).json(&body);
        if let Some(token) = &self.auth_token {
            req = req.bearer_auth(token);
        }

        let resp = req.send().await.map_err(|e| {
            GaussError::provider("a2a", format!("HTTP request failed: {e}"))
        })?;

        if !resp.status().is_success() {
            return Err(GaussError::provider(
                "a2a",
                format!("HTTP {} from remote agent", resp.status()),
            ));
        }

        let rpc_resp: JsonRpcResponse = resp.json().await.map_err(|e| {
            GaussError::provider("a2a", format!("Invalid JSON-RPC response: {e}"))
        })?;

        if let Some(err) = rpc_resp.error {
            return Err(GaussError::provider(
                "a2a",
                format!("Remote error {}: {}", err.code, err.message),
            ));
        }

        rpc_resp
            .result
            .ok_or_else(|| GaussError::internal("JSON-RPC response missing both result and error"))
    }
}

// ── Result enum ──────────────────────────────────────────────────────────────

/// Mirrors the two possible shapes a `message/send` response can take.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum SendMessageResult {
    Task(Task),
    Message(A2aMessage),
}

use serde::{Deserialize, Serialize};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Extract the first text part from an A2A message.
fn extract_text(msg: &A2aMessage) -> Result<String> {
    for part in &msg.parts {
        if let Part::Text { text } = part {
            return Ok(text.clone());
        }
    }
    Err(GaussError::provider("a2a", "Message contains no text part"))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── 1. Client construction ───────────────────────────────────────────

    #[test]
    fn test_client_construction_defaults() {
        let client = A2aClient::new("https://agent.example.com");
        assert_eq!(client.base_url, "https://agent.example.com");
        assert!(client.auth_token.is_none());
    }

    #[test]
    fn test_client_trims_trailing_slash() {
        let client = A2aClient::new("https://agent.example.com/");
        assert_eq!(client.base_url, "https://agent.example.com");
    }

    #[test]
    fn test_client_with_auth_token() {
        let client = A2aClient::new("https://a.com").with_auth_token("tok-123");
        assert_eq!(client.auth_token.as_deref(), Some("tok-123"));
    }

    #[test]
    fn test_client_with_timeout() {
        let client = A2aClient::new("https://a.com")
            .with_timeout(Duration::from_secs(60));
        // Ensure builder succeeds without panicking; the http client is opaque.
        assert_eq!(client.base_url, "https://a.com");
    }

    // ── 2. JSON-RPC request serialization ────────────────────────────────

    #[test]
    fn test_jsonrpc_request_has_correct_shape() {
        let req = JsonRpcRequest::new(json!("id-1"), "message/send", json!({"key": "val"}));
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["jsonrpc"], "2.0");
        assert_eq!(v["method"], "message/send");
        assert_eq!(v["id"], "id-1");
        assert_eq!(v["params"]["key"], "val");
    }

    #[test]
    fn test_jsonrpc_request_roundtrip() {
        let req = JsonRpcRequest::new(json!("abc"), "tasks/get", json!({"id": "t1"}));
        let serialized = serde_json::to_string(&req).unwrap();
        let back: JsonRpcRequest = serde_json::from_str(&serialized).unwrap();
        assert_eq!(back.method, "tasks/get");
        assert_eq!(back.params["id"], "t1");
    }

    // ── 3. SendMessageResult deserialization ─────────────────────────────

    #[test]
    fn test_send_message_result_task_variant() {
        let j = json!({
            "type": "task",
            "id": "t-1",
            "status": { "state": "completed", "timestamp": "2025-01-01T00:00:00Z" },
            "messages": [],
            "artifacts": []
        });
        let r: SendMessageResult = serde_json::from_value(j).unwrap();
        match r {
            SendMessageResult::Task(t) => assert_eq!(t.id, "t-1"),
            _ => panic!("expected Task variant"),
        }
    }

    #[test]
    fn test_send_message_result_message_variant() {
        let j = json!({
            "type": "message",
            "role": "agent",
            "parts": [{ "type": "text", "text": "Hello!" }]
        });
        let r: SendMessageResult = serde_json::from_value(j).unwrap();
        match r {
            SendMessageResult::Message(m) => {
                assert_eq!(m.role, A2aMessageRole::Agent);
                assert_eq!(m.parts.len(), 1);
            }
            _ => panic!("expected Message variant"),
        }
    }

    // ── 4. Error response handling ───────────────────────────────────────

    #[test]
    fn test_jsonrpc_error_response_parsed() {
        let resp = JsonRpcResponse::error(json!(1), -32001, "Task not found");
        let err = resp.error.as_ref().unwrap();
        assert_eq!(err.code, -32001);
        assert_eq!(err.message, "Task not found");
        assert!(resp.result.is_none());
    }

    #[test]
    fn test_jsonrpc_error_with_data() {
        let mut resp = JsonRpcResponse::error(json!(2), -32603, "Internal");
        resp.error.as_mut().unwrap().data = Some(json!({"detail": "something broke"}));
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["error"]["data"]["detail"], "something broke");
    }

    // ── 5. Agent card URL construction ───────────────────────────────────

    #[test]
    fn test_discover_url_construction() {
        let client = A2aClient::new("https://agent.example.com");
        let expected = "https://agent.example.com/.well-known/agent.json";
        assert_eq!(format!("{}/.well-known/agent.json", client.base_url), expected);
    }

    #[test]
    fn test_discover_url_with_trailing_slash() {
        let client = A2aClient::new("https://agent.example.com/");
        let expected = "https://agent.example.com/.well-known/agent.json";
        assert_eq!(format!("{}/.well-known/agent.json", client.base_url), expected);
    }

    // ── 6. Ask helper message construction ───────────────────────────────

    #[test]
    fn test_ask_builds_user_text_message() {
        let msg = A2aMessage::user_text("What is 2+2?");
        assert_eq!(msg.role, A2aMessageRole::User);
        assert_eq!(msg.parts.len(), 1);
        match &msg.parts[0] {
            Part::Text { text } => assert_eq!(text, "What is 2+2?"),
            _ => panic!("expected text part"),
        }
    }

    // ── 7. extract_text helper ───────────────────────────────────────────

    #[test]
    fn test_extract_text_success() {
        let msg = A2aMessage::agent_text("The answer is 4.");
        let result = extract_text(&msg).unwrap();
        assert_eq!(result, "The answer is 4.");
    }

    #[test]
    fn test_extract_text_no_text_part_returns_error() {
        let msg = A2aMessage {
            role: A2aMessageRole::Agent,
            parts: vec![Part::Data { data: json!(42) }],
            metadata: None,
        };
        assert!(extract_text(&msg).is_err());
    }
}
