//! A2A (Agent-to-Agent) server-side logic.
//!
//! Provides a framework-agnostic JSON-RPC router that dispatches A2A protocol
//! requests to a user-defined [`A2aHandler`] implementation.

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::a2a::*;
use crate::error::{GaussError, Result};

// ── JSON-RPC Standard Error Codes ────────────────────────────────────────────

const PARSE_ERROR: i32 = -32700;
const METHOD_NOT_FOUND: i32 = -32601;
const INVALID_PARAMS: i32 = -32602;
const INTERNAL_ERROR: i32 = -32603;

// ── Response / Event Enums ───────────────────────────────────────────────────

/// Response from [`A2aHandler::handle_send_message`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum SendMessageResponse {
    Task(Task),
    Message(A2aMessage),
}

/// Events emitted during A2A streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum A2aStreamEvent {
    Task(Task),
    StatusUpdate(TaskStatusUpdateEvent),
    ArtifactUpdate(TaskArtifactUpdateEvent),
    Message(A2aMessage),
}

// ── Handler Trait ────────────────────────────────────────────────────────────

/// Async trait that users implement to handle A2A requests.
#[async_trait::async_trait]
pub trait A2aHandler: Send + Sync {
    /// Handle a `message/send` request.
    async fn handle_send_message(&self, request: SendMessageRequest) -> Result<SendMessageResponse>;

    /// Handle a `message/stream` request (optional — default returns error).
    async fn handle_stream_message(
        &self,
        _request: SendMessageRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = A2aStreamEvent> + Send>>> {
        Err(GaussError::Internal {
            message: "Streaming not supported".into(),
        })
    }

    /// Get a task by ID.
    async fn handle_get_task(&self, task_id: &str, history_length: Option<u32>) -> Result<Task>;

    /// List tasks, optionally filtered by context ID.
    async fn handle_list_tasks(&self, context_id: Option<&str>) -> Result<Vec<Task>>;

    /// Cancel a task.
    async fn handle_cancel_task(&self, task_id: &str) -> Result<Task>;

    /// Return the agent card.
    fn agent_card(&self) -> &AgentCard;
}

// ── Router ───────────────────────────────────────────────────────────────────

/// Framework-agnostic JSON-RPC request router for A2A.
pub struct A2aRouter<H: A2aHandler> {
    handler: Arc<H>,
}

impl<H: A2aHandler + 'static> A2aRouter<H> {
    /// Create a new router wrapping the given handler.
    pub fn new(handler: H) -> Self {
        Self {
            handler: Arc::new(handler),
        }
    }

    /// Process a raw JSON-RPC request string and return a JSON-RPC response string.
    pub async fn handle_jsonrpc(&self, request_body: &str) -> String {
        let req = match serde_json::from_str::<JsonRpcRequest>(request_body) {
            Ok(r) => r,
            Err(e) => {
                let resp = JsonRpcResponse::error(
                    serde_json::Value::Null,
                    PARSE_ERROR,
                    format!("Parse error: {e}"),
                );
                return serde_json::to_string(&resp).unwrap_or_default();
            }
        };

        let resp = self.dispatch(&req).await;
        serde_json::to_string(&resp).unwrap_or_default()
    }

    /// Process a JSON-RPC request and return streaming events.
    ///
    /// Returns `Ok(None)` if the method doesn't support streaming.
    /// Each yielded `String` is an SSE-formatted line: `data: {json}\n\n`.
    pub async fn handle_jsonrpc_stream(
        &self,
        request_body: &str,
    ) -> Result<Option<Pin<Box<dyn Stream<Item = String> + Send>>>> {
        let req: JsonRpcRequest = serde_json::from_str(request_body).map_err(|e| {
            GaussError::Internal {
                message: format!("Parse error: {e}"),
            }
        })?;

        if req.method != "message/stream" {
            return Ok(None);
        }

        let params: SendMessageRequest =
            serde_json::from_value(req.params.clone()).map_err(|e| GaussError::Internal {
                message: format!("Invalid params: {e}"),
            })?;

        let inner = self.handler.handle_stream_message(params).await?;
        let id = req.id.clone();

        let sse_stream = Box::pin(futures::stream::unfold(
            (inner, id),
            |(mut stream, id)| async move {
                use futures::StreamExt;
                match stream.next().await {
                    Some(event) => {
                        let resp = JsonRpcResponse::success(
                            id.clone(),
                            serde_json::to_value(&event).unwrap_or_default(),
                        );
                        let json = serde_json::to_string(&resp).unwrap_or_default();
                        let sse = format!("data: {json}\n\n");
                        Some((sse, (stream, id)))
                    }
                    None => None,
                }
            },
        ));

        Ok(Some(sse_stream))
    }

    /// Return the agent card as a JSON string (for `/.well-known/agent.json`).
    pub fn agent_card_json(&self) -> String {
        serde_json::to_string(self.handler.agent_card()).unwrap_or_default()
    }

    // ── Internal dispatch ────────────────────────────────────────────────

    async fn dispatch(&self, req: &JsonRpcRequest) -> JsonRpcResponse {
        match req.method.as_str() {
            "message/send" => self.dispatch_send_message(req).await,
            "message/stream" => {
                JsonRpcResponse::error(
                    req.id.clone(),
                    METHOD_NOT_FOUND,
                    "Use the streaming endpoint for message/stream",
                )
            }
            "tasks/get" => self.dispatch_get_task(req).await,
            "tasks/list" => self.dispatch_list_tasks(req).await,
            "tasks/cancel" => self.dispatch_cancel_task(req).await,
            _ => JsonRpcResponse::error(
                req.id.clone(),
                METHOD_NOT_FOUND,
                format!("Method not found: {}", req.method),
            ),
        }
    }

    async fn dispatch_send_message(&self, req: &JsonRpcRequest) -> JsonRpcResponse {
        let params: SendMessageRequest = match serde_json::from_value(req.params.clone()) {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(
                    req.id.clone(),
                    INVALID_PARAMS,
                    format!("Invalid params: {e}"),
                )
            }
        };

        match self.handler.handle_send_message(params).await {
            Ok(resp) => JsonRpcResponse::success(
                req.id.clone(),
                serde_json::to_value(&resp).unwrap_or_default(),
            ),
            Err(e) => handler_error_to_response(req.id.clone(), e),
        }
    }

    async fn dispatch_get_task(&self, req: &JsonRpcRequest) -> JsonRpcResponse {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Params {
            id: String,
            history_length: Option<u32>,
        }

        let params: Params = match serde_json::from_value(req.params.clone()) {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(
                    req.id.clone(),
                    INVALID_PARAMS,
                    format!("Invalid params: {e}"),
                )
            }
        };

        match self
            .handler
            .handle_get_task(&params.id, params.history_length)
            .await
        {
            Ok(task) => JsonRpcResponse::success(
                req.id.clone(),
                serde_json::to_value(&task).unwrap_or_default(),
            ),
            Err(e) => handler_error_to_response(req.id.clone(), e),
        }
    }

    async fn dispatch_list_tasks(&self, req: &JsonRpcRequest) -> JsonRpcResponse {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Params {
            context_id: Option<String>,
        }

        let params: Params = match serde_json::from_value(req.params.clone()) {
            Ok(p) => p,
            Err(_) => Params { context_id: None },
        };

        match self
            .handler
            .handle_list_tasks(params.context_id.as_deref())
            .await
        {
            Ok(tasks) => JsonRpcResponse::success(
                req.id.clone(),
                serde_json::to_value(&tasks).unwrap_or_default(),
            ),
            Err(e) => handler_error_to_response(req.id.clone(), e),
        }
    }

    async fn dispatch_cancel_task(&self, req: &JsonRpcRequest) -> JsonRpcResponse {
        #[derive(Deserialize)]
        struct Params {
            id: String,
        }

        let params: Params = match serde_json::from_value(req.params.clone()) {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(
                    req.id.clone(),
                    INVALID_PARAMS,
                    format!("Invalid params: {e}"),
                )
            }
        };

        match self.handler.handle_cancel_task(&params.id).await {
            Ok(task) => JsonRpcResponse::success(
                req.id.clone(),
                serde_json::to_value(&task).unwrap_or_default(),
            ),
            Err(e) => handler_error_to_response(req.id.clone(), e),
        }
    }
}

// ── Error mapping ────────────────────────────────────────────────────────────

fn handler_error_to_response(id: serde_json::Value, err: GaussError) -> JsonRpcResponse {
    let message = err.to_string();
    let code = match &err {
        GaussError::Internal { .. } => INTERNAL_ERROR,
        _ => INTERNAL_ERROR,
    };
    // Check for well-known A2A errors embedded in the message.
    if message.contains("not found") {
        return JsonRpcResponse::error(id, TASK_NOT_FOUND, message);
    }
    if message.contains("not supported") || message.contains("Unsupported") {
        return JsonRpcResponse::error(id, UNSUPPORTED_OPERATION, message);
    }
    JsonRpcResponse::error(id, code, message)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── Mock handler ─────────────────────────────────────────────────────

    struct MockHandler {
        card: AgentCard,
    }

    impl MockHandler {
        fn new() -> Self {
            Self {
                card: AgentCard::new("TestAgent", "A test agent", "https://test.example.com", "1.0"),
            }
        }
    }

    #[async_trait::async_trait]
    impl A2aHandler for MockHandler {
        async fn handle_send_message(
            &self,
            _request: SendMessageRequest,
        ) -> Result<SendMessageResponse> {
            let task = Task {
                id: "task-1".into(),
                context_id: None,
                status: TaskStatus::new(TaskState::Completed, "2025-01-01T00:00:00Z"),
                messages: vec![A2aMessage::agent_text("Done")],
                artifacts: vec![],
                metadata: None,
            };
            Ok(SendMessageResponse::Task(task))
        }

        async fn handle_get_task(
            &self,
            task_id: &str,
            _history_length: Option<u32>,
        ) -> Result<Task> {
            if task_id == "task-1" {
                Ok(Task {
                    id: "task-1".into(),
                    context_id: None,
                    status: TaskStatus::new(TaskState::Completed, "2025-01-01T00:00:00Z"),
                    messages: vec![],
                    artifacts: vec![],
                    metadata: None,
                })
            } else {
                Err(GaussError::internal(format!("Task {task_id} not found")))
            }
        }

        async fn handle_list_tasks(&self, _context_id: Option<&str>) -> Result<Vec<Task>> {
            Ok(vec![])
        }

        async fn handle_cancel_task(&self, task_id: &str) -> Result<Task> {
            Ok(Task {
                id: task_id.to_string(),
                context_id: None,
                status: TaskStatus::new(TaskState::Canceled, "2025-01-01T00:00:00Z"),
                messages: vec![],
                artifacts: vec![],
                metadata: None,
            })
        }

        fn agent_card(&self) -> &AgentCard {
            &self.card
        }
    }

    fn make_router() -> A2aRouter<MockHandler> {
        A2aRouter::new(MockHandler::new())
    }

    // ── Tests ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_dispatch_message_send() {
        let router = make_router();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": { "role": "user", "parts": [{ "type": "text", "text": "hello" }] }
            }
        });
        let resp_str = router.handle_jsonrpc(&req.to_string()).await;
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.error.is_none());
        assert!(resp.result.is_some());
    }

    #[tokio::test]
    async fn test_dispatch_tasks_get() {
        let router = make_router();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tasks/get",
            "params": { "id": "task-1" }
        });
        let resp_str = router.handle_jsonrpc(&req.to_string()).await;
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert_eq!(result["id"], "task-1");
    }

    #[tokio::test]
    async fn test_dispatch_tasks_list() {
        let router = make_router();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tasks/list",
            "params": {}
        });
        let resp_str = router.handle_jsonrpc(&req.to_string()).await;
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert!(result.as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_dispatch_tasks_cancel() {
        let router = make_router();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tasks/cancel",
            "params": { "id": "task-99" }
        });
        let resp_str = router.handle_jsonrpc(&req.to_string()).await;
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert_eq!(result["status"]["state"], "canceled");
    }

    #[tokio::test]
    async fn test_method_not_found() {
        let router = make_router();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "unknown/method",
            "params": {}
        });
        let resp_str = router.handle_jsonrpc(&req.to_string()).await;
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.result.is_none());
        let err = resp.error.unwrap();
        assert_eq!(err.code, METHOD_NOT_FOUND);
    }

    #[tokio::test]
    async fn test_parse_error() {
        let router = make_router();
        let resp_str = router.handle_jsonrpc("not json at all").await;
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.result.is_none());
        let err = resp.error.unwrap();
        assert_eq!(err.code, PARSE_ERROR);
    }

    #[tokio::test]
    async fn test_invalid_params() {
        let router = make_router();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tasks/get",
            "params": { "wrong_field": true }
        });
        let resp_str = router.handle_jsonrpc(&req.to_string()).await;
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.result.is_none());
        let err = resp.error.unwrap();
        assert_eq!(err.code, INVALID_PARAMS);
    }

    #[tokio::test]
    async fn test_agent_card_json() {
        let router = make_router();
        let card_str = router.agent_card_json();
        let card: AgentCard = serde_json::from_str(&card_str).unwrap();
        assert_eq!(card.name, "TestAgent");
        assert_eq!(card.protocol_version, "0.3.0");
    }

    #[test]
    fn test_send_message_response_serialization() {
        let task = Task {
            id: "t1".into(),
            context_id: None,
            status: TaskStatus::new(TaskState::Completed, "2025-01-01T00:00:00Z"),
            messages: vec![],
            artifacts: vec![],
            metadata: None,
        };
        let resp = SendMessageResponse::Task(task);
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["type"], "task");
        assert_eq!(json["id"], "t1");

        let msg_resp = SendMessageResponse::Message(A2aMessage::agent_text("hi"));
        let json = serde_json::to_value(&msg_resp).unwrap();
        assert_eq!(json["type"], "message");
    }

    #[test]
    fn test_stream_event_serialization() {
        let evt = A2aStreamEvent::StatusUpdate(TaskStatusUpdateEvent {
            id: "t1".into(),
            status: TaskStatus::new(TaskState::Working, "2025-01-01T00:00:00Z"),
            final_: false,
        });
        let json = serde_json::to_value(&evt).unwrap();
        assert_eq!(json["type"], "statusUpdate");
        assert_eq!(json["id"], "t1");

        let evt2 = A2aStreamEvent::ArtifactUpdate(TaskArtifactUpdateEvent {
            id: "t1".into(),
            artifact: Artifact {
                name: Some("out".into()),
                description: None,
                parts: vec![Part::Text { text: "chunk".into() }],
                index: Some(0),
                append: None,
                last_chunk: None,
                metadata: None,
            },
        });
        let json2 = serde_json::to_value(&evt2).unwrap();
        assert_eq!(json2["type"], "artifactUpdate");
    }

    #[tokio::test]
    async fn test_task_not_found_error() {
        let router = make_router();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tasks/get",
            "params": { "id": "nonexistent" }
        });
        let resp_str = router.handle_jsonrpc(&req.to_string()).await;
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        let err = resp.error.unwrap();
        assert_eq!(err.code, TASK_NOT_FOUND);
        assert!(err.message.contains("not found"));
    }

    #[tokio::test]
    async fn test_stream_endpoint_via_non_stream_returns_method_not_found() {
        let router = make_router();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "message/stream",
            "params": {
                "message": { "role": "user", "parts": [{ "type": "text", "text": "hi" }] }
            }
        });
        // message/stream through the non-streaming endpoint returns error
        let resp_str = router.handle_jsonrpc(&req.to_string()).await;
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        let err = resp.error.unwrap();
        assert_eq!(err.code, METHOD_NOT_FOUND);
    }
}
