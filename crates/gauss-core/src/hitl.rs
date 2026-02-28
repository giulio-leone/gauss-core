//! Human-in-the-Loop — suspend/resume agent execution for human approval.
//!
//! Provides approval gates for tool calls, serializable agent state
//! for checkpoint/resume, and workflow-level human review steps.

use crate::error;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Approval Types
// ---------------------------------------------------------------------------

/// Status of an approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalStatus {
    Pending,
    Approved,
    Denied,
    TimedOut,
}

/// A request for human approval before tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub id: String,
    pub tool_name: String,
    pub args: serde_json::Value,
    pub step_index: usize,
    pub session_id: String,
    pub status: ApprovalStatus,
    pub created_at: u64,
    /// Modified args (if human edited them).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_args: Option<serde_json::Value>,
    /// Reason for denial.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub denial_reason: Option<String>,
}

/// Configuration for HITL behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HitlConfig {
    /// Tool names that require approval. Empty = all tools.
    #[serde(default)]
    pub require_approval_for: Vec<String>,
    /// Timeout in ms for approval requests (0 = no timeout).
    pub timeout_ms: u64,
    /// What to do on timeout.
    pub on_timeout: TimeoutAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeoutAction {
    Deny,
    Approve,
    Error,
}

impl Default for HitlConfig {
    fn default() -> Self {
        Self {
            require_approval_for: Vec::new(),
            timeout_ms: 0,
            on_timeout: TimeoutAction::Deny,
        }
    }
}

// ---------------------------------------------------------------------------
// Checkpoint (serializable agent state)
// ---------------------------------------------------------------------------

/// Serializable snapshot of agent execution state for suspend/resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub id: String,
    pub session_id: String,
    pub step_index: usize,
    pub messages: Vec<crate::message::Message>,
    pub pending_approval: Option<ApprovalRequest>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: u64,
    /// Schema version for migration support.
    pub schema_version: u32,
}

impl Checkpoint {
    pub fn new(session_id: String, messages: Vec<crate::message::Message>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            session_id,
            step_index: 0,
            messages,
            pending_approval: None,
            metadata: HashMap::new(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            schema_version: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Checkpoint Store Trait
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
#[async_trait::async_trait]
pub trait CheckpointStore: Send + Sync {
    async fn save(&self, checkpoint: &Checkpoint) -> error::Result<()>;
    async fn load(&self, id: &str) -> error::Result<Option<Checkpoint>>;
    async fn load_latest(&self, session_id: &str) -> error::Result<Option<Checkpoint>>;
    async fn delete(&self, id: &str) -> error::Result<()>;
    async fn list(&self, session_id: &str) -> error::Result<Vec<Checkpoint>>;
}

#[cfg(target_arch = "wasm32")]
#[async_trait::async_trait(?Send)]
pub trait CheckpointStore {
    async fn save(&self, checkpoint: &Checkpoint) -> error::Result<()>;
    async fn load(&self, id: &str) -> error::Result<Option<Checkpoint>>;
    async fn load_latest(&self, session_id: &str) -> error::Result<Option<Checkpoint>>;
    async fn delete(&self, id: &str) -> error::Result<()>;
    async fn list(&self, session_id: &str) -> error::Result<Vec<Checkpoint>>;
}

// ---------------------------------------------------------------------------
// In-Memory Checkpoint Store
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct InMemoryCheckpointStore {
    checkpoints: std::sync::Mutex<Vec<Checkpoint>>,
}

impl InMemoryCheckpointStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl CheckpointStore for InMemoryCheckpointStore {
    async fn save(&self, checkpoint: &Checkpoint) -> error::Result<()> {
        let mut store = self
            .checkpoints
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;
        // Update if exists, otherwise insert
        if let Some(existing) = store.iter_mut().find(|c| c.id == checkpoint.id) {
            *existing = checkpoint.clone();
        } else {
            store.push(checkpoint.clone());
        }
        Ok(())
    }

    async fn load(&self, id: &str) -> error::Result<Option<Checkpoint>> {
        let store = self
            .checkpoints
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;
        Ok(store.iter().find(|c| c.id == id).cloned())
    }

    async fn load_latest(&self, session_id: &str) -> error::Result<Option<Checkpoint>> {
        let store = self
            .checkpoints
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;
        Ok(store
            .iter()
            .filter(|c| c.session_id == session_id)
            .max_by_key(|c| c.created_at)
            .cloned())
    }

    async fn delete(&self, id: &str) -> error::Result<()> {
        let mut store = self
            .checkpoints
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;
        store.retain(|c| c.id != id);
        Ok(())
    }

    async fn list(&self, session_id: &str) -> error::Result<Vec<Checkpoint>> {
        let store = self
            .checkpoints
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;
        Ok(store
            .iter()
            .filter(|c| c.session_id == session_id)
            .cloned()
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Approval Manager
// ---------------------------------------------------------------------------

/// Manages pending approval requests.
#[derive(Debug, Default)]
pub struct ApprovalManager {
    pending: std::sync::Mutex<HashMap<String, ApprovalRequest>>,
}

impl ApprovalManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new approval request.
    pub fn request_approval(
        &self,
        tool_name: String,
        args: serde_json::Value,
        step_index: usize,
        session_id: String,
    ) -> error::Result<ApprovalRequest> {
        let request = ApprovalRequest {
            id: uuid::Uuid::new_v4().to_string(),
            tool_name,
            args,
            step_index,
            session_id,
            status: ApprovalStatus::Pending,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            modified_args: None,
            denial_reason: None,
        };

        self.pending
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?
            .insert(request.id.clone(), request.clone());

        Ok(request)
    }

    /// Approve a pending request.
    pub fn approve(
        &self,
        id: &str,
        modified_args: Option<serde_json::Value>,
    ) -> error::Result<ApprovalRequest> {
        let mut pending = self
            .pending
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;
        let request = pending.get_mut(id).ok_or_else(|| {
            error::GaussError::internal(format!("Approval request '{}' not found", id))
        })?;
        request.status = ApprovalStatus::Approved;
        request.modified_args = modified_args;
        let result = request.clone();
        pending.remove(id);
        Ok(result)
    }

    /// Deny a pending request.
    pub fn deny(&self, id: &str, reason: Option<String>) -> error::Result<ApprovalRequest> {
        let mut pending = self
            .pending
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;
        let request = pending.get_mut(id).ok_or_else(|| {
            error::GaussError::internal(format!("Approval request '{}' not found", id))
        })?;
        request.status = ApprovalStatus::Denied;
        request.denial_reason = reason;
        let result = request.clone();
        pending.remove(id);
        Ok(result)
    }

    /// Get all pending requests.
    pub fn list_pending(&self) -> error::Result<Vec<ApprovalRequest>> {
        let pending = self
            .pending
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;
        Ok(pending.values().cloned().collect())
    }
}
