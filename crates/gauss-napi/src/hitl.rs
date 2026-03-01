use crate::registry::HandleRegistry;
use gauss_core::hitl;
use napi::bindgen_prelude::*;
use std::sync::{Arc, Mutex};

// ============ Approval ============

static APPROVALS: HandleRegistry<Arc<Mutex<hitl::ApprovalManager>>> = HandleRegistry::new();

#[napi]
pub fn create_approval_manager() -> u32 {
    APPROVALS.insert(Arc::new(Mutex::new(hitl::ApprovalManager::new())))
}

#[napi]
pub fn approval_request(
    handle: u32,
    tool_name: String,
    args_json: String,
    session_id: String,
) -> Result<String> {
    let mgr = APPROVALS.get_clone(handle)?;
    let args: serde_json::Value = serde_json::from_str(&args_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid args: {e}")))?;
    let req = mgr
        .lock()
        .unwrap()
        .request_approval(tool_name, args, 0, session_id)
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    Ok(req.id.clone())
}

#[napi]
pub fn approval_approve(
    handle: u32,
    request_id: String,
    modified_args: Option<String>,
) -> Result<()> {
    let mgr = APPROVALS.get_clone(handle)?;
    let args: Option<serde_json::Value> = match modified_args {
        Some(j) => Some(
            serde_json::from_str(&j)
                .map_err(|e| napi::Error::from_reason(format!("Invalid args: {e}")))?,
        ),
        None => None,
    };
    mgr.lock()
        .unwrap()
        .approve(&request_id, args)
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    Ok(())
}

#[napi]
pub fn approval_deny(handle: u32, request_id: String, reason: Option<String>) -> Result<()> {
    let mgr = APPROVALS.get_clone(handle)?;
    mgr.lock()
        .unwrap()
        .deny(&request_id, reason)
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    Ok(())
}

#[napi]
pub fn approval_list_pending(handle: u32) -> Result<serde_json::Value> {
    let mgr = APPROVALS.get_clone(handle)?;
    let pending = mgr
        .lock()
        .unwrap()
        .list_pending()
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&pending).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn destroy_approval_manager(handle: u32) -> Result<()> {
    APPROVALS.remove(handle)?;
    Ok(())
}

// ============ Checkpoint Store ============

static CHECKPOINTS: HandleRegistry<Arc<hitl::InMemoryCheckpointStore>> = HandleRegistry::new();

#[napi]
pub fn create_checkpoint_store() -> u32 {
    CHECKPOINTS.insert(Arc::new(hitl::InMemoryCheckpointStore::new()))
}

#[napi]
pub async fn checkpoint_save(handle: u32, checkpoint_json: String) -> Result<()> {
    use hitl::CheckpointStore;
    let store = CHECKPOINTS.get_clone(handle)?;
    let cp: hitl::Checkpoint = serde_json::from_str(&checkpoint_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid checkpoint: {e}")))?;
    store
        .save(&cp)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub async fn checkpoint_load(handle: u32, checkpoint_id: String) -> Result<serde_json::Value> {
    use hitl::CheckpointStore;
    let store = CHECKPOINTS.get_clone(handle)?;
    let cp = store
        .load(&checkpoint_id)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&cp).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub async fn checkpoint_load_latest(handle: u32, session_id: String) -> Result<serde_json::Value> {
    use hitl::CheckpointStore;
    let store = CHECKPOINTS.get_clone(handle)?;
    let cp = store
        .load_latest(&session_id)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&cp).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn destroy_checkpoint_store(handle: u32) -> Result<()> {
    CHECKPOINTS.remove(handle)?;
    Ok(())
}
