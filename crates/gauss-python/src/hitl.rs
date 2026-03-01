use crate::registry::{py_err, HandleRegistry};
use gauss_core::hitl;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

// ============ Approval ============

static APPROVALS: HandleRegistry<Arc<Mutex<hitl::ApprovalManager>>> = HandleRegistry::new();

#[pyfunction]
pub fn create_approval_manager() -> u32 {
    APPROVALS.insert(Arc::new(Mutex::new(hitl::ApprovalManager::new())))
}

#[pyfunction]
pub fn approval_request(
    handle: u32,
    tool_name: String,
    args_json: String,
    session_id: String,
) -> PyResult<String> {
    let mgr = APPROVALS.get_clone(handle)?;
    let args: serde_json::Value = serde_json::from_str(&args_json).map_err(py_err)?;
    let req = mgr
        .lock()
        .unwrap()
        .request_approval(tool_name, args, 0, session_id)
        .map_err(py_err)?;
    Ok(req.id.clone())
}

#[pyfunction]
#[pyo3(signature = (handle, request_id, modified_args=None))]
pub fn approval_approve(
    handle: u32,
    request_id: &str,
    modified_args: Option<String>,
) -> PyResult<()> {
    let mgr = APPROVALS.get_clone(handle)?;
    let args: Option<serde_json::Value> = match modified_args {
        Some(j) => Some(serde_json::from_str(&j).map_err(py_err)?),
        None => None,
    };
    mgr.lock()
        .unwrap()
        .approve(request_id, args)
        .map_err(py_err)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (handle, request_id, reason=None))]
pub fn approval_deny(handle: u32, request_id: &str, reason: Option<String>) -> PyResult<()> {
    let mgr = APPROVALS.get_clone(handle)?;
    mgr.lock()
        .unwrap()
        .deny(request_id, reason)
        .map_err(py_err)?;
    Ok(())
}

#[pyfunction]
pub fn approval_list_pending(handle: u32) -> PyResult<String> {
    let mgr = APPROVALS.get_clone(handle)?;
    let pending = mgr
        .lock()
        .expect("registry mutex poisoned")
        .list_pending()
        .map_err(py_err)?;
    serde_json::to_string(&pending).map_err(py_err)
}

#[pyfunction]
pub fn destroy_approval_manager(handle: u32) -> PyResult<()> {
    APPROVALS.remove(handle)
}

// ============ Checkpoint Store ============

static CHECKPOINTS: HandleRegistry<Arc<hitl::InMemoryCheckpointStore>> = HandleRegistry::new();

#[pyfunction]
pub fn create_checkpoint_store() -> u32 {
    CHECKPOINTS.insert(Arc::new(hitl::InMemoryCheckpointStore::new()))
}

#[pyfunction]
pub fn checkpoint_save(
    py: Python<'_>,
    handle: u32,
    checkpoint_json: String,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        use hitl::CheckpointStore;
        let store = CHECKPOINTS.get_clone(handle)?;
        let cp: hitl::Checkpoint = serde_json::from_str(&checkpoint_json).map_err(py_err)?;
        store.save(&cp).await.map_err(py_err)
    })
}

#[pyfunction]
pub fn checkpoint_load(
    py: Python<'_>,
    handle: u32,
    checkpoint_id: String,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        use hitl::CheckpointStore;
        let store = CHECKPOINTS.get_clone(handle)?;
        let cp = store.load(&checkpoint_id).await.map_err(py_err)?;
        serde_json::to_string(&cp).map_err(py_err)
    })
}

#[pyfunction]
pub fn destroy_checkpoint_store(handle: u32) -> PyResult<()> {
    CHECKPOINTS.remove(handle)
}

#[pyfunction]
pub fn checkpoint_load_latest<'py>(
    py: Python<'py>,
    handle: u32,
    session_id: String,
) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let store = CHECKPOINTS.get_clone(handle)?;
        use gauss_core::hitl::CheckpointStore;
        match store.load_latest(&session_id).await {
            Ok(Some(cp)) => serde_json::to_string(&cp).map_err(py_err),
            Ok(None) => Ok("null".to_string()),
            Err(e) => Err(py_err(e)),
        }
    })
}
