use crate::registry::{py_err, HandleRegistry};
use crate::types::parse_messages;
use gauss_core::memory::Memory;
use gauss_core::{context, memory, rag};
use pyo3::prelude::*;
use std::sync::Arc;

// ============ Memory ============

static MEMORIES: HandleRegistry<Arc<memory::InMemoryMemory>> = HandleRegistry::new();

#[pyfunction]
pub fn create_memory() -> u32 {
    MEMORIES.insert(Arc::new(memory::InMemoryMemory::new()))
}

#[pyfunction]
pub fn memory_store(
    py: Python<'_>,
    handle: u32,
    entry_json: String,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        use memory::Memory;
        let mem = MEMORIES.get_clone(handle)?;
        let entry: memory::MemoryEntry = serde_json::from_str(&entry_json).map_err(py_err)?;
        mem.store(entry).await.map_err(py_err)
    })
}

#[pyfunction]
#[pyo3(signature = (handle, options_json=None))]
pub fn memory_recall(
    py: Python<'_>,
    handle: u32,
    options_json: Option<String>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        use memory::Memory;
        let mem = MEMORIES.get_clone(handle)?;
        let opts: memory::RecallOptions = match options_json {
            Some(j) => serde_json::from_str(&j).map_err(py_err)?,
            None => memory::RecallOptions::default(),
        };
        let entries = mem.recall(opts).await.map_err(py_err)?;
        serde_json::to_string(&entries).map_err(py_err)
    })
}

#[pyfunction]
#[pyo3(signature = (handle, session_id=None))]
pub fn memory_clear(
    py: Python<'_>,
    handle: u32,
    session_id: Option<String>,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let mem = MEMORIES.get_clone(handle)?;
        memory::Memory::clear(&*mem, session_id.as_deref())
            .await
            .map_err(py_err)
    })
}

#[pyfunction]
pub fn destroy_memory(handle: u32) -> PyResult<()> {
    MEMORIES.remove(handle)
}

#[pyfunction]
pub fn memory_stats<'py>(py: Python<'py>, handle: u32) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
    let mem = MEMORIES.get_clone(handle)?;
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let stats = mem.stats().await.map_err(py_err)?;
        serde_json::to_string(&stats).map_err(py_err)
    })
}

// ============ Context ============

#[pyfunction]
pub fn count_tokens(text: &str) -> u32 {
    context::count_tokens(text) as u32
}

#[pyfunction]
pub fn count_tokens_for_model(text: &str, model: &str) -> u32 {
    context::count_tokens_for_model(text, model) as u32
}

#[pyfunction]
pub fn count_message_tokens(messages_json: &str) -> PyResult<u32> {
    let msgs = parse_messages(messages_json)?;
    Ok(context::count_messages_tokens(&msgs) as u32)
}

#[pyfunction]
pub fn get_context_window_size(model: &str) -> u32 {
    context::context_window_size(model) as u32
}

// ============ RAG ============

static VECTOR_STORES: HandleRegistry<Arc<rag::InMemoryVectorStore>> = HandleRegistry::new();

#[pyfunction]
pub fn create_vector_store() -> u32 {
    VECTOR_STORES.insert(Arc::new(rag::InMemoryVectorStore::new()))
}

#[pyfunction]
pub fn vector_store_upsert(
    py: Python<'_>,
    handle: u32,
    chunks_json: String,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        use rag::VectorStore;
        let store = VECTOR_STORES.get_clone(handle)?;
        let chunks: Vec<rag::Chunk> = serde_json::from_str(&chunks_json).map_err(py_err)?;
        store.upsert(chunks).await.map_err(py_err)
    })
}

#[pyfunction]
pub fn vector_store_search(
    py: Python<'_>,
    handle: u32,
    embedding_json: String,
    top_k: u32,
) -> PyResult<Bound<'_, pyo3::types::PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        use rag::VectorStore;
        let store = VECTOR_STORES.get_clone(handle)?;
        let embedding: Vec<f32> = serde_json::from_str(&embedding_json).map_err(py_err)?;
        let results = store
            .search(&embedding, top_k as usize)
            .await
            .map_err(py_err)?;
        serde_json::to_string(&results).map_err(py_err)
    })
}

#[pyfunction]
pub fn destroy_vector_store(handle: u32) -> PyResult<()> {
    VECTOR_STORES.remove(handle)
}

#[pyfunction]
pub fn cosine_similarity(a_json: &str, b_json: &str) -> PyResult<f64> {
    let a: Vec<f32> = serde_json::from_str(a_json).map_err(py_err)?;
    let b: Vec<f32> = serde_json::from_str(b_json).map_err(py_err)?;
    Ok(rag::cosine_similarity(&a, &b) as f64)
}
