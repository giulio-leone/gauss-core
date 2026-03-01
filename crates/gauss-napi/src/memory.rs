use crate::registry::HandleRegistry;
use crate::types::JsMessage;
use gauss_core::{context, memory, rag};
use napi::bindgen_prelude::*;
use std::sync::Arc;

// ============ Memory ============

static MEMORIES: HandleRegistry<Arc<memory::InMemoryMemory>> = HandleRegistry::new();

#[napi]
pub fn create_memory() -> u32 {
    MEMORIES.insert(Arc::new(memory::InMemoryMemory::new()))
}

#[napi]
pub async fn memory_store(handle: u32, entry_json: String) -> Result<()> {
    use memory::Memory;
    let mem = MEMORIES.get_clone(handle)?;
    let entry: memory::MemoryEntry = serde_json::from_str(&entry_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid entry: {e}")))?;
    mem.store(entry)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub async fn memory_recall(handle: u32, options_json: Option<String>) -> Result<serde_json::Value> {
    use memory::Memory;
    let mem = MEMORIES.get_clone(handle)?;
    let opts: memory::RecallOptions = match options_json {
        Some(j) => serde_json::from_str(&j)
            .map_err(|e| napi::Error::from_reason(format!("Invalid options: {e}")))?,
        None => memory::RecallOptions::default(),
    };
    let entries = mem
        .recall(opts)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&entries).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub async fn memory_clear(handle: u32, session_id: Option<String>) -> Result<()> {
    let mem = MEMORIES.get_clone(handle)?;
    memory::Memory::clear(&*mem, session_id.as_deref())
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub async fn memory_stats(handle: u32) -> Result<serde_json::Value> {
    use memory::Memory;
    let mem = MEMORIES.get_clone(handle)?;
    let stats = mem
        .stats()
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&stats).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn destroy_memory(handle: u32) -> Result<()> {
    MEMORIES.remove(handle)?;
    Ok(())
}

// ============ Context ============

#[napi]
pub fn count_tokens(text: String) -> u32 {
    context::count_tokens(&text) as u32
}

#[napi]
pub fn count_tokens_for_model(text: String, model: String) -> u32 {
    context::count_tokens_for_model(&text, &model) as u32
}

#[napi]
pub fn count_message_tokens(messages: Vec<JsMessage>) -> u32 {
    let msgs: Vec<gauss_core::message::Message> =
        messages.iter().map(crate::types::js_message_to_rust).collect();
    context::count_messages_tokens(&msgs) as u32
}

#[napi]
pub fn get_context_window_size(model: String) -> u32 {
    context::context_window_size(&model) as u32
}

// ============ RAG ============

static VECTOR_STORES: HandleRegistry<Arc<rag::InMemoryVectorStore>> = HandleRegistry::new();

#[napi]
pub fn create_vector_store() -> u32 {
    VECTOR_STORES.insert(Arc::new(rag::InMemoryVectorStore::new()))
}

#[napi]
pub async fn vector_store_upsert(handle: u32, chunks_json: String) -> Result<()> {
    use rag::VectorStore;
    let store = VECTOR_STORES.get_clone(handle)?;
    let chunks: Vec<rag::Chunk> = serde_json::from_str(&chunks_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid chunks: {e}")))?;
    store
        .upsert(chunks)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub async fn vector_store_search(
    handle: u32,
    embedding_json: String,
    top_k: u32,
) -> Result<serde_json::Value> {
    use rag::VectorStore;
    let store = VECTOR_STORES.get_clone(handle)?;
    let embedding: Vec<f32> = serde_json::from_str(&embedding_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid embedding: {e}")))?;
    let results = store
        .search(&embedding, top_k as usize)
        .await
        .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
    serde_json::to_value(&results).map_err(|e| napi::Error::from_reason(format!("{e}")))
}

#[napi]
pub fn destroy_vector_store(handle: u32) -> Result<()> {
    VECTOR_STORES.remove(handle)?;
    Ok(())
}

/// Compute cosine similarity between two vectors.
#[napi]
pub fn cosine_similarity(a: Vec<f64>, b: Vec<f64>) -> f64 {
    let af: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let bf: Vec<f32> = b.iter().map(|&x| x as f32).collect();
    rag::cosine_similarity(&af, &bf) as f64
}
