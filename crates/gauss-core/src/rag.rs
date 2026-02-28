//! Embedding & RAG pipeline — document processing, vector search, retrieval.
//!
//! Provides embedding generation, text splitting, vector store abstraction,
//! and a RAG tool for agent auto-retrieval.

use crate::error;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Document & Chunk Types
// ---------------------------------------------------------------------------

/// A source document before chunking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A chunk of a document after splitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub document_id: String,
    pub content: String,
    pub index: usize,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Populated after embedding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

/// A search result with similarity score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub score: f32,
}

// ---------------------------------------------------------------------------
// Embedding Trait
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait Embedding: Send + Sync {
    /// Embed a single text into a vector.
    async fn embed(&self, text: &str) -> error::Result<Vec<f32>>;

    /// Embed multiple texts in batch.
    async fn embed_batch(&self, texts: &[&str]) -> error::Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// Dimension of the embedding vectors.
    fn dimensions(&self) -> usize;
}

#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait Embedding {
    async fn embed(&self, text: &str) -> error::Result<Vec<f32>>;

    async fn embed_batch(&self, texts: &[&str]) -> error::Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    fn dimensions(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Vector Store Trait
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Insert or update chunks with embeddings.
    async fn upsert(&self, chunks: Vec<Chunk>) -> error::Result<()>;

    /// Search by embedding vector, return top-k results.
    async fn search(&self, embedding: &[f32], top_k: usize) -> error::Result<Vec<SearchResult>>;

    /// Delete chunks by their IDs.
    async fn delete(&self, ids: &[String]) -> error::Result<()>;

    /// Delete all chunks for a given document.
    async fn delete_document(&self, document_id: &str) -> error::Result<()>;
}

#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait VectorStore {
    async fn upsert(&self, chunks: Vec<Chunk>) -> error::Result<()>;
    async fn search(&self, embedding: &[f32], top_k: usize) -> error::Result<Vec<SearchResult>>;
    async fn delete(&self, ids: &[String]) -> error::Result<()>;
    async fn delete_document(&self, document_id: &str) -> error::Result<()>;
}

// ---------------------------------------------------------------------------
// Text Splitter
// ---------------------------------------------------------------------------

/// Configuration for text splitting.
#[derive(Debug, Clone)]
pub struct SplitterConfig {
    /// Maximum chunk size in characters.
    pub chunk_size: usize,
    /// Overlap between consecutive chunks.
    pub chunk_overlap: usize,
    /// Separators to try, in order of preference.
    pub separators: Vec<String>,
}

impl Default for SplitterConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 200,
            separators: vec![
                "\n\n".to_string(),
                "\n".to_string(),
                ". ".to_string(),
                " ".to_string(),
            ],
        }
    }
}

/// Recursive character text splitter.
pub struct TextSplitter {
    config: SplitterConfig,
}

impl TextSplitter {
    pub fn new(config: SplitterConfig) -> Self {
        Self { config }
    }

    /// Split a document into chunks.
    pub fn split(&self, document: &Document) -> Vec<Chunk> {
        let texts = self.split_text(&document.content, &self.config.separators);
        texts
            .into_iter()
            .enumerate()
            .map(|(i, content)| Chunk {
                id: format!("{}_{}", document.id, i),
                document_id: document.id.clone(),
                content,
                index: i,
                metadata: document.metadata.clone(),
                embedding: None,
            })
            .collect()
    }

    fn split_text(&self, text: &str, separators: &[String]) -> Vec<String> {
        if text.len() <= self.config.chunk_size {
            return vec![text.to_string()];
        }

        let separator = separators
            .iter()
            .find(|s| text.contains(s.as_str()))
            .cloned()
            .unwrap_or_default();

        let remaining_separators = if !separator.is_empty() {
            separators
                .iter()
                .skip_while(|s| s.as_str() != separator.as_str())
                .skip(1)
                .cloned()
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let splits: Vec<&str> = if separator.is_empty() {
            // Character-level split
            let mut result = Vec::new();
            let mut start = 0;
            while start < text.len() {
                let end = (start + self.config.chunk_size).min(text.len());
                result.push(&text[start..end]);
                start = if end > self.config.chunk_overlap {
                    end - self.config.chunk_overlap
                } else {
                    end
                };
                if start >= text.len() {
                    break;
                }
            }
            return result.into_iter().map(String::from).collect();
        } else {
            text.split(&separator).collect()
        };

        let mut chunks = Vec::new();
        let mut current = String::new();

        for split in splits {
            let candidate = if current.is_empty() {
                split.to_string()
            } else {
                format!("{}{}{}", current, separator, split)
            };

            if candidate.len() > self.config.chunk_size && !current.is_empty() {
                chunks.push(current.clone());
                // Overlap: start next chunk with end of current
                let overlap_start = if current.len() > self.config.chunk_overlap {
                    current.len() - self.config.chunk_overlap
                } else {
                    0
                };
                current = format!("{}{}{}", &current[overlap_start..], separator, split);
                if current.len() > self.config.chunk_size && !remaining_separators.is_empty() {
                    let sub = self.split_text(&current, &remaining_separators);
                    chunks.extend(sub);
                    current = String::new();
                }
            } else {
                current = candidate;
            }
        }

        if !current.is_empty() {
            chunks.push(current);
        }

        chunks
    }
}

impl Default for TextSplitter {
    fn default() -> Self {
        Self::new(SplitterConfig::default())
    }
}

// ---------------------------------------------------------------------------
// In-Memory Vector Store
// ---------------------------------------------------------------------------

/// Simple in-memory vector store with cosine similarity.
#[derive(Debug, Default)]
pub struct InMemoryVectorStore {
    chunks: std::sync::Mutex<Vec<Chunk>>,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self::default()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl VectorStore for InMemoryVectorStore {
    async fn upsert(&self, new_chunks: Vec<Chunk>) -> error::Result<()> {
        let mut store = self
            .chunks
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;
        for chunk in new_chunks {
            if let Some(existing) = store.iter_mut().find(|c| c.id == chunk.id) {
                *existing = chunk;
            } else {
                store.push(chunk);
            }
        }
        Ok(())
    }

    async fn search(&self, embedding: &[f32], top_k: usize) -> error::Result<Vec<SearchResult>> {
        let store = self
            .chunks
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;

        let mut scored: Vec<SearchResult> = store
            .iter()
            .filter_map(|chunk| {
                chunk.embedding.as_ref().map(|emb| SearchResult {
                    chunk: chunk.clone(),
                    score: cosine_similarity(embedding, emb),
                })
            })
            .collect();

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        Ok(scored)
    }

    async fn delete(&self, ids: &[String]) -> error::Result<()> {
        let mut store = self
            .chunks
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;
        store.retain(|c| !ids.contains(&c.id));
        Ok(())
    }

    async fn delete_document(&self, document_id: &str) -> error::Result<()> {
        let mut store = self
            .chunks
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;
        store.retain(|c| c.document_id != document_id);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RAG Pipeline
// ---------------------------------------------------------------------------

/// High-level RAG pipeline: ingest documents → chunk → embed → store → query.
pub struct RagPipeline {
    pub embedding: crate::Shared<dyn Embedding>,
    pub vector_store: crate::Shared<dyn VectorStore>,
    pub splitter: TextSplitter,
}

impl RagPipeline {
    pub fn new(
        embedding: crate::Shared<dyn Embedding>,
        vector_store: crate::Shared<dyn VectorStore>,
        splitter: TextSplitter,
    ) -> Self {
        Self {
            embedding,
            vector_store,
            splitter,
        }
    }

    /// Ingest a document: split → embed → store.
    pub async fn ingest(&self, document: Document) -> error::Result<usize> {
        let mut chunks = self.splitter.split(&document);
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        let embeddings = self.embedding.embed_batch(&texts).await?;

        for (chunk, emb) in chunks.iter_mut().zip(embeddings) {
            chunk.embedding = Some(emb);
        }

        let count = chunks.len();
        self.vector_store.upsert(chunks).await?;
        Ok(count)
    }

    /// Query the pipeline: embed query → search → return results.
    pub async fn query(&self, query: &str, top_k: usize) -> error::Result<Vec<SearchResult>> {
        let embedding = self.embedding.embed(query).await?;
        self.vector_store.search(&embedding, top_k).await
    }

    /// Delete all chunks for a document.
    pub async fn remove_document(&self, document_id: &str) -> error::Result<()> {
        self.vector_store.delete_document(document_id).await
    }
}
