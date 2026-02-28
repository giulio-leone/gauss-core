//! Memory system — persistent agent memory with tiers.
//!
//! Provides conversation history, working memory (key-value with TTL),
//! and semantic memory (vector-based recall).

use crate::error;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Memory tier classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTier {
    Short,
    Working,
    Semantic,
    Observation,
}

/// Memory entry type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryEntryType {
    Conversation,
    Fact,
    Preference,
    Task,
    Summary,
}

/// A single memory entry stored by the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub content: String,
    pub entry_type: MemoryEntryType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tier: Option<MemoryTier>,
    pub timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    /// Importance score 0.0–1.0
    #[serde(skip_serializing_if = "Option::is_none")]
    pub importance: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Embedding vector (populated when semantic memory is used).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

/// Options for recalling memories.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecallOptions {
    pub limit: Option<usize>,
    pub entry_type: Option<MemoryEntryType>,
    pub tier: Option<MemoryTier>,
    pub include_tiers: Option<Vec<MemoryTier>>,
    pub session_id: Option<String>,
    pub min_importance: Option<f64>,
    /// Text query for keyword/semantic search.
    pub query: Option<String>,
}

/// Memory statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_entries: usize,
    pub by_type: HashMap<String, usize>,
    pub by_tier: Option<HashMap<String, usize>>,
    pub oldest_entry: Option<String>,
    pub newest_entry: Option<String>,
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Core memory trait — must be implemented by all memory backends.
#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait Memory: Send + Sync {
    /// Store a memory entry.
    async fn store(&self, entry: MemoryEntry) -> error::Result<()>;

    /// Recall memories matching the given options.
    async fn recall(&self, options: RecallOptions) -> error::Result<Vec<MemoryEntry>>;

    /// Summarize a set of entries into a single string.
    async fn summarize(&self, entries: &[MemoryEntry]) -> error::Result<String>;

    /// Clear all entries (optionally filtered by session).
    async fn clear(&self, session_id: Option<&str>) -> error::Result<()>;

    /// Get memory statistics.
    async fn stats(&self) -> error::Result<MemoryStats>;
}

#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait Memory {
    async fn store(&self, entry: MemoryEntry) -> error::Result<()>;
    async fn recall(&self, options: RecallOptions) -> error::Result<Vec<MemoryEntry>>;
    async fn summarize(&self, entries: &[MemoryEntry]) -> error::Result<String>;
    async fn clear(&self, session_id: Option<&str>) -> error::Result<()>;
    async fn stats(&self) -> error::Result<MemoryStats>;
}

// ---------------------------------------------------------------------------
// Working Memory Trait (key-value with TTL)
// ---------------------------------------------------------------------------

/// A key-value entry with optional expiration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemoryEntry {
    pub key: String,
    pub value: serde_json::Value,
    pub created_at: u64,
    /// Epoch millis. 0 = no expiry.
    pub expires_at: u64,
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait WorkingMemory: Send + Sync {
    async fn get(&self, key: &str) -> error::Result<Option<serde_json::Value>>;
    async fn set(
        &self,
        key: &str,
        value: serde_json::Value,
        ttl_ms: Option<u64>,
    ) -> error::Result<()>;
    async fn delete(&self, key: &str) -> error::Result<bool>;
    async fn list(&self) -> error::Result<Vec<WorkingMemoryEntry>>;
    async fn clear(&self) -> error::Result<()>;
}

#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait WorkingMemory {
    async fn get(&self, key: &str) -> error::Result<Option<serde_json::Value>>;
    async fn set(
        &self,
        key: &str,
        value: serde_json::Value,
        ttl_ms: Option<u64>,
    ) -> error::Result<()>;
    async fn delete(&self, key: &str) -> error::Result<bool>;
    async fn list(&self) -> error::Result<Vec<WorkingMemoryEntry>>;
    async fn clear(&self) -> error::Result<()>;
}

// ---------------------------------------------------------------------------
// In-Memory Backend
// ---------------------------------------------------------------------------

/// In-memory implementation of both Memory and WorkingMemory.
#[derive(Debug, Default)]
pub struct InMemoryMemory {
    entries: std::sync::Mutex<Vec<MemoryEntry>>,
    kv: std::sync::Mutex<HashMap<String, WorkingMemoryEntry>>,
}

impl InMemoryMemory {
    pub fn new() -> Self {
        Self::default()
    }

    fn now_millis() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Memory for InMemoryMemory {
    async fn store(&self, entry: MemoryEntry) -> error::Result<()> {
        self.entries
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?
            .push(entry);
        Ok(())
    }

    async fn recall(&self, options: RecallOptions) -> error::Result<Vec<MemoryEntry>> {
        let entries = self
            .entries
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;

        let limit = options.limit.unwrap_or(10);
        let mut results: Vec<MemoryEntry> = entries
            .iter()
            .filter(|e| {
                if options
                    .entry_type
                    .as_ref()
                    .is_some_and(|t| &e.entry_type != t)
                {
                    return false;
                }
                if options
                    .tier
                    .as_ref()
                    .is_some_and(|tier| e.tier.as_ref() != Some(tier))
                {
                    return false;
                }
                if options
                    .include_tiers
                    .as_ref()
                    .zip(e.tier.as_ref())
                    .is_some_and(|(tiers, et)| !tiers.contains(et))
                {
                    return false;
                }
                if options
                    .session_id
                    .as_ref()
                    .is_some_and(|sid| e.session_id.as_ref() != Some(sid))
                {
                    return false;
                }
                if options
                    .min_importance
                    .is_some_and(|min| e.importance.unwrap_or(0.0) < min)
                {
                    return false;
                }
                if let Some(ref q) = options.query {
                    let q_lower = q.to_lowercase();
                    if !e.content.to_lowercase().contains(&q_lower) {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();

        // Most recent first
        results.reverse();
        results.truncate(limit);
        Ok(results)
    }

    async fn summarize(&self, entries: &[MemoryEntry]) -> error::Result<String> {
        // Simple concatenation for in-memory; real impl would use LLM.
        let summary = entries
            .iter()
            .map(|e| e.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        Ok(summary)
    }

    async fn clear(&self, session_id: Option<&str>) -> error::Result<()> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;
        if let Some(sid) = session_id {
            entries.retain(|e| e.session_id.as_deref() != Some(sid));
        } else {
            entries.clear();
        }
        Ok(())
    }

    async fn stats(&self) -> error::Result<MemoryStats> {
        let entries = self
            .entries
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;

        let mut by_type: HashMap<String, usize> = HashMap::new();
        let mut by_tier: HashMap<String, usize> = HashMap::new();
        let mut oldest: Option<&str> = None;
        let mut newest: Option<&str> = None;

        for e in entries.iter() {
            *by_type.entry(format!("{:?}", e.entry_type)).or_insert(0) += 1;
            if let Some(ref t) = e.tier {
                *by_tier.entry(format!("{t:?}")).or_insert(0) += 1;
            }
            let ts = e.timestamp.as_str();
            if oldest.is_none() || ts < oldest.unwrap_or("") {
                oldest = Some(ts);
            }
            if newest.is_none() || ts > newest.unwrap_or("") {
                newest = Some(ts);
            }
        }

        Ok(MemoryStats {
            total_entries: entries.len(),
            by_type,
            by_tier: Some(by_tier),
            oldest_entry: oldest.map(String::from),
            newest_entry: newest.map(String::from),
        })
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl WorkingMemory for InMemoryMemory {
    async fn get(&self, key: &str) -> error::Result<Option<serde_json::Value>> {
        let kv = self
            .kv
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;
        let now = Self::now_millis();
        if let Some(entry) = kv.get(key) {
            if entry.expires_at > 0 && entry.expires_at <= now {
                return Ok(None);
            }
            Ok(Some(entry.value.clone()))
        } else {
            Ok(None)
        }
    }

    async fn set(
        &self,
        key: &str,
        value: serde_json::Value,
        ttl_ms: Option<u64>,
    ) -> error::Result<()> {
        let now = Self::now_millis();
        let expires_at = ttl_ms.map_or(0, |ttl| now + ttl);
        let entry = WorkingMemoryEntry {
            key: key.to_string(),
            value,
            created_at: now,
            expires_at,
        };
        self.kv
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?
            .insert(key.to_string(), entry);
        Ok(())
    }

    async fn delete(&self, key: &str) -> error::Result<bool> {
        let removed = self
            .kv
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?
            .remove(key)
            .is_some();
        Ok(removed)
    }

    async fn list(&self) -> error::Result<Vec<WorkingMemoryEntry>> {
        let kv = self
            .kv
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?;
        let now = Self::now_millis();
        Ok(kv
            .values()
            .filter(|e| e.expires_at == 0 || e.expires_at > now)
            .cloned()
            .collect())
    }

    async fn clear(&self) -> error::Result<()> {
        self.kv
            .lock()
            .map_err(|e| error::GaussError::internal(e.to_string()))?
            .clear();
        Ok(())
    }
}
