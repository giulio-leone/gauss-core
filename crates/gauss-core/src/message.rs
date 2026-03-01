use serde::{Deserialize, Serialize};

/// Message role in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Content part within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    Text {
        text: String,
    },
    Image {
        #[serde(skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        base64: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
    },
    Audio {
        #[serde(skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        base64: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    ToolResult {
        tool_call_id: String,
        content: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    Reasoning {
        text: String,
    },
    File {
        #[serde(skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        base64: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
    },
    /// Document for citation-aware responses (Anthropic).
    Document {
        source_type: String, // "text", "base64", "url"
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(default)]
        citations_enabled: bool,
    },
}

/// A citation referencing a source document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Citation type: "char_location", "page_location", "content_block_location".
    pub citation_type: String,
    /// The cited text from the source document.
    pub cited_text: String,
    /// Document title (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_title: Option<String>,
    /// Start index (character offset for char_location, page for page_location).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<u64>,
    /// End index.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<u64>,
}

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![Content::Text { text: text.into() }],
            name: None,
        }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![Content::Text { text: text.into() }],
            name: None,
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![Content::Text { text: text.into() }],
            name: None,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: serde_json::Value) -> Self {
        Self {
            role: Role::Tool,
            content: vec![Content::ToolResult {
                tool_call_id: tool_call_id.into(),
                content,
                is_error: None,
            }],
            name: None,
        }
    }

    /// Extract the first text content from this message.
    pub fn text(&self) -> Option<&str> {
        self.content.iter().find_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            _ => None,
        })
    }

    /// Extract all tool calls from this message.
    pub fn tool_calls(&self) -> Vec<(&str, &str, &serde_json::Value)> {
        self.content
            .iter()
            .filter_map(|c| match c {
                Content::ToolCall {
                    id,
                    name,
                    arguments,
                } => Some((id.as_str(), name.as_str(), arguments)),
                _ => None,
            })
            .collect()
    }
}

/// Usage statistics from a generation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_tokens: Option<u64>,
}

impl Usage {
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}
