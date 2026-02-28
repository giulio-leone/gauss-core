use std::fs;
use std::path::Path;

const TEMPLATES: &[&str] = &["basic", "rag", "multi-agent"];

pub fn scaffold_project(name: &str, template: &str) -> Result<(), String> {
    if !TEMPLATES.contains(&template) {
        return Err(format!(
            "Unknown template '{template}'. Available: {}",
            TEMPLATES.join(", ")
        ));
    }

    let root = Path::new(name);
    if root.exists() {
        return Err(format!("Directory '{name}' already exists"));
    }

    fs::create_dir_all(root.join("src")).map_err(|e| e.to_string())?;

    write_file(root.join("gauss.yaml"), &agent_yaml(name, template))?;
    write_file(root.join("src/main.rs"), &main_rs(template))?;
    write_file(root.join("Cargo.toml"), &cargo_toml(name))?;
    write_file(root.join(".env.example"), &env_example(template))?;
    write_file(root.join("README.md"), &readme(name, template))?;

    println!("✅ Created project '{name}' with template '{template}'");
    println!();
    println!("  cd {name}");
    println!("  cp .env.example .env    # add your API keys");
    println!("  gauss run               # run the agent");
    println!();
    Ok(())
}

fn write_file(path: impl AsRef<Path>, content: &str) -> Result<(), String> {
    fs::write(&path, content).map_err(|e| format!("{}: {e}", path.as_ref().display()))
}

fn agent_yaml(name: &str, template: &str) -> String {
    match template {
        "basic" => format!(
            r#"# Gauss Agent Configuration
name: {name}
provider: openai
model: gpt-4o

instructions: |
  You are a helpful AI assistant called {name}.
  Be concise and accurate.

tools: []
"#
        ),
        "rag" => format!(
            r#"# Gauss Agent Configuration — RAG template
name: {name}
provider: openai
model: gpt-4o

instructions: |
  You are a knowledge assistant called {name}.
  Answer questions using the provided context from documents.
  Always cite your sources.

memory:
  type: conversation
  max_messages: 50

rag:
  embedding_model: text-embedding-3-small
  chunk_size: 512
  chunk_overlap: 64
  vector_store: in_memory
  documents_dir: ./docs

tools: []
"#
        ),
        "multi-agent" => format!(
            r#"# Gauss Agent Network Configuration
name: {name}
provider: openai
model: gpt-4o

network:
  supervisor:
    name: coordinator
    instructions: |
      You are a coordinator agent. Delegate tasks to the appropriate specialist.
      - Use 'researcher' for information gathering
      - Use 'writer' for content creation

  agents:
    - name: researcher
      model: gpt-4o
      instructions: |
        You are a research specialist. Find and synthesize information.

    - name: writer
      model: gpt-4o
      instructions: |
        You are a writing specialist. Create well-structured content.

tools: []
"#
        ),
        _ => unreachable!(),
    }
}

fn main_rs(template: &str) -> String {
    match template {
        "basic" => r#"//! Gauss Basic Agent
//! Run with: gauss run

use gauss_core::agent::Agent;
use gauss_core::message::Message;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::ProviderConfig;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let provider = Arc::new(OpenAiProvider::new("gpt-4o", ProviderConfig::new(&api_key)));

    let agent = Agent::builder("assistant")
        .instructions("You are a helpful AI assistant.")
        .provider(provider)
        .build();

    let messages = vec![Message::user("Hello! What can you help me with?")];
    let response = agent.generate(&messages).await?;
    println!("{}", response.text().unwrap_or(""));
    Ok(())
}
"#
        .to_string(),
        "rag" => r#"//! Gauss RAG Agent
//! Run with: gauss run

use gauss_core::agent::Agent;
use gauss_core::message::Message;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::ProviderConfig;
use gauss_core::rag::{InMemoryVectorStore, RagPipeline, SplitterConfig, TextSplitter};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let provider = Arc::new(OpenAiProvider::new("gpt-4o", ProviderConfig::new(&api_key)));

    // Set up RAG pipeline
    let _splitter = TextSplitter::new(SplitterConfig {
        chunk_size: 512,
        chunk_overlap: 64,
        ..Default::default()
    });
    let _store = InMemoryVectorStore::new();

    let agent = Agent::builder("rag-assistant")
        .instructions("Answer questions using the provided context. Cite your sources.")
        .provider(provider)
        .build();

    let messages = vec![Message::user("What do the documents say?")];
    let response = agent.generate(&messages).await?;
    println!("{}", response.text().unwrap_or(""));
    Ok(())
}
"#
        .to_string(),
        "multi-agent" => r#"//! Gauss Multi-Agent Network
//! Run with: gauss run

use gauss_core::agent::Agent;
use gauss_core::message::Message;
use gauss_core::network::AgentNetwork;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::ProviderConfig;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let config = ProviderConfig::new(&api_key);

    let researcher = Agent::builder("researcher")
        .instructions("You are a research specialist.")
        .provider(Arc::new(OpenAiProvider::new("gpt-4o", config.clone())))
        .build();

    let writer = Agent::builder("writer")
        .instructions("You are a writing specialist.")
        .provider(Arc::new(OpenAiProvider::new("gpt-4o", config.clone())))
        .build();

    let coordinator = Agent::builder("coordinator")
        .instructions("Delegate tasks: use 'researcher' for info, 'writer' for content.")
        .provider(Arc::new(OpenAiProvider::new("gpt-4o", config)))
        .build();

    let mut network = AgentNetwork::new();
    network.add_agent(researcher);
    network.add_agent(writer);
    network.set_supervisor(coordinator);

    println!("Multi-agent network ready with 3 agents.");
    Ok(())
}
"#
        .to_string(),
        _ => unreachable!(),
    }
}

fn cargo_toml(name: &str) -> String {
    format!(
        r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2024"

[dependencies]
gauss-core = {{ version = "0.3", features = ["native"] }}
tokio = {{ version = "1", features = ["full"] }}
"#
    )
}

fn env_example(template: &str) -> String {
    let mut s = String::from("# Gauss Environment Variables\nOPENAI_API_KEY=sk-...\n");
    if template == "rag" {
        s.push_str("# OPENAI_EMBEDDING_MODEL=text-embedding-3-small\n");
    }
    if template == "multi-agent" {
        s.push_str("# ANTHROPIC_API_KEY=sk-ant-...\n");
    }
    s
}

fn readme(name: &str, template: &str) -> String {
    format!(
        r#"# {name}

A Gauss AI agent project using the **{template}** template.

## Quick Start

```bash
cp .env.example .env   # add your API keys
gauss run              # run the agent
```

## Configuration

Edit `gauss.yaml` to customize your agent's behavior, model, and tools.

## Learn More

- [Gauss Documentation](https://github.com/giulio-leone/gauss-core)
"#
    )
}
