# Gauss Core

> High-performance AI agent engine in Rust. NAPI (Node.js) + WASM (Browser) + PyO3 (Python) + CLI.

## Architecture

```
gauss-core (Rust workspace)
├── crates/
│   ├── gauss-core/        # Core library (providers, agent, streaming, tools, workflow, team)
│   ├── gauss-napi/        # NAPI-RS bindings for Node.js
│   ├── gauss-wasm/        # wasm-bindgen bindings for browser/edge
│   ├── gauss-python/      # PyO3 bindings for Python
│   └── gauss-cli/         # Standalone CLI binary
├── packages/
│   └── gauss/             # TypeScript SDK (auto-detect NAPI → WASM)
```

## Quick Start

### TypeScript / Node.js

```typescript
import { createProvider, createAgent } from '@giulio-leone/gauss';

const provider = createProvider('openai', 'gpt-5.2', {
  apiKey: process.env.OPENAI_API_KEY!,
});

const agent = createAgent('my-agent', provider, {
  instructions: 'You are a helpful assistant.',
  temperature: 0.7,
  maxSteps: 10,
});

const result = await agent.run([{ role: 'user', content: 'Hello!' }]);
console.log(result.text);

provider.destroy();
```

### Rust

```rust
use gauss_core::agent::Agent;
use gauss_core::message::Message;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::ProviderConfig;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let provider = Arc::new(OpenAiProvider::new(
        "gpt-5.2",
        ProviderConfig::new(std::env::var("OPENAI_API_KEY").unwrap()),
    ));

    let agent = Agent::builder("my-agent", provider)
        .instructions("You are a helpful assistant.")
        .temperature(0.7)
        .max_steps(10)
        .build();

    let output = agent.run(vec![Message::user("Hello!")]).await.unwrap();
    println!("{}", output.text);
}
```

### Python

```python
import gauss_core

provider = gauss_core.create_provider("openai", "gpt-5.2", "sk-...")
result = await gauss_core.generate(provider, '[{"role":"user","content":"Hello!"}]')
print(result)
gauss_core.destroy_provider(provider)
```

### CLI

```bash
# Interactive chat
gauss chat -p openai -m gpt-5.2

# Single prompt
gauss chat -p anthropic -m claude-sonnet-4-20250514 "Explain quantum computing"

# List providers
gauss providers

# Scaffold a new project
gauss init my-agent --template basic    # basic | rag | multi-agent

# Run an agent from config
cd my-agent
gauss run                               # reads gauss.yaml
gauss run "What is the meaning of life?"
```

## Features

- **Multi-provider**: OpenAI, Anthropic, Google, Groq, Ollama, DeepSeek with automatic retry
- **Agent loop**: Tool calling with stop conditions (MaxSteps, HasToolCall, TextGenerated)
- **Streaming**: SSE-based token streaming with AgentStreamEvent + stream transformers
- **Structured output**: JSON Schema validation via jsonschema
- **Callbacks**: on_step_finish, on_tool_call hooks
- **Workflow**: DAG-based multi-step pipelines with dependency tracking
- **Team**: Multi-agent coordination (Sequential + Parallel strategies)
- **Memory**: Conversation, Working, Semantic memory with pluggable backends
- **RAG**: Embedding trait, text splitting, vector store, retrieval pipeline
- **MCP**: Model Context Protocol — client & server, JSON-RPC, stdio/HTTP/SSE transports
- **Agent Network**: A2A protocol, routing, delegation, supervisor/worker topology
- **Middleware**: Before/after hooks with priority ordering, logging, caching
- **Context Management**: tiktoken-rs precise counting, pruning strategies, sliding window
- **Human-in-the-Loop**: Approval gates, checkpoints, suspend/resume
- **Evaluation**: Scorer trait, built-in scorers, dataset loading, batch runner
- **Observability**: Span tracing, agent metrics, telemetry collector
- **Guardrails**: Input/output validation — content moderation, PII detection, token limits, regex filters, schema validation
- **Resilience**: Fallback provider chains, circuit breaker (3-state), composed retry+fallback+CB
- **Stream Transforms**: Partial JSON parser, object accumulator, map/filter/tap transformers, pipeline composition
- **Plugin System**: Plugin lifecycle, event bus (pub/sub), registry with topological dependency sort
- **Agent Patterns**: ToolValidator (multi-stage coercion), ToolChain (sequential composition), ReflectionAgent, PlanningAgent
- **Config DSL**: Declarative agent definition from JSON/YAML/TOML with `${ENV}` resolution
- **Benchmarks**: Criterion micro-benchmarks for hot paths (JSON parsing, validation, serialization)
- **Property Tests**: proptest-based fuzzing for partial JSON, validators, message serde, PII regex
- **CLI**: `gauss init`, `gauss run`, `gauss chat`, project templates
- **4 targets**: Native (NAPI), Browser (WASM), Python (PyO3), CLI

## Multi-Agent Team

```rust
use gauss_core::team::{Team, Strategy};

let team = Team::builder("research-team")
    .agent(researcher)
    .agent(writer)
    .agent(editor)
    .strategy(Strategy::Sequential)
    .build();

let output = team.run(messages).await?;
```

## Workflow

```rust
use gauss_core::workflow::Workflow;

let workflow = Workflow::builder()
    .agent_step("research", researcher, |_| messages.clone())
    .agent_step("write", writer, |ctx| {
        vec![Message::user(&ctx["research"].text)]
    })
    .dependency("write", "research")
    .build();

let results = workflow.run(messages).await?;
```

## Memory

```rust
use gauss_core::memory::{InMemoryMemory, Memory, MemoryEntry, MemoryTier, RecallOptions};

let mem = InMemoryMemory::new();
let entry = MemoryEntry::new(MemoryTier::Short, serde_json::json!("User prefers Rust"));
mem.store(entry).await?;

let results = mem.recall(RecallOptions::default()).await?;
```

## RAG Pipeline

```rust
use gauss_core::rag::{InMemoryVectorStore, TextSplitter, SplitterConfig, RagPipeline};

let splitter = TextSplitter::new(SplitterConfig { chunk_size: 512, ..Default::default() });
let store = InMemoryVectorStore::new();
// Split documents, embed, store, then search
```

## Agent Network

```rust
use gauss_core::network::{AgentNetwork, AgentNode, AgentCard};

let mut network = AgentNetwork::new();
network.add_agent(AgentNode {
    agent: researcher,
    card: AgentCard { name: "researcher".into(), ..Default::default() },
    connections: vec!["writer".into()],
});
network.set_supervisor("coordinator");

let result = network.delegate("researcher", messages).await?;
```

## MCP Transport

```rust
use gauss_core::mcp::*;

// Server: expose tools over stdio
let mut server = McpServer::new("my-server", "1.0.0");
server.add_tool(my_tool);
let transport = StdioTransport::new();
serve(&server, &transport).await?;

// Client: connect to an external MCP server
let transport = ChildProcessTransport::spawn("npx", &["-y", "@some/mcp-server"])?;
let mut client = TransportMcpClient::new(transport);
let caps = client.initialize().await?;
let tools = client.list_tools().await?;

// Client: connect via HTTP
let transport = HttpTransport::new("http://localhost:8080/mcp");
let mut client = TransportMcpClient::new(transport);
```

## Token Counting (tiktoken)

```rust
use gauss_core::context::{count_tokens, count_tokens_for_model};

let tokens = count_tokens("Hello, world!");       // cl100k_base (precise)
let tokens = count_tokens_for_model("Hello!", "gpt-4o"); // model-specific encoding
```

## Python

```python
import asyncio
from gauss_core import (
    create_provider, generate, create_memory, memory_store,
    count_tokens, create_vector_store, destroy_provider,
)

async def main():
    provider = create_provider("openai", "gpt-5.2", "sk-...")
    result = await generate(provider, '[{"role":"user","content":"Hello!"}]')
    print(result)

    # Memory
    mem = create_memory()
    await memory_store(mem, '{"tier":"Short","content":"User likes Rust"}')

    # Token counting (tiktoken-powered)
    tokens = count_tokens("Hello, world!")

    destroy_provider(provider)

asyncio.run(main())
```

## Guardrails

```rust
use gauss_core::guardrail::*;

let mut chain = GuardrailChain::new();

// Content moderation (block harmful patterns)
chain.add(Arc::new(ContentModerationGuardrail::new()
    .block_pattern("hack", "No hacking instructions")));

// PII detection (redact emails, phones, SSNs, credit cards)
chain.add(Arc::new(PiiDetectionGuardrail::new(PiiAction::Redact)));

// Token limits
chain.add(Arc::new(TokenLimitGuardrail::new().max_input(4096)));

// Validate input
let messages = vec![Message::user("Hello, my email is test@example.com")];
let result = chain.validate_input(&messages);
match result.action {
    GuardrailAction::Allow => println!("Safe!"),
    GuardrailAction::Block { reason } => println!("Blocked: {reason}"),
    GuardrailAction::Rewrite { rewritten, .. } => println!("Rewritten: {rewritten}"),
    _ => {}
}
```

## Resilience

```rust
use gauss_core::resilience::*;

// Fallback chain: try primary, then fallback providers in order
let fallback = FallbackProvider::new(vec![primary, backup1, backup2]);

// Circuit breaker: trip after 5 failures, recover after 30s
let cb = CircuitBreaker::new(provider, CircuitBreakerConfig::default());

// Compose: retry → circuit breaker → fallback
let resilient = ResilientProviderBuilder::new(primary)
    .retry(RetryConfig::default())
    .circuit_breaker(CircuitBreakerConfig::default())
    .fallback(backup)
    .build();
```

## Stream Transforms

```rust
use gauss_core::stream_transform::*;

// Parse partial JSON from streaming chunks
let value = parse_partial_json(r#"{"name": "test", "age": 2"#);
// → Some({"name": "test", "age": 2})

// Object accumulator (deduplicate partial updates)
let mut acc = ObjectAccumulator::new();
acc.feed("{");           // → Some({})
acc.feed(r#""key": 1"#); // → Some({"key": 1})

// Stream pipeline with transformers
let pipeline = StreamPipeline::new()
    .add(MapText::new(|s| s.to_uppercase()))
    .add(FilterEvents::new(|e| !matches!(e, StreamEvent::Done)));
```

## Plugin System

```rust
use gauss_core::plugin::*;

let mut registry = PluginRegistry::new();
registry.register(Arc::new(TelemetryPlugin));   // built-in
registry.register(Arc::new(MemoryPlugin));       // built-in

// Initialize all plugins (topologically sorted by dependencies)
registry.init_all().unwrap();

// Emit events
registry.emit(&GaussEvent::AgentStart {
    agent_id: "agent-1".into(),
    model: "gpt-4o".into(),
});
```

## Agent Patterns

```rust
use gauss_core::patterns::*;

// Tool input validation with multi-stage coercion
let validator = ToolValidator::new();
let coerced = validator.validate(input, &schema)?;

// Sequential tool composition
let chain = ToolChain::new("pipeline")
    .add(fetch_tool)
    .add(parse_tool)
    .add(summarize_tool);
let result = chain.execute(initial_input, &provider).await?;
```

## Config DSL

```yaml
# gauss.yaml
name: my-agent
provider: { kind: openai, model: gpt-5.2, api_key: "${OPENAI_API_KEY}" }
instructions: "You are a helpful assistant."
options: { temperature: 0.7, max_tokens: 4096 }
tools:
  - name: search
    description: "Search the web"
    parameters: { type: object, properties: { query: { type: string } } }
```

```rust
let config = AgentConfig::from_file("gauss.yaml")?;
```

## Building

```bash
# Core library
cargo build -p gauss-core

# All crates
cargo build --workspace

# CLI (release)
cargo build -p gauss-cli --release

# Run tests
cargo test -p gauss-core

# Clippy
cargo clippy --workspace -- -D warnings
```

## License

MIT
