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
```

## Features

- **Multi-provider**: OpenAI, Anthropic, Google with automatic retry
- **Agent loop**: Tool calling with stop conditions (MaxSteps, HasToolCall, TextGenerated)
- **Streaming**: SSE-based token streaming with AgentStreamEvent
- **Structured output**: JSON Schema validation via jsonschema
- **Callbacks**: on_step_finish, on_tool_call hooks
- **Workflow**: DAG-based multi-step pipelines with dependency tracking
- **Team**: Multi-agent coordination (Sequential + Parallel strategies)
- **4 targets**: Native (NAPI), Browser (WASM), Python (PyO3), CLI
- **TypeScript SDK**: Auto-detects NAPI → WASM backend

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
