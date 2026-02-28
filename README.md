# Gauss Core

> High-performance AI agent engine in Rust. NAPI (Node.js) + WASM (Browser) + PyO3 (Python) + CLI.

## Architecture

```
gauss-core (Rust workspace)
├── crates/
│   ├── gauss-core/        # Core library (providers, agent, streaming, tools)
│   ├── gauss-napi/        # NAPI-RS bindings for Node.js
│   ├── gauss-wasm/        # wasm-bindgen bindings for browser/edge
│   ├── gauss-python/      # PyO3 bindings for Python
│   └── gauss-cli/         # Standalone CLI binary
```

## Quick Start (Rust)

```rust
use gauss_core::agent::Agent;
use gauss_core::message::Message;
use gauss_core::provider::openai::OpenAiProvider;
use gauss_core::provider::ProviderConfig;
use gauss_core::tool::Tool;
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

## Features

- **Multi-provider**: OpenAI, Anthropic, Google (more coming)
- **Agent loop**: Automatic tool calling with configurable stop conditions
- **Streaming**: SSE-based token streaming with backpressure
- **Structured output**: JSON Schema validation
- **Reasoning**: First-class `reasoning_effort` support
- **All generation params**: temperature, top_p, top_k, seed, penalties, etc.
- **4 targets**: Native (NAPI), Browser (WASM), Python (PyO3), CLI

## Building

```bash
# Core library
cargo build -p gauss-core

# CLI
cargo build -p gauss-cli --release

# Run tests
cargo test
```

## License

MIT
