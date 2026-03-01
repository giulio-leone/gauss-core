<div align="center">

# 🔮 gauss-core

**The Rust engine powering the Gauss AI SDK**

[![CI](https://github.com/giulio-leone/gauss-core/actions/workflows/ci.yml/badge.svg)](https://github.com/giulio-leone/gauss-core/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Rust](https://img.shields.io/badge/rust-stable-orange.svg)

</div>

---

High-performance, multi-provider AI agent framework written in Rust. Powers [gauss-ts](https://github.com/giulio-leone/gauss) and [gauss-py](https://github.com/giulio-leone/gauss-py) via native bindings.

## ✨ Features

| Category | Capabilities |
|----------|-------------|
| **Providers** | OpenAI, Anthropic, Google, DeepSeek, Groq, Ollama, OpenRouter |
| **Agents** | Multi-step tool-calling, configurable stop conditions, streaming |
| **Teams** | Round-robin, consensus strategies, parallel execution |
| **Tools** | Type-safe definitions, validation, examples, batch execution, ToolRegistry |
| **Graphs** | DAG pipelines with conditional routing |
| **Workflows** | Sequential step-by-step processing |
| **Networks** | Agent-to-agent delegation with supervisor oversight |
| **MCP** | Model Context Protocol server and client |
| **Memory** | Semantic recall with session isolation |
| **RAG** | Vector store with cosine similarity search |
| **Reasoning** | Anthropic extended thinking + OpenAI reasoning effort |
| **Code Exec** | Python, JavaScript, Bash sandboxed runtimes |
| **Middleware** | Composable request/response interceptors |
| **Guardrails** | Content filtering and safety checks |
| **Telemetry** | Token tracking, latency metrics, cost estimation |
| **HITL** | Human-in-the-loop approval workflows |
| **Resilience** | Circuit breaker, retry, fallback patterns |

## 🏗️ Architecture

```
gauss-core/
├── crates/
│   ├── gauss-core/      # Core library — 24 modules, 100+ structs, 15 traits
│   ├── gauss-napi/      # Node.js N-API bindings — 75 exported functions
│   └── gauss-python/    # Python PyO3 bindings — 75 exported functions
```

## 🔧 Building

```bash
# Core library
cargo build --release

# Node.js bindings (for gauss-ts)
cargo build -p gauss-napi --release

# Python bindings (for gauss-py)
cargo build -p gauss-python --release
```

## 🧪 Testing

```bash
cargo test --workspace   # 326 tests
```

## 📦 Bindings

gauss-core compiles to native bindings for both Node.js and Python:

- **NAPI** (`gauss-napi`): Loaded by gauss-ts as a native `.node` module
- **PyO3** (`gauss-python`): Loaded by gauss-py as a native `.so`/`.dylib` module

Both binding crates expose the full feature set with automatic snake_case → camelCase conversion for JavaScript.

## 🔗 Related

| Package | Language | Repository |
|---------|----------|------------|
| **gauss-ts** | TypeScript | [giulio-leone/gauss](https://github.com/giulio-leone/gauss) |
| **gauss-py** | Python | [giulio-leone/gauss-py](https://github.com/giulio-leone/gauss-py) |

## License

MIT
