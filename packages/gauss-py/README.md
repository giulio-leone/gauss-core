# Gauss AI — Python SDK

A Rust-powered agentic AI framework for Python. Zero external dependencies — the entire LLM engine runs natively in Rust via PyO3.

## Quickstart

```python
from gauss import Agent, gauss

# 3 lines to start
agent = Agent(model=gauss("openai", "gpt-4o"), instructions="You are helpful.")
result = await agent.run("What's the weather in Rome?")
print(result.text)
```

## Features

- **Native Rust core** — All LLM operations run in Rust (100x faster than pure Python)
- **Agent primitive** — `Agent()` with tools, structured output, multi-step loops
- **Streaming** — `async for chunk in agent.stream("...")`
- **Decorators** — Memory, telemetry, guardrails, resilience
- **Multi-agent** — Agent networks, subagent-as-tool
- **6 providers** — OpenAI, Anthropic, Google, Groq, Ollama, DeepSeek

## Install

```bash
pip install gauss-ai
```

## API

### Agent

```python
from gauss import Agent, gauss, tool

@tool(description="Get weather for a city")
async def weather(city: str) -> str:
    return f"Sunny in {city}"

agent = Agent(
    model=gauss("openai", "gpt-4o"),
    instructions="You are a weather assistant.",
    tools=[weather],
    max_steps=5,
)

# Run
result = await agent.run("Weather in Rome?")

# Stream
async for chunk in agent.stream("Weather in Milan?"):
    print(chunk.text, end="")
```

### Middleware

```python
from gauss import Agent, gauss, middleware

agent = Agent(
    model=gauss("openai", "gpt-4o"),
    instructions="...",
    middleware=middleware(
        logging=True,
        caching={"ttl_ms": 60_000},
        guardrail={"content_moderation": {"threshold": 0.8}},
    ),
)
```

### Memory & RAG

```python
from gauss import Memory, VectorStore

memory = Memory()
memory.store("user_123", "User prefers Italian food")
results = memory.recall("user_123", "food preferences")

store = VectorStore()
store.upsert("doc1", [0.1, 0.2, 0.3], {"text": "Hello"})
results = store.search([0.1, 0.2, 0.3], top_k=5)
```

## License

MIT
