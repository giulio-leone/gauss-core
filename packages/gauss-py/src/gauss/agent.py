"""
Agent — The core primitive for agentic AI execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional, Sequence

from gauss.provider import Provider
from gauss.tools import ToolDef


@dataclass
class AgentResult:
    """Result of an agent run."""

    text: str
    steps: int
    usage: dict[str, int]
    structured_output: Any = None


@dataclass
class StreamChunk:
    """A single chunk from a streaming agent execution."""

    type: str
    text: str = ""
    delta: str = ""
    step: int = 0
    tool_name: Optional[str] = None
    tool_result: Any = None


class Agent:
    """
    The core agent primitive — wraps a Rust agent loop.

    Example:
        >>> agent = Agent(model=gauss("openai", "gpt-4o"), instructions="You are helpful.")
        >>> result = await agent.run("Hello!")
        >>> print(result.text)
    """

    def __init__(
        self,
        model: Provider,
        instructions: str = "",
        tools: Optional[Sequence[Any]] = None,
        max_steps: int = 10,
        name: str = "agent",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        output_schema: Optional[dict[str, Any]] = None,
    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.output_schema = output_schema

        self._tools: list[ToolDef] = []
        if tools:
            for t in tools:
                if isinstance(t, ToolDef):
                    self._tools.append(t)
                elif callable(t) and hasattr(t, "_gauss_tool"):
                    self._tools.append(t._gauss_tool)
                else:
                    raise TypeError(f"Invalid tool: {t}. Use @tool decorator or ToolDef.")

    async def run(self, prompt: str) -> AgentResult:
        """Run the agent. Matches Rust: agent_run(name, handle, messages_json, options_json)."""
        from gauss._native import agent_run

        messages = []
        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})
        messages.append({"role": "user", "content": prompt})

        options: dict[str, Any] = {"max_steps": self.max_steps}
        if self.instructions:
            options["instructions"] = self.instructions
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.max_tokens is not None:
            options["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.output_schema is not None:
            options["output_schema"] = self.output_schema

        result_json = await agent_run(
            self.name,
            self.model.handle,
            json.dumps(messages),
            json.dumps(options),
        )
        result = json.loads(result_json) if isinstance(result_json, str) else result_json

        return AgentResult(
            text=result.get("text", ""),
            steps=result.get("steps", 0),
            usage={
                "input_tokens": result.get("usage", {}).get("input_tokens", 0),
                "output_tokens": result.get("usage", {}).get("output_tokens", 0),
            },
            structured_output=result.get("structured_output"),
        )

    async def stream(self, prompt: str) -> AsyncIterator[StreamChunk]:
        """Stream agent execution via Rust provider streaming."""
        from gauss._native import stream_generate

        messages = []
        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})
        messages.append({"role": "user", "content": prompt})

        events_json = await stream_generate(
            self.model.handle,
            json.dumps(messages),
            self.temperature,
            self.max_tokens,
        )
        events = json.loads(events_json) if isinstance(events_json, str) else events_json

        text_acc = ""
        for event in events:
            evt = event if isinstance(event, dict) else json.loads(event)
            evt_type = evt.get("type", "")

            if evt_type == "text_delta":
                delta = evt.get("TextDelta", evt.get("text_delta", ""))
                if isinstance(delta, str):
                    text_acc += delta
                    yield StreamChunk(type="text_delta", delta=delta, text=text_acc)
            elif evt_type == "done":
                yield StreamChunk(type="done", text=text_acc)
            elif evt_type == "finish_reason":
                continue
            elif evt_type == "usage":
                continue
            else:
                yield StreamChunk(type=evt_type, text=text_acc)
