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
    tool_name: str | None = None
    tool_result: Any = None


class Agent:
    """
    The core agent primitive — wraps a Rust agent loop with tools and middleware.

    Example:
        >>> agent = Agent(model=gauss("openai", "gpt-4o"), instructions="You are helpful.")
        >>> result = await agent.run("Hello!")
        >>> print(result.text)
    """

    def __init__(
        self,
        model: Provider,
        instructions: str = "",
        tools: Sequence[ToolDef | Callable] | None = None,
        max_steps: int = 10,
        name: str = "agent",
    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name

        # Normalize tools
        self._tools: list[ToolDef] = []
        if tools:
            for t in tools:
                if isinstance(t, ToolDef):
                    self._tools.append(t)
                elif callable(t) and hasattr(t, "_gauss_tool"):
                    self._tools.append(t._gauss_tool)  # type: ignore
                else:
                    raise TypeError(f"Invalid tool: {t}. Use @tool decorator or ToolDef.")

    async def run(self, prompt: str) -> AgentResult:
        """
        Run the agent with the given prompt.
        Tools are executed via callbacks from Rust.
        """
        from gauss._native import agent_run

        messages = []
        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})
        messages.append({"role": "user", "content": prompt})

        messages_json = json.dumps(messages)

        # Build tool definitions
        tools_json = json.dumps(
            [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters or {},
                }
                for t in self._tools
            ]
        )

        # Tool executor callback for Rust → Python
        tool_map = {t.name: t for t in self._tools}

        async def tool_executor(call_json: str) -> str:
            call = json.loads(call_json)
            tool_name = call.get("tool", call.get("name", ""))
            args = call.get("args", {})
            tool_def = tool_map.get(tool_name)
            if not tool_def or not tool_def.execute:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
            try:
                result = await tool_def.execute(**args)
                return json.dumps(result) if not isinstance(result, str) else result
            except Exception as e:
                return json.dumps({"error": str(e)})

        result = await agent_run(
            self.name,
            self.model.handle,
            messages_json,
            tools_json,
            max_steps=self.max_steps,
        )

        return AgentResult(
            text=result.get("text", ""),
            steps=result.get("steps", 0),
            usage={
                "input_tokens": result.get("inputTokens", 0),
                "output_tokens": result.get("outputTokens", 0),
            },
            structured_output=result.get("structuredOutput"),
        )

    async def stream(self, prompt: str) -> AsyncIterator[StreamChunk]:
        """
        Stream agent execution, yielding chunks as they arrive.

        Example:
            >>> async for chunk in agent.stream("Tell me a story"):
            ...     print(chunk.delta, end="")
        """
        from gauss._native import generate

        messages = []
        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})
        messages.append({"role": "user", "content": prompt})

        messages_json = json.dumps(messages)

        # For now, use non-streaming generate and yield result as single chunk
        # TODO: Wire to agent_stream when PyO3 async streaming is available
        result = await generate(self.model.handle, messages_json, None, None)

        yield StreamChunk(
            type="text_delta",
            delta=result.get("text", ""),
            text=result.get("text", ""),
        )
        yield StreamChunk(type="done", text=result.get("text", ""))
