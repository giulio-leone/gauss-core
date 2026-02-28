from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from .provider import Provider

@dataclass
class AgentResult:
    text: str
    steps: int
    usage: dict[str, int]
    structured_output: Any = None

@dataclass
class StreamChunk:
    type: str
    text: str = ""
    delta: str = ""
    step: int = 0
    tool_name: str | None = None
    tool_result: Any = None

class Agent:
    model: Provider
    instructions: str
    max_steps: int
    name: str
    temperature: float | None
    max_tokens: int | None
    top_p: float | None
    output_schema: dict[str, Any] | None

    def __init__(
        self,
        model: Provider,
        instructions: str = "",
        tools: Sequence[Any] | None = None,
        max_steps: int = 10,
        name: str = "agent",
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> None: ...
    async def run(self, prompt: str) -> AgentResult: ...
    async def stream(self, prompt: str) -> AsyncIterator[StreamChunk]: ...
