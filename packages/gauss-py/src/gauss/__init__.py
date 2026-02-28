"""
Gauss AI — A Rust-powered agentic AI framework for Python.

Usage:
    from gauss import Agent, gauss

    agent = Agent(model=gauss("openai", "gpt-4o"), instructions="You are helpful.")
    result = await agent.run("Hello!")
"""

from gauss.provider import gauss, Provider
from gauss.agent import Agent, AgentResult, StreamChunk
from gauss.tools import tool
from gauss.middleware import middleware, MiddlewareConfig
from gauss.memory import Memory
from gauss.vector_store import VectorStore
from gauss.telemetry import Telemetry
from gauss.guardrail import GuardrailChain

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentResult",
    "StreamChunk",
    "gauss",
    "Provider",
    "tool",
    "middleware",
    "MiddlewareConfig",
    "Memory",
    "VectorStore",
    "Telemetry",
    "GuardrailChain",
]
