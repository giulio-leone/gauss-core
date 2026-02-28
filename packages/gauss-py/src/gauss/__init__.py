"""
Gauss AI — A Rust-powered agentic AI framework for Python.

Usage:
    from gauss import Agent, gauss

    agent = Agent(model=gauss("openai", "gpt-4o"), instructions="You are helpful.")
    result = await agent.run("Hello!")
"""

from gauss.provider import gauss, Provider
from gauss.agent import Agent, AgentResult, StreamChunk
from gauss.tools import tool, ToolDef
from gauss.middleware import middleware, MiddlewareConfig
from gauss.memory import Memory
from gauss.vector_store import VectorStore, cosine_similarity
from gauss.telemetry import Telemetry
from gauss.guardrail import GuardrailChain
from gauss.resilience import create_fallback, create_circuit_breaker, create_resilient
from gauss.plugin import PluginRegistry
from gauss.config import config_from_json, resolve_env
from gauss.tool_validator import ToolValidator
from gauss.network import Network
from gauss.mcp import McpServer
from gauss.eval import EvalRunner, load_dataset_jsonl, load_dataset_json
from gauss.hitl import ApprovalManager, CheckpointStore
from gauss.context import count_tokens, count_tokens_for_model, count_message_tokens, get_context_window_size
from gauss.stream import parse_partial_json

__version__ = "1.0.0"

__all__ = [
    # Core
    "Agent",
    "AgentResult",
    "StreamChunk",
    "gauss",
    "Provider",
    # Tools
    "tool",
    "ToolDef",
    # Middleware
    "middleware",
    "MiddlewareConfig",
    # Memory & RAG
    "Memory",
    "VectorStore",
    "cosine_similarity",
    # Safety
    "GuardrailChain",
    # Telemetry
    "Telemetry",
    # Resilience
    "create_fallback",
    "create_circuit_breaker",
    "create_resilient",
    # Plugin
    "PluginRegistry",
    # Config
    "config_from_json",
    "resolve_env",
    # Validation
    "ToolValidator",
    # Multi-agent
    "Network",
    # MCP
    "McpServer",
    # Evaluation
    "EvalRunner",
    "load_dataset_jsonl",
    "load_dataset_json",
    # HITL
    "ApprovalManager",
    "CheckpointStore",
    # Context
    "count_tokens",
    "count_tokens_for_model",
    "count_message_tokens",
    "get_context_window_size",
    # Stream
    "parse_partial_json",
]
