"""
Tests for gauss Python SDK — comprehensive unit tests that don't require NAPI binary.
Tests the Python layer logic (tool inference, agent config, middleware, etc.)
"""

import inspect
import json
import math
import pytest

from tests.conftest import requires_native
from gauss.tools import tool, ToolDef, _infer_parameters


class TestToolDecorator:
    def test_basic_tool(self):
        @tool(description="Get weather")
        async def weather(city: str) -> str:
            return f"Sunny in {city}"

        assert hasattr(weather, "_gauss_tool")
        t = weather._gauss_tool
        assert isinstance(t, ToolDef)
        assert t.name == "weather"
        assert t.description == "Get weather"
        assert t.execute is weather
        assert t.parameters is not None
        assert "city" in t.parameters["properties"]

    def test_custom_name(self):
        @tool(description="Search", name="web_search")
        def search(query: str) -> str:
            return query

        assert search._gauss_tool.name == "web_search"

    def test_infer_parameters_types(self):
        def fn(name: str, count: int, score: float, active: bool) -> None:
            pass

        params = _infer_parameters(fn)
        assert params["properties"]["name"]["type"] == "string"
        assert params["properties"]["count"]["type"] == "integer"
        assert params["properties"]["score"]["type"] == "number"
        assert params["properties"]["active"]["type"] == "boolean"
        assert set(params["required"]) == {"name", "count", "score", "active"}

    def test_infer_optional_params(self):
        def fn(required: str, optional: str = "default") -> None:
            pass

        params = _infer_parameters(fn)
        assert "required" in params["required"]
        assert "optional" not in params["required"]

    def test_tool_with_no_params(self):
        @tool(description="No args")
        def noop() -> str:
            return "done"

        t = noop._gauss_tool
        assert t.parameters is not None
        assert len(t.parameters["properties"]) == 0
        assert "required" not in t.parameters or t.parameters["required"] == []

    def test_tool_with_explicit_parameters(self):
        custom_params = {
            "type": "object",
            "properties": {"x": {"type": "number"}},
            "required": ["x"],
        }
        @tool(description="Custom", parameters=custom_params)
        def custom(x: float) -> float:
            return x * 2

        assert custom._gauss_tool.parameters == custom_params


class TestToolDef:
    def test_dataclass(self):
        t = ToolDef(name="test", description="A test tool")
        assert t.name == "test"
        assert t.description == "A test tool"
        assert t.parameters is None
        assert t.execute is None

    def test_dataclass_with_all_fields(self):
        fn = lambda x: x
        params = {"type": "object", "properties": {}}
        t = ToolDef(name="full", description="Full tool", parameters=params, execute=fn)
        assert t.execute is fn
        assert t.parameters == params


class TestAgentConfig:
    def test_agent_init(self):
        from gauss.agent import Agent
        from gauss.provider import Provider

        provider = Provider(handle=1, provider_type="openai", model="gpt-4o")

        @tool(description="Test tool")
        async def my_tool(x: str) -> str:
            return x

        agent = Agent(
            model=provider,
            instructions="You are helpful",
            tools=[my_tool],
            max_steps=5,
            name="test-agent",
            temperature=0.7,
            max_tokens=1024,
        )

        assert agent.name == "test-agent"
        assert agent.instructions == "You are helpful"
        assert agent.max_steps == 5
        assert agent.temperature == 0.7
        assert agent.max_tokens == 1024
        assert len(agent._tools) == 1
        assert agent._tools[0].name == "my_tool"

    def test_agent_rejects_invalid_tools(self):
        from gauss.agent import Agent
        from gauss.provider import Provider

        provider = Provider(handle=1, provider_type="openai", model="gpt-4o")

        try:
            Agent(model=provider, tools=["not a tool"])
            assert False, "Should have raised TypeError"
        except TypeError:
            pass

    def test_agent_default_values(self):
        from gauss.agent import Agent
        from gauss.provider import Provider

        provider = Provider(handle=1, provider_type="openai", model="gpt-4o")
        agent = Agent(model=provider)

        assert agent.name == "agent"
        assert agent.instructions == ""
        assert agent.max_steps == 10
        assert agent.temperature is None
        assert agent.max_tokens is None
        assert agent.top_p is None
        assert agent.output_schema is None

    def test_agent_with_output_schema(self):
        from gauss.agent import Agent
        from gauss.provider import Provider

        provider = Provider(handle=1, provider_type="openai", model="gpt-4o")
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        agent = Agent(model=provider, output_schema=schema)

        assert agent.output_schema == schema

    def test_agent_with_multiple_tools(self):
        from gauss.agent import Agent
        from gauss.provider import Provider

        provider = Provider(handle=1, provider_type="openai", model="gpt-4o")

        @tool(description="Tool A")
        def tool_a(x: str) -> str:
            return x

        @tool(description="Tool B")
        def tool_b(y: int) -> int:
            return y

        agent = Agent(model=provider, tools=[tool_a, tool_b])
        assert len(agent._tools) == 2
        assert agent._tools[0].name == "tool_a"
        assert agent._tools[1].name == "tool_b"


class TestMiddlewareConfig:
    def test_middleware_config_dataclass(self):
        from gauss.middleware import MiddlewareConfig

        config = MiddlewareConfig(
            guardrail_handle=42,
            telemetry_handle=43,
        )
        assert config.guardrail_handle == 42
        assert config.telemetry_handle == 43

    def test_middleware_config_defaults(self):
        from gauss.middleware import MiddlewareConfig

        config = MiddlewareConfig()
        assert config.guardrail_handle is None
        assert config.telemetry_handle is None
        assert config.resilient_provider_handle is None

    @requires_native
    def test_middleware_factory_telemetry_only(self):
        from gauss.middleware import middleware

        config = middleware(telemetry=True)
        assert isinstance(config, MiddlewareConfig)


class TestGuardrailChainSignatures:
    def test_guardrail_chain_init(self):
        from gauss.guardrail import GuardrailChain

        assert hasattr(GuardrailChain, "add_content_moderation")
        assert hasattr(GuardrailChain, "add_pii_detection")
        assert hasattr(GuardrailChain, "add_token_limit")
        assert hasattr(GuardrailChain, "add_regex_filter")
        assert hasattr(GuardrailChain, "add_schema")
        assert hasattr(GuardrailChain, "list")

    @requires_native
    def test_guardrail_fluent_chaining(self):
        from gauss.guardrail import GuardrailChain

        chain = GuardrailChain()
        result = chain.add_content_moderation(block_patterns=["bad"])
        assert result is chain

        result = chain.add_pii_detection(action="redact")
        assert result is chain

        result = chain.add_token_limit(max_input=1000)
        assert result is chain

        result = chain.add_regex_filter(block_rules=[r"\d{3}-\d{2}-\d{4}"])
        assert result is chain

        result = chain.add_schema({"type": "object"})
        assert result is chain


class TestResilienceAPI:
    def test_exports_exist(self):
        from gauss.resilience import create_fallback, create_circuit_breaker, create_resilient

        assert callable(create_fallback)
        assert callable(create_circuit_breaker)
        assert callable(create_resilient)


class TestPluginAPI:
    def test_class_exists(self):
        from gauss.plugin import PluginRegistry

        assert hasattr(PluginRegistry, "add_telemetry")
        assert hasattr(PluginRegistry, "add_memory")
        assert hasattr(PluginRegistry, "list")
        assert hasattr(PluginRegistry, "emit")

    @requires_native
    def test_fluent_api(self):
        from gauss.plugin import PluginRegistry

        reg = PluginRegistry()
        result = reg.add_telemetry()
        assert result is reg

        result = reg.add_memory()
        assert result is reg


class TestContextAPI:
    def test_exports_exist(self):
        from gauss.context import count_tokens, count_tokens_for_model, count_message_tokens, get_context_window_size

        assert callable(count_tokens)
        assert callable(count_tokens_for_model)
        assert callable(count_message_tokens)
        assert callable(get_context_window_size)


class TestConfigAPI:
    def test_exports_exist(self):
        from gauss.config import config_from_json, resolve_env

        assert callable(config_from_json)
        assert callable(resolve_env)


class TestToolValidatorAPI:
    def test_class_exists(self):
        from gauss.tool_validator import ToolValidator

        assert hasattr(ToolValidator, "validate")

    @requires_native
    def test_init_default(self):
        from gauss.tool_validator import ToolValidator

        v = ToolValidator()
        assert v is not None

    @requires_native
    def test_init_with_strategies(self):
        from gauss.tool_validator import ToolValidator

        v = ToolValidator(strategies=["json_schema"])
        assert v is not None


class TestNetworkAPI:
    def test_class_exists(self):
        from gauss.network import Network

        assert hasattr(Network, "add_agent")
        assert hasattr(Network, "set_supervisor")
        assert hasattr(Network, "delegate")

    @requires_native
    def test_fluent_api(self):
        from gauss.network import Network

        net = Network()
        result = net.add_agent("agent-1", provider_handle=1)
        assert result is net

        result = net.set_supervisor("agent-1")
        assert result is net


class TestMcpServerAPI:
    def test_class_exists(self):
        from gauss.mcp import McpServer

        assert hasattr(McpServer, "add_tool")
        assert hasattr(McpServer, "handle")

    @requires_native
    def test_init(self):
        from gauss.mcp import McpServer

        server = McpServer(name="test-server", version="1.0.0")
        assert server is not None

    @requires_native
    def test_fluent_add_tool(self):
        from gauss.mcp import McpServer

        server = McpServer(name="test", version="0.1.0")
        result = server.add_tool({"name": "echo", "description": "Echo input"})
        assert result is server


class TestEvalAPI:
    def test_class_exists(self):
        from gauss.eval import EvalRunner

        assert hasattr(EvalRunner, "add_scorer")

    @requires_native
    def test_init_default(self):
        from gauss.eval import EvalRunner

        runner = EvalRunner()
        assert runner is not None

    @requires_native
    def test_init_with_threshold(self):
        from gauss.eval import EvalRunner

        runner = EvalRunner(threshold=0.8)
        assert runner is not None

    @requires_native
    def test_fluent_add_scorer(self):
        from gauss.eval import EvalRunner

        runner = EvalRunner()
        result = runner.add_scorer("accuracy")
        assert result is runner

    @requires_native
    def test_load_dataset_jsonl(self):
        from gauss.eval import load_dataset_jsonl

        data = '{"input": "hello", "expected": "hi"}\n{"input": "bye", "expected": "goodbye"}'
        result = load_dataset_jsonl(data)
        assert len(result) == 2
        assert result[0]["input"] == "hello"
        assert result[1]["expected"] == "goodbye"

    @requires_native
    def test_load_dataset_json(self):
        from gauss.eval import load_dataset_json

        data = json.dumps([{"input": "a"}, {"input": "b"}])
        result = load_dataset_json(data)
        assert len(result) == 2


class TestHitlAPI:
    def test_approval_manager_exists(self):
        from gauss.hitl import ApprovalManager

        assert hasattr(ApprovalManager, "request")
        assert hasattr(ApprovalManager, "approve")
        assert hasattr(ApprovalManager, "deny")
        assert hasattr(ApprovalManager, "list_pending")

    def test_checkpoint_store_exists(self):
        from gauss.hitl import CheckpointStore

        assert hasattr(CheckpointStore, "save")
        assert hasattr(CheckpointStore, "load")

    @requires_native
    def test_approval_manager_init(self):
        from gauss.hitl import ApprovalManager

        mgr = ApprovalManager()
        assert mgr is not None

    @requires_native
    def test_checkpoint_store_init(self):
        from gauss.hitl import CheckpointStore

        store = CheckpointStore()
        assert store is not None


class TestStreamAPI:
    def test_parse_partial_json(self):
        from gauss.stream import parse_partial_json

        assert callable(parse_partial_json)

    @requires_native
    def test_parse_partial_json_valid(self):
        from gauss.stream import parse_partial_json

        result = parse_partial_json('{"key": "value"}')
        assert result is not None

    @requires_native
    def test_parse_partial_json_partial(self):
        from gauss.stream import parse_partial_json

        result = parse_partial_json('{"key": "val')
        # Should handle partial JSON gracefully (may return None or partial)
        # Just verify it doesn't crash
        assert result is None or isinstance(result, str)


class TestVectorStoreAPI:
    def test_class_exists(self):
        from gauss.vector_store import VectorStore, cosine_similarity

        assert hasattr(VectorStore, "upsert")
        assert hasattr(VectorStore, "search")
        assert callable(cosine_similarity)

    @requires_native
    def test_cosine_similarity(self):
        from gauss.vector_store import cosine_similarity

        # Identical vectors → similarity = 1.0
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        result = cosine_similarity(a, b)
        assert abs(result - 1.0) < 1e-6

    @requires_native
    def test_cosine_similarity_orthogonal(self):
        from gauss.vector_store import cosine_similarity

        a = [1.0, 0.0]
        b = [0.0, 1.0]
        result = cosine_similarity(a, b)
        assert abs(result) < 1e-6

    @requires_native
    def test_cosine_similarity_opposite(self):
        from gauss.vector_store import cosine_similarity

        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        result = cosine_similarity(a, b)
        assert abs(result - (-1.0)) < 1e-6


class TestMemoryAPI:
    def test_class_exists(self):
        from gauss.memory import Memory

        assert hasattr(Memory, "store")
        assert hasattr(Memory, "recall")
        assert hasattr(Memory, "clear")

    @requires_native
    def test_init(self):
        from gauss.memory import Memory

        mem = Memory()
        assert mem is not None


class TestTelemetryAPI:
    def test_class_exists(self):
        from gauss.telemetry import Telemetry

        assert hasattr(Telemetry, "record_span")
        assert hasattr(Telemetry, "export_spans")
        assert hasattr(Telemetry, "export_metrics")
        assert hasattr(Telemetry, "clear")

    @requires_native
    def test_init(self):
        from gauss.telemetry import Telemetry

        t = Telemetry()
        assert t is not None


class TestProviderAPI:
    def test_provider_dataclass(self):
        from gauss.provider import Provider

        p = Provider(handle=1, provider_type="openai", model="gpt-4o")
        assert p.handle == 1
        assert p.provider_type == "openai"
        assert p.model == "gpt-4o"

    def test_gauss_factory_exists(self):
        from gauss.provider import gauss

        assert callable(gauss)


class TestAllExports:
    def test_all_exports_importable(self):
        import gauss

        expected = [
            "Agent", "AgentResult", "StreamChunk", "gauss", "Provider",
            "tool", "ToolDef", "middleware", "MiddlewareConfig",
            "Memory", "VectorStore", "cosine_similarity",
            "GuardrailChain", "Telemetry",
            "create_fallback", "create_circuit_breaker", "create_resilient",
            "PluginRegistry", "config_from_json", "resolve_env",
            "ToolValidator", "Network", "McpServer",
            "EvalRunner", "load_dataset_jsonl", "load_dataset_json",
            "ApprovalManager", "CheckpointStore",
            "count_tokens", "count_tokens_for_model", "count_message_tokens",
            "get_context_window_size", "parse_partial_json",
        ]

        for name in expected:
            assert hasattr(gauss, name), f"Missing export: {name}"

    def test_version(self):
        import gauss
        assert gauss.__version__ == "1.0.0"

    def test_all_list_complete(self):
        import gauss
        for name in gauss.__all__:
            assert hasattr(gauss, name), f"__all__ contains '{name}' but it's not importable"
