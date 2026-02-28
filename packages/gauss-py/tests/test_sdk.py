"""
Tests for gauss Python SDK — unit tests that don't require NAPI binary.
Tests the Python layer logic (tool inference, agent config, etc.)
"""

import inspect
import json

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


class TestToolDef:
    def test_dataclass(self):
        t = ToolDef(name="test", description="A test tool")
        assert t.name == "test"
        assert t.description == "A test tool"
        assert t.parameters is None
        assert t.execute is None


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


class TestMiddlewareConfig:
    def test_middleware_config_dataclass(self):
        from gauss.middleware import MiddlewareConfig

        config = MiddlewareConfig(
            guardrail_handle=42,
            telemetry_handle=43,
        )
        assert config.guardrail_handle == 42
        assert config.telemetry_handle == 43


class TestGuardrailChainSignatures:
    """Verify guardrail methods accept the correct Rust-aligned signatures."""

    def test_guardrail_chain_init(self):
        from gauss.guardrail import GuardrailChain

        # Just verify the class exists and has the right methods
        assert hasattr(GuardrailChain, "add_content_moderation")
        assert hasattr(GuardrailChain, "add_pii_detection")
        assert hasattr(GuardrailChain, "add_token_limit")
        assert hasattr(GuardrailChain, "add_regex_filter")
        assert hasattr(GuardrailChain, "add_schema")
        assert hasattr(GuardrailChain, "list")


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


class TestContextAPI:
    def test_exports_exist(self):
        from gauss.context import count_tokens, count_tokens_for_model, count_message_tokens, get_context_window_size

        assert callable(count_tokens)
        assert callable(count_tokens_for_model)
        assert callable(count_message_tokens)
        assert callable(get_context_window_size)


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
