import pytest
from unittest.mock import MagicMock, AsyncMock
from pydantic_graph import GraphRunContext

from solutions.models import (
    QueryIntent,
    IntentClassification,
    MissionCriticality,
    LatencyCriticality,
    RoutingDecision,
    MetaRoutingDecision,
)
from solutions.pipelines import (
    MetaRoutingNode,
    IntentNode,
    ParallelScoringNode,
    FastDecisionNode,
    DecisionNode,
    GraphState,
    GraphDeps,
    AgentResult,
)
from solutions.utils import DEFAULT_MODEL_ASSIGNMENTS
from src.model_registry import MODEL_REGISTRY, ModelTier, EDGE_MODELS


ALL_MODEL_KEYS = list(MODEL_REGISTRY.keys())


class _FakeUsage:
    def __init__(self, req: int = 10, resp: int = 20):
        self.request_tokens = req
        self.response_tokens = resp


def _stub_result(output, model_key: str = "gemma-3n-e4b") -> AgentResult:
    return AgentResult(output=output, usage=_FakeUsage(), model_key=model_key)


def _make_intent(intent: QueryIntent, confidence: float = 0.95) -> IntentClassification:
    return IntentClassification(intent=intent, confidence=confidence, reasoning="test")


def _make_state(
    *,
    query: str = "test query",
    available_models=None,
    model_assignments=None,
    intent=None,
    mission=None,
    latency=None,
) -> GraphState:
    return GraphState(
        query=query,
        available_models=available_models if available_models is not None else ALL_MODEL_KEYS,
        model_assignments=model_assignments if model_assignments is not None else {},
        intent=intent,
        mission=mission,
        latency=latency,
    )


def _make_context(state: GraphState, deps: GraphDeps) -> GraphRunContext:
    ctx = MagicMock(spec=GraphRunContext)
    ctx.state = state
    ctx.deps = deps
    return ctx


def _default_deps(**overrides) -> GraphDeps:
    kwargs = {"model_assignments": dict(DEFAULT_MODEL_ASSIGNMENTS)}
    kwargs.update(overrides)
    return GraphDeps(**kwargs)


@pytest.fixture
def mock_agent_factories(monkeypatch):
    """Patch all agent factories in solutions.pipelines to return distinct MagicMock agents."""
    factories = {
        "meta": MagicMock(return_value=MagicMock(name="meta_agent")),
        "intent": MagicMock(return_value=MagicMock(name="intent_agent")),
        "mission": MagicMock(return_value=MagicMock(name="mission_agent")),
        "latency": MagicMock(return_value=MagicMock(name="latency_agent")),
        "decision": MagicMock(return_value=MagicMock(name="decision_agent")),
    }
    monkeypatch.setattr("solutions.pipelines.create_meta_routing_agent", factories["meta"])
    monkeypatch.setattr("solutions.pipelines.create_intention_agent", factories["intent"])
    monkeypatch.setattr("solutions.pipelines.create_mission_criticality_agent", factories["mission"])
    monkeypatch.setattr("solutions.pipelines.create_latency_criticality_agent", factories["latency"])
    monkeypatch.setattr("solutions.pipelines.creat_decision_making_agent", factories["decision"])
    return factories


class TestMetaRoutingNode:
    @pytest.mark.asyncio
    async def test_success_applies_llm_assignments(self, mock_agent_factories, monkeypatch):
        state = _make_state()
        deps = _default_deps()
        decision = MetaRoutingDecision(
            intent_model="gemma-3-4b",
            mission_model="llama-3.2-3b",
            latency_model="gemma-3n-e4b",
            decision_model="mistral-small-24b",
            reasoning="test",
        )
        monkeypatch.setattr(
            "solutions.pipelines.run_agent",
            AsyncMock(return_value=_stub_result(decision)),
        )

        await MetaRoutingNode().run(_make_context(state, deps))

        assert state.model_assignments["intent"] == "gemma-3-4b"
        assert state.model_assignments["mission"] == "llama-3.2-3b"
        assert state.model_assignments["latency"] == "gemma-3n-e4b"
        assert state.model_assignments["decision"] == "mistral-small-24b"
        assert "meta_routing" in state.step_cost
        assert "meta_routing" not in state.errors

    @pytest.mark.asyncio
    async def test_invalid_model_falls_per_role(self, mock_agent_factories, monkeypatch):
        """LLM returning a model not in available_models falls back per-role to deps."""
        state = _make_state()
        deps = _default_deps()
        decision = MetaRoutingDecision(
            intent_model="nonexistent-model",
            mission_model="llama-3.2-3b",
            latency_model="gemma-3n-e4b",
            decision_model="mistral-small-24b",
            reasoning="test",
        )
        monkeypatch.setattr(
            "solutions.pipelines.run_agent",
            AsyncMock(return_value=_stub_result(decision)),
        )

        await MetaRoutingNode().run(_make_context(state, deps))

        assert state.model_assignments["intent"] == deps.model_assignments["intent"]
        assert state.model_assignments["mission"] == "llama-3.2-3b"
        assert state.model_assignments["decision"] == "mistral-small-24b"

    @pytest.mark.asyncio
    async def test_run_agent_raises_falls_to_deps(self, mock_agent_factories, monkeypatch):
        state = _make_state()
        deps = _default_deps()
        monkeypatch.setattr(
            "solutions.pipelines.run_agent",
            AsyncMock(side_effect=RuntimeError("boom")),
        )

        await MetaRoutingNode().run(_make_context(state, deps))

        assert state.model_assignments == deps.model_assignments
        assert "meta_routing" in state.errors

    @pytest.mark.asyncio
    async def test_meta_routing_disabled_uses_deps(self, mock_agent_factories):
        state = _make_state()
        deps = _default_deps(meta_routing=False)

        await MetaRoutingNode().run(_make_context(state, deps))

        assert state.model_assignments == deps.model_assignments
        assert "meta_routing" not in state.step_cost


class TestIntentNodeRun:
    """IntentNode.run — async orchestration."""
    _ASSIGNMENTS = {"intent": "gemma-3n-e4b"}
    def test_simple_factual_high_conf_triggers(self):
        intent = _make_intent(QueryIntent.SIMPLE_FACTUAL, confidence=0.95)
        assert IntentNode._can_fast_route(intent, confidence_threshold=0.9) is True

    def test_below_threshold_does_not_trigger(self):
        intent = _make_intent(QueryIntent.SIMPLE_FACTUAL, confidence=0.7)
        assert IntentNode._can_fast_route(intent, confidence_threshold=0.9) is False

    @pytest.mark.parametrize("intent_type", [
        QueryIntent.EXPLANATION,
        QueryIntent.ANALYSIS,
        QueryIntent.REASONING,
        QueryIntent.CODING,
    ])
    def test_rejects_non_factual(self, intent_type):
        """All intents except SIMPLE_FACTUAL must go through full pipeline."""
        intent = _make_intent(intent_type, confidence=0.99)
        assert IntentNode._can_fast_route(intent, confidence_threshold=0.9) is False

    @pytest.mark.asyncio
    async def test_non_simple_routes_to_parallel(self, mock_agent_factories, monkeypatch):
        state = _make_state(model_assignments=dict(self._ASSIGNMENTS))
        deps = _default_deps()
        intent_out = IntentClassification(intent=QueryIntent.CODING, confidence=0.9, reasoning="t")
        monkeypatch.setattr(
            "solutions.pipelines.run_agent",
            AsyncMock(return_value=_stub_result(intent_out)),
        )

        next_node = await IntentNode().run(_make_context(state, deps))

        assert isinstance(next_node, ParallelScoringNode)
        assert state.intent.intent == QueryIntent.CODING
        assert mock_agent_factories["intent"].call_args.kwargs.get("model_key") == "gemma-3n-e4b"

    @pytest.mark.asyncio
    async def test_simple_high_conf_routes_to_fast(self, mock_agent_factories, monkeypatch):
        state = _make_state(model_assignments=dict(self._ASSIGNMENTS))
        deps = _default_deps()
        intent_out = IntentClassification(
            intent=QueryIntent.SIMPLE_FACTUAL, confidence=0.95, reasoning="t"
        )
        monkeypatch.setattr(
            "solutions.pipelines.run_agent",
            AsyncMock(return_value=_stub_result(intent_out)),
        )

        next_node = await IntentNode().run(_make_context(state, deps))

        assert isinstance(next_node, FastDecisionNode)

    @pytest.mark.asyncio
    async def test_simple_low_conf_routes_to_parallel(self, mock_agent_factories, monkeypatch):
        """Boundary: SIMPLE_FACTUAL below threshold must not shortcut."""
        state = _make_state(model_assignments=dict(self._ASSIGNMENTS))
        deps = _default_deps()
        intent_out = IntentClassification(
            intent=QueryIntent.SIMPLE_FACTUAL, confidence=0.5, reasoning="t"
        )
        monkeypatch.setattr(
            "solutions.pipelines.run_agent",
            AsyncMock(return_value=_stub_result(intent_out)),
        )

        next_node = await IntentNode().run(_make_context(state, deps))

        assert isinstance(next_node, ParallelScoringNode)

    @pytest.mark.asyncio
    async def test_run_agent_raises_uses_fallback(self, mock_agent_factories, monkeypatch):
        state = _make_state(model_assignments=dict(self._ASSIGNMENTS))
        deps = _default_deps()
        monkeypatch.setattr(
            "solutions.pipelines.run_agent",
            AsyncMock(side_effect=RuntimeError("boom")),
        )

        next_node = await IntentNode().run(_make_context(state, deps))

        assert state.intent.intent == QueryIntent.EXPLANATION
        assert state.intent.confidence == 0.0
        assert isinstance(next_node, ParallelScoringNode)
        assert "intent" in state.errors

    def test_fallback_intent_value_pinned(self):
        """Pin the fallback intent's string value — leaks into logs/telemetry."""
        assert QueryIntent.EXPLANATION.value == "explanation"


class TestParallelScoringNode:
    _ASSIGNMENTS = {"mission": "gemma-3n-e4b", "latency": "gemma-3n-e4b"}

    def _state(self):
        return _make_state(
            model_assignments=dict(self._ASSIGNMENTS),
            intent=IntentClassification(
                intent=QueryIntent.EXPLANATION, confidence=0.8, reasoning="t"
            ),
        )

    @pytest.mark.asyncio
    async def test_both_succeed(self, mock_agent_factories, monkeypatch):
        state = self._state()
        deps = _default_deps()
        mission_agent = mock_agent_factories["mission"].return_value
        latency_agent = mock_agent_factories["latency"].return_value
        mission_out = MissionCriticality(score=0.7, confidence=0.9, reasoning="m")
        latency_out = LatencyCriticality(score=0.2, confidence=0.8, reasoning="l")

        async def fake_run_agent(agent, prompt, **_):
            if agent is mission_agent:
                return _stub_result(mission_out)
            if agent is latency_agent:
                return _stub_result(latency_out)
            raise AssertionError("unexpected agent")

        monkeypatch.setattr("solutions.pipelines.run_agent", fake_run_agent)

        await ParallelScoringNode().run(_make_context(state, deps))

        assert state.mission.score == 0.7
        assert state.latency.score == 0.2
        assert "mission" in state.step_cost
        assert "latency" in state.step_cost
        assert "mission" not in state.errors
        assert "latency" not in state.errors

    @pytest.mark.asyncio
    async def test_mission_fails_latency_succeeds(self, mock_agent_factories, monkeypatch):
        """Covers the symmetric latency-fails case too: failure handling is identical."""
        state = self._state()
        deps = _default_deps()
        mission_agent = mock_agent_factories["mission"].return_value
        latency_agent = mock_agent_factories["latency"].return_value
        latency_out = LatencyCriticality(score=0.3, confidence=0.7, reasoning="l")

        async def fake_run_agent(agent, prompt, **_):
            if agent is mission_agent:
                raise RuntimeError("mission boom")
            if agent is latency_agent:
                return _stub_result(latency_out)
            raise AssertionError("unexpected agent")

        monkeypatch.setattr("solutions.pipelines.run_agent", fake_run_agent)

        await ParallelScoringNode().run(_make_context(state, deps))

        assert state.mission.score == 0.5
        assert state.mission.confidence == 0.0
        assert "mission" in state.errors
        assert state.latency.score == 0.3
        assert "latency" not in state.errors

    @pytest.mark.asyncio
    async def test_both_fail_use_fallbacks(self, mock_agent_factories, monkeypatch):
        state = self._state()
        deps = _default_deps()

        async def fake_run_agent(agent, prompt, **_):
            raise RuntimeError("boom")

        monkeypatch.setattr("solutions.pipelines.run_agent", fake_run_agent)

        await ParallelScoringNode().run(_make_context(state, deps))

        assert state.mission.score == 0.5
        assert state.mission.confidence == 0.0
        assert state.latency.score == 0.5
        assert state.latency.confidence == 0.0
        assert "mission" in state.errors
        assert "latency" in state.errors


class TestFastDecisionNode:
    def _state(self):
        return _make_state(intent=_make_intent(QueryIntent.SIMPLE_FACTUAL))

    @pytest.mark.asyncio
    async def test_picks_small_edge(self):
        state = self._state()
        deps = GraphDeps()

        await FastDecisionNode().run(_make_context(state, deps))

        decision = state.decision
        assert decision is not None
        assert decision.deployment == "edge"
        assert MODEL_REGISTRY[decision.model_key].tier == ModelTier.SMALL

    @pytest.mark.asyncio
    async def test_records_step_latency(self):
        state = self._state()
        deps = GraphDeps()

        await FastDecisionNode().run(_make_context(state, deps))

        assert "fast_decision" in state.step_latency_ms


class TestDecisionNodeRun:
    """DecisionNode.run — async orchestration."""

    def _state(self):
        return _make_state(
            model_assignments={"decision": "gemma-3n-e4b"},
            intent=IntentClassification(
                intent=QueryIntent.REASONING, confidence=0.8, reasoning="t"
            ),
            mission=MissionCriticality(score=0.7, confidence=0.8, reasoning="m"),
            latency=LatencyCriticality(score=0.3, confidence=0.7, reasoning="l"),
        )

    @pytest.mark.asyncio
    async def test_success_passes_decision_through(self, mock_agent_factories, monkeypatch):
        state = self._state()
        deps = _default_deps()
        decision_out = RoutingDecision(
            model_key="llama-3.3-70b",
            deployment="cloud",
            confidence=0.85,
            reasoning="r",
        )
        monkeypatch.setattr(
            "solutions.pipelines.run_agent",
            AsyncMock(return_value=_stub_result(decision_out)),
        )

        await DecisionNode().run(_make_context(state, deps))

        assert state.decision is decision_out
        assert "decision" in state.step_cost
        assert "decision" not in state.errors
        assert mock_agent_factories["decision"].call_args.kwargs.get("model_key") == "gemma-3n-e4b"

    @pytest.mark.asyncio
    async def test_run_agent_raises_uses_fallback(self, mock_agent_factories, monkeypatch):
        state = self._state()
        deps = _default_deps()
        monkeypatch.setattr(
            "solutions.pipelines.run_agent",
            AsyncMock(side_effect=RuntimeError("boom")),
        )

        await DecisionNode().run(_make_context(state, deps))

        assert state.decision is not None
        assert state.decision.model_key in MODEL_REGISTRY
        assert "decision" in state.errors


class TestDecisionNodeFallback:
    """DecisionNode._fall_back — pure static router."""

    def _state(self, intent, mission_score=0.5, latency_score=0.5):
        return _make_state(
            intent=_make_intent(intent),
            mission=MissionCriticality(score=mission_score, confidence=0.8, reasoning="m"),
            latency=LatencyCriticality(score=latency_score, confidence=0.8, reasoning="l"),
        )

    @pytest.mark.parametrize("intent_type", list(QueryIntent))
    def test_returns_valid_decision_for_all_intents(self, intent_type):
        """Schema canary: fallback always returns a structurally valid decision."""
        state = self._state(intent_type)
        decision = DecisionNode._fall_back(state)

        assert isinstance(decision, RoutingDecision)
        assert decision.model_key in MODEL_REGISTRY
        assert decision.deployment in ("edge", "cloud")

    @pytest.mark.parametrize("intent_type", [
        QueryIntent.CODING,
        QueryIntent.REASONING,
        QueryIntent.ANALYSIS,
    ])
    def test_escalates_for_hard_intents(self, intent_type):
        state = self._state(intent_type, mission_score=0.3, latency_score=0.3)
        decision = DecisionNode._fall_back(state)
        tier = MODEL_REGISTRY[decision.model_key].tier
        assert tier in (ModelTier.REASONING, ModelTier.LARGE, ModelTier.MEDIUM)

    def test_high_mission_escalates(self):
        state = self._state(QueryIntent.EXPLANATION, mission_score=0.8, latency_score=0.3)
        decision = DecisionNode._fall_back(state)
        tier = MODEL_REGISTRY[decision.model_key].tier
        assert tier in (ModelTier.LARGE, ModelTier.REASONING, ModelTier.MEDIUM)

    def test_high_latency_prefers_edge(self):
        state = self._state(QueryIntent.EXPLANATION, mission_score=0.3, latency_score=0.8)
        decision = DecisionNode._fall_back(state)
        tier = MODEL_REGISTRY[decision.model_key].tier
        assert tier in (ModelTier.SMALL, ModelTier.MEDIUM)
