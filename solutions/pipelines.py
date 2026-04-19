import asyncio
import logging
import random
import timeit
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal, Tuple

from pydantic import ValidationError
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_graph import GraphRunContext, BaseNode, End, Graph

logger = logging.getLogger(__name__)

from src.model_registry import MODEL_REGISTRY, EDGE_MODELS, ModelTier

from .agents import create_meta_routing_agent, create_intention_agent, create_mission_criticality_agent, \
    create_latency_criticality_agent, creat_decision_making_agent
from .models import IntentClassification, MissionCriticality, LatencyCriticality, RoutingDecision, QueryIntent
from .utils import AGENT_ROLES, _parse_json_output


@dataclass
class AgentResult:
    """
    Lightweight wrapper for run agent.
    """
    output: object
    usage: object
    model_key: str = ""


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Check if an exception is an HTTP 429 rate limit error."""
    return isinstance(exc, ModelHTTPError) and exc.status_code == 429


async def run_agent(
    agent,
    user_prompt: str,
    max_http_retries: int = 3,
    base_delay: float = 3.0,
    max_delay: float = 15.0,
):
    """
    Run an agent
    - Handling models without system prompt use support.
    - Retries on HTTP 429 (rate limit) with exponential backoff + jitter.
    - For models without output_type use: parses raw text into the expected pydantic model.
      On JSON parse failure, feeds back the error so the model can self-correct.
    """
    if getattr(agent, '_prepend_prompt', None):
        user_prompt = f"{agent._prepend_prompt}\n\n{user_prompt}"

    model_key = getattr(agent, '_model_key', "")
    output_type = getattr(agent, '_output_type', None)
    retries = getattr(agent, '_parse_retries', 1)

    last_429_error = None
    for http_attempt in range(max_http_retries):
        try:
            last_error = None
            prompt = user_prompt
            for attempt in range(retries):
                result = await agent.run(user_prompt=prompt)

                if output_type is not None and isinstance(result.output, str):
                    # llm does not support output_type. parse manually.
                    try:
                        parsed = _parse_json_output(result.output, output_type)
                        return AgentResult(output=parsed, usage=result.usage(), model_key=model_key)
                    except (ValueError, ValidationError) as e:
                        last_error = e
                        logger.warning("JSON parse attempt %d/%d failed for %s: %s", attempt + 1, retries, model_key, e)
                        prompt = f"""
                            {user_prompt}

                            Your previous response was not valid JSON.
                            f"Your output: {result.output[:500]}
                            Error: {e}

                            Please respond ONLY with valid JSON matching the required schema.
                            Do not include explanation, markdown fences, or any text outside the JSON object.
                            """
                        continue
                return AgentResult(output=result.output, usage=result.usage(), model_key=model_key)

            raise last_error

        except Exception as e:
            if _is_rate_limit_error(e):
                last_429_error = e
                delay = min(base_delay * (2 ** http_attempt), max_delay) + random.uniform(0, base_delay)
                logger.warning(
                    "Rate limited (429) on %s, attempt %d/%d, waiting %.1fs",
                    model_key, http_attempt + 1, max_http_retries, delay,
                )
                await asyncio.sleep(delay)
                continue
            raise

    raise last_429_error


def _pick_model(
    available: List[str],
    tier_priority: List[ModelTier],
    prefer_edge: bool = False,
) -> Tuple[str, Literal["edge", "cloud"]]:
    """
    Select the cheapest model following tier priority, with optional edge preference.
    """
    for tier in tier_priority:
        candidates = [m for m in available if MODEL_REGISTRY[m].tier == tier]
        if not candidates:
            continue
        if prefer_edge:
            edge = [m for m in candidates if m in EDGE_MODELS]
            if edge:
                return min(edge, key=lambda m: MODEL_REGISTRY[m].cost_per_million_input), "edge"
        return min(candidates, key=lambda m: MODEL_REGISTRY[m].cost_per_million_input), "cloud"
    # ultimate fallback: cheapest available model
    model_key = min(available, key=lambda m: MODEL_REGISTRY[m].cost_per_million_input)
    deployment = "edge" if model_key in EDGE_MODELS else "cloud"
    return model_key, deployment


def _compute_cost(usage, model_key: str) -> float:
    """
    Compute US dollar cost from a pydantic-ai RunUsage and the model registry.
    """
    model_config = MODEL_REGISTRY.get(model_key)
    if model_config is None or usage is None:
        return 0.0
    input_tokens = getattr(usage, "request_tokens", 0) or 0
    output_tokens = getattr(usage, "response_tokens", 0) or 0
    return (input_tokens * model_config.cost_per_million_input + output_tokens * model_config.cost_per_million_output) / 1_000_000


_AGENT_KEYS = ("meta", *AGENT_ROLES)


@dataclass
class GraphState:
    query:str
    available_models: Optional[List[str]] = None
    model_assignments: Optional[Dict[str, str]] = field(default_factory=dict)
    intent: IntentClassification | None = None
    mission: MissionCriticality | None = None
    latency: LatencyCriticality | None = None
    decision: RoutingDecision | None = None
    routing_latency_ms: float = 0.0
    step_latency_ms: Dict[str, float] = field(default_factory=dict)
    reasoning: Dict[str, str] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    cost: float = 0.0
    step_cost: Dict[str, float] = field(default_factory=dict)


@dataclass
class GraphDeps:
    confidence_threshold: float = 0.9
    retries: int = 3
    temperature: float = 0.0
    max_tokens: int = 4096
    prompt_version: str | Dict[str, str] | None = None
    meta_routing: bool = True
    model_assignments: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.prompt_version is None:
            self.prompt_version = {k: "v1" for k in _AGENT_KEYS}
        elif isinstance(self.prompt_version, str):
            self.prompt_version = {k: self.prompt_version for k in _AGENT_KEYS}


@dataclass
class MetaRoutingNode(BaseNode):
    """
    A meta routing run to select the model for intent agent, mission agent, latency agent, and decision agent.
    """
    async def run(self, context: GraphRunContext[GraphState, GraphDeps]) -> "IntentNode":
        logger.info("Running MetaRoutingNode")
        start = timeit.default_timer()

        if context.deps.meta_routing:
            agent = create_meta_routing_agent(
                retries=context.deps.retries,
                temperature=context.deps.temperature,
                max_tokens=context.deps.max_tokens,
                available_models=context.state.available_models,
                prompt_version=context.deps.prompt_version.get("meta", None),
            )
            prompt = f"Query: {context.state.query}"
            try:
                logger.debug("Running MetaRoutingNode")
                response = await run_agent(agent, prompt)
                meta_routing_decision = response.output
                for role in AGENT_ROLES:
                    model_key = getattr(meta_routing_decision, f"{role}_model")
                    if model_key in context.state.available_models:
                        context.state.model_assignments[role] = model_key
                    else:
                        context.state.model_assignments[role] = context.deps.model_assignments[role]
                step_cost = _compute_cost(response.usage, response.model_key)
                context.state.step_cost["meta_routing"] = step_cost
                context.state.cost += step_cost
            except Exception as e:
                context.state.model_assignments = context.deps.model_assignments
                context.state.errors["meta_routing"] = str(e)
                logger.error("MetaRoutingNode failed, using defaults: %s", e)
        else:
            context.state.model_assignments = context.deps.model_assignments
        elapsed = (timeit.default_timer() - start) * 1000
        context.state.step_latency_ms["meta_routing"] = elapsed
        context.state.routing_latency_ms += elapsed
        logger.info(f"Meta routing completed in {elapsed:.1f}ms | assignments={context.state.model_assignments}")
        return IntentNode()


@dataclass
class IntentNode(BaseNode):
    """
    Classify the query intent
    """
    async def run(self, context: GraphRunContext[GraphState, GraphDeps]) -> "ParallelScoringNode | FastDecisionNode":
        logger.info("Running IntentNode")
        start = timeit.default_timer()
        agent_model_key = context.state.model_assignments["intent"]
        agent = create_intention_agent(
            model_key=agent_model_key,
            retries=context.deps.retries,
            temperature=context.deps.temperature,
            max_tokens=context.deps.max_tokens,
            prompt_version=context.deps.prompt_version.get("intent", None),
        )
        prompt = f"Query: {context.state.query}"

        try:
            logger.debug("Running IntentNode")
            response = await run_agent(agent, prompt)
            intent_classification = response.output
            context.state.intent = intent_classification
            step_cost = _compute_cost(response.usage, response.model_key)
            context.state.step_cost["intent"] = step_cost
            context.state.cost += step_cost
        except Exception as e:
            context.state.errors["intent"] = str(e)
            logger.error("IntentNode failed, using fallback: %s", e)
            context.state.intent = IntentClassification(
                intent=QueryIntent.EXPLANATION,
                confidence=0.0,
                reasoning="Fallback: fail to classify intent",
            )
        elapsed = (timeit.default_timer() - start) * 1000
        context.state.step_latency_ms["intent"] = elapsed
        context.state.routing_latency_ms += elapsed
        logger.info(f"Intent agent completed in {elapsed:.1f}ms | "
                    f"intent={context.state.intent.intent.value}, "
                    f"confidence={context.state.intent.confidence}, reason={context.state.intent.reasoning}")

        if self._can_fast_route(context.state.intent, context.deps.confidence_threshold):
            logger.info("Intent is clear enough for fast routing, skipping ParallelScoringNode")
            return FastDecisionNode()
        return ParallelScoringNode()

    @staticmethod
    def _can_fast_route(intent: IntentClassification, confidence_threshold: float = 0.9) -> bool:
        """
        Return True if intent is enough to make a routing decision,
        skipping the ParallelScoringNode.

        Fast routing is allowed when:
        - The classifier is highly confident, AND
        - The intent maps to a clear model tier (SIMPLE → small/edge, REASONING → reasoning).
        """
        fast_route_intents = {QueryIntent.SIMPLE_FACTUAL}
        return intent.confidence >= confidence_threshold and intent.intent in fast_route_intents


class ParallelScoringNode(BaseNode):
    """
    Run Mission Criticality scoring agent and Latency Criticality scoring agent in parallel with multi-thread (IO-Bound)
    """
    async def run(self, context: GraphRunContext[GraphState, GraphDeps]) -> "DecisionNode":
        logger.info("Running ParallelScoringNode")
        start = timeit.default_timer()
        intent = context.state.intent

        prompt = f"""
        User query:
        {context.state.query}
        
        Intent classification result:
        - intent: {intent.intent.value}
        - confidence: {intent.confidence}
        - reasoning: {intent.reasoning}
        """

        mission_prompt = f"""
        {prompt}
        
        Estimate the mission criticality score for this query.
        """
        latency_prompt = f"""
        {prompt}
        
        Estimate the latency criticality score for this query.
        """

        mission_agent = create_mission_criticality_agent(
            model_key=context.state.model_assignments["mission"],
            retries=context.deps.retries,
            temperature=context.deps.temperature,
            max_tokens=context.deps.max_tokens,
            prompt_version=context.deps.prompt_version.get("mission", None),
        )
        latency_agent = create_latency_criticality_agent(
            model_key=context.state.model_assignments["latency"],
            retries=context.deps.retries,
            temperature=context.deps.temperature,
            max_tokens=context.deps.max_tokens,
            prompt_version=context.deps.prompt_version.get("latency", None),
        )

        parallel_result = await asyncio.gather(
            self.single_agent_run(mission_agent, mission_prompt),
            self.single_agent_run(latency_agent, latency_prompt, delay=0.5),
            return_exceptions=True,
        )
        mission_result = parallel_result[0]
        latency_result = parallel_result[1]

        if mission_result["result"] is None:
            context.state.errors["mission"] = mission_result['error']
            logger.error("MissionCriticality agent failed: %s", mission_result['error'])
            context.state.mission = MissionCriticality(
                score=0.5,
                confidence=0.0,
                reasoning="Fallback: fail to give mission criticality",
            )
        else:
            context.state.mission = mission_result['result']
            logger.info(f"Mission Agent: score={context.state.mission.score}, reasoning={context.state.mission.reasoning}")

        if latency_result["result"] is None:
            context.state.errors["latency"] = latency_result['error']
            logger.error("LatencyCriticality agent failed: %s", latency_result['error'])
            context.state.latency = LatencyCriticality(
                score=0.5,
                confidence=0.0,
                reasoning="Fallback: fail to give latency criticality",
            )
        else:
            context.state.latency = latency_result['result']
            logger.info(f"Latency Agent: score={context.state.latency.score}, reasoning={context.state.latency.reasoning}")

        context.state.step_latency_ms["mission"] = (mission_result["finish_time"] - start) * 1000
        context.state.step_latency_ms["latency"] = (latency_result["finish_time"] - start) * 1000

        context.state.step_cost["mission"] = mission_result["cost"]
        context.state.step_cost["latency"] = latency_result["cost"]
        context.state.cost += mission_result["cost"] + latency_result["cost"]

        elapsed = (timeit.default_timer() - start) * 1000
        context.state.step_latency_ms["scoring"] = elapsed
        context.state.routing_latency_ms += elapsed
        logger.info(f"ParallelScoringNode completed in {elapsed:.1f}ms | "
                    f"mission={context.state.mission.score:.2f} latency={context.state.latency.score:.2f}")
        return DecisionNode()

    @staticmethod
    async def single_agent_run(agent, prompt, delay: float = 0.0):
        if delay > 0:
            await asyncio.sleep(delay)
        logger.debug(f"Agent {agent.name} running")
        while True:
            try:
                response = await run_agent(agent, prompt)
                result = response.output
                cost = _compute_cost(response.usage, response.model_key)
                error = ""
                break
            except Exception as e:
                result = None
                cost = 0.0
                error = str(e)
                break
        finish_time = timeit.default_timer()
        logger.debug(f"Agent {agent.name} finished")
        return {
            "result": result,
            "cost": cost,
            "error": error,
            "finish_time": finish_time,
        }


class FastDecisionNode(BaseNode):
    """
    Skip scoring and decide directly from intent. Used for high-confidence simple queries.
    """
    async def run(self, context: GraphRunContext[GraphState, GraphDeps]) -> End[RoutingDecision]:
        logger.info("Running FastDecisionNode")
        start = timeit.default_timer()
        intent = context.state.intent

        available = context.state.available_models

        # Only SIMPLE_FACTUAL reaches FastDecisionNode
        model_key, deployment = _pick_model(
            available, [ModelTier.SMALL, ModelTier.MEDIUM], prefer_edge=True
        )

        context.state.decision = RoutingDecision(
            model_key=model_key,
            deployment=deployment,
            confidence=1.0,
            reasoning=f"Fast route: intent={intent.intent.value} confidence={intent.confidence:.2f}",
        )

        elapsed = (timeit.default_timer() - start) * 1000
        context.state.step_latency_ms["fast_decision"] = elapsed
        context.state.routing_latency_ms += elapsed
        logger.info(f"FastDecisionNode completed in {elapsed:.1f}ms | "
                    f"model={model_key}, deployment={deployment}")
        return End(context.state.decision)



class DecisionNode(BaseNode):
    """
    Make routing decision base on Intent, Mission and Latency.
    """
    async def run(self, context: GraphRunContext[GraphState, GraphDeps]) -> End[RoutingDecision]:
        start = timeit.default_timer()
        agent_model_key = context.state.model_assignments["decision"]
        agent = creat_decision_making_agent(
            model_key=agent_model_key,
            retries=context.deps.retries,
            temperature=context.deps.temperature,
            max_tokens=context.deps.max_tokens,
            prompt_version=context.deps.prompt_version.get("decision", None),
            available_models=context.state.available_models,
        )
        prompt = f"""
        Query:
        {context.state.query}
        
        Upstream analysis:
        
        Intent classification:
        - intent: {context.state.intent.intent.value}
        - confidence: {context.state.intent.confidence}
        - reasoning: {context.state.intent.reasoning}
        
        Mission criticality:
        - score: {context.state.mission.score}
        - confidence: {context.state.mission.confidence}
        - reasoning: {context.state.mission.reasoning}
        
        Latency criticality:
        - score: {context.state.latency.score}
        - confidence: {context.state.latency.confidence}
        - reasoning: {context.state.latency.reasoning}
        """

        try:
            logger.debug("Running DecisionNode Agent")
            response = await run_agent(agent, prompt)
            result = response.output
            context.state.decision = result
            step_cost = _compute_cost(response.usage, response.model_key)
            context.state.step_cost["decision"] = step_cost
            context.state.cost += step_cost
        except Exception as e:
            context.state.decision = self._fall_back(context.state)
            context.state.errors["decision"] = str(e)
            logger.error("DecisionNode failed, using fallback model: %s", e)

        elapsed = (timeit.default_timer() - start) * 1000
        context.state.step_latency_ms["decision"] = elapsed
        context.state.routing_latency_ms += elapsed

        logger.info(f"DecisionNode completed in {elapsed:.1f}ms | "
                    f"model={context.state.decision.model_key}, deployment={context.state.decision.deployment}")
        return End(context.state.decision)

    @staticmethod
    def _fall_back(state: GraphState) -> RoutingDecision:
        available = state.available_models
        intent = state.intent.intent.value

        if state.mission.score > 0.6:
            model_key, deployment = _pick_model(available, [ModelTier.LARGE, ModelTier.REASONING, ModelTier.MEDIUM])
        elif state.latency.score > 0.6:
            model_key, deployment = _pick_model(available, [ModelTier.SMALL, ModelTier.MEDIUM], prefer_edge=True)
        elif intent in ["coding", "reasoning", "analysis"]:
            model_key, deployment = _pick_model(available, [ModelTier.REASONING, ModelTier.LARGE, ModelTier.MEDIUM])
        else:
            model_key, deployment = _pick_model(available, [ModelTier.MEDIUM, ModelTier.SMALL, ModelTier.LARGE])

        return RoutingDecision(
            model_key=model_key,
            deployment=deployment,
            confidence=0.5,
            reasoning="Fallback: fail to give routing decision",
        )


def build_routing_decision_graph():
    return Graph(
        nodes=[MetaRoutingNode, IntentNode, ParallelScoringNode, FastDecisionNode, DecisionNode], state_type=GraphState
    )