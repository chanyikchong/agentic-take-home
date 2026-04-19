import json
from typing import List, Optional

from pydantic_ai import Agent

from src.model_registry import MODEL_REGISTRY, ModelTier, get_models_by_tier, CLOUD_MODELS, EDGE_MODELS
from src.latency import EDGE_LATENCY_MULTIPLIER, CLOUD_LATENCY_MULTIPLIER
from .models import RoutingDecision, IntentClassification, MissionCriticality, LatencyCriticality, MetaRoutingDecision
from .prompts import INTENT_PROMPT, META_PROMPT, MISSION_PROMPT, DECISION_PROMPT, LATENCY_PROMPT
from .utils import SYSTEM_PROMPT_NOT_ALLOW_MODEL, TOOL_USE_NOT_SUPPORTED_MODELS


def _model_string(model_key: str) -> str:
    return f"openrouter:{MODEL_REGISTRY[model_key].model_id}"


def _build_agent(model_key: str, system_prompt: str, **agent_kwargs) -> Agent:
    """Create an Agent, handling models that lack system prompt and/or tool use support.

    For models without tool use: removes output_type, appends the JSON schema to the prompt,
    and stores the type for manual parsing in run_agent().

    For models without system prompt: stores prompt and add to user message.
    """
    output_type = None
    parse_retries = 1

    if model_key in TOOL_USE_NOT_SUPPORTED_MODELS:
        # Extract output_type — we'll handle structured output via prompt + parsing
        output_type = agent_kwargs.pop('output_type', None)
        # we do the retries with our implemented agent run
        parse_retries = agent_kwargs.get('retries', 1)
        if output_type is not None:
            schema = json.dumps(output_type.model_json_schema(), indent=2)
            system_prompt += (
                f"\n\nYou MUST respond with ONLY valid JSON matching this schema:\n"
                f"{schema}\n"
                f"Do not include markdown fences, backticks, or any text outside the JSON."
            )

    if model_key in SYSTEM_PROMPT_NOT_ALLOW_MODEL:
        agent = Agent(model=_model_string(model_key), **agent_kwargs)
        agent._prepend_prompt = system_prompt
    else:
        agent = Agent(model=_model_string(model_key), system_prompt=system_prompt, **agent_kwargs)
        agent._prepend_prompt = None

    agent._output_type = output_type
    agent._model_key = model_key
    agent._parse_retries = parse_retries
    return agent

def _get_routing_models():
    """
    Return list of routing decision models.
    Only small and medium models are allowed to use as the LLM in the router
    """
    models = get_models_by_tier(ModelTier.SMALL)
    models.extend(get_models_by_tier(ModelTier.MEDIUM))
    return models

def _formulate_available_model(models: list) -> str:
    """
    Format available information as
    - model_key (model tier = ModelTier.Tier, cost_in=number/M, cost_out=number/M)
    """
    model_string = [
        (f"- {key}: {MODEL_REGISTRY[key].display_name} (model tier={MODEL_REGISTRY[key].tier.value}, "
         f"model name={MODEL_REGISTRY[key].display_name}, "
         f"cost_in=${MODEL_REGISTRY[key].cost_per_million_input}/M, "
         f"cost_out=${MODEL_REGISTRY[key].cost_per_million_output}/M)")
        for key in models]
    model_string = "\n".join(model_string)
    return model_string

def create_meta_routing_agent(
        retries: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        available_models: Optional[List[str]] = None,
        prompt_version: str = "v1",
) -> Agent:
    """
    Create a Meta Routing Agent.
    The system prompt should be a string including {available_models} as placeholder
    """
    # use the cheapest model for meta routing
    # meta_routing_models = get_models_by_tier(ModelTier.SMALL)
    available_decision_models = _get_routing_models()
    if available_models:
        available_decision_models = list(set(available_decision_models).intersection(available_models))

    min_cost_model_key = min(available_decision_models, key=lambda model: MODEL_REGISTRY[model].cost_per_million_input)

    prompt = META_PROMPT.get(prompt_version, META_PROMPT["v1"]).format(available_models=_formulate_available_model(available_decision_models))

    meta_routing_agent = _build_agent(
        min_cost_model_key,
        system_prompt=prompt,
        name="meta-router",
        retries=retries,
        model_settings={
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        output_type=MetaRoutingDecision,
    )
    return meta_routing_agent


def create_intention_agent(
        model_key: str,
        retries: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        prompt_version: str = "v1",
) -> Agent:
    """
    Create an Agent, handling models that lack system prompt and tool use support.
    """
    system_prompt = INTENT_PROMPT.get(prompt_version, INTENT_PROMPT["v1"])
    agent = _build_agent(
        model_key,
        system_prompt=system_prompt,
        name="intent-classifier",
        retries=retries,
        model_settings={
            "temperature": temperature,
            "max_tokens": max_tokens
        },
        output_type=IntentClassification,
    )
    return agent


def create_mission_criticality_agent(
        model_key: str,
        retries: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        prompt_version: str = "v1",
) -> Agent:
    """
    Creates a mission criticality decision agent.
    """
    system_prompt = MISSION_PROMPT.get(prompt_version, MISSION_PROMPT["v1"])
    agent = _build_agent(
        model_key,
        system_prompt=system_prompt,
        name="mission-score-agent",
        retries=retries,
        model_settings={
            "temperature": temperature,
            "max_tokens": max_tokens
        },
        output_type=MissionCriticality,
    )
    return agent


def create_latency_criticality_agent(
        model_key: str,
        retries: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        prompt_version: str = "v1",
) -> Agent:
    """
    Create a latency criticality agent
    """
    system_prompt = LATENCY_PROMPT.get(prompt_version, LATENCY_PROMPT["v1"])
    agent = _build_agent(
        model_key,
        system_prompt=system_prompt,
        name="latency-score-agent",
        retries=retries,
        model_settings={
            "temperature": temperature,
            "max_tokens": max_tokens
        },
        output_type=LatencyCriticality,
    )
    return agent


def creat_decision_making_agent(
        model_key: str,
        retries: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        prompt_version: str = "v1",
        available_models: Optional[List[str]] = None
) -> Agent:
    """
    Create Decision Agent
    Provide available models and deployments in the system prompt
    The system prompt should be a string including {edge_latency_multiplier}, {cloud_latency_multiplier},
    {edge_models}, {cloud_models} as placeholder
    """
    if available_models:
        edge_model = list(set(available_models).intersection(set(EDGE_MODELS)))
        cloud_model = list(set(available_models).intersection(set(CLOUD_MODELS)))
    else:
        edge_model = EDGE_MODELS
        cloud_model = CLOUD_MODELS

    system_prompt = DECISION_PROMPT.get(prompt_version, DECISION_PROMPT["v1"]).format(
        edge_latency_multiplier=EDGE_LATENCY_MULTIPLIER,
        cloud_latency_multiplier=CLOUD_LATENCY_MULTIPLIER,
        edge_models=_formulate_available_model(edge_model),
        cloud_models=_formulate_available_model(cloud_model),
    )
    agent = _build_agent(
        model_key,
        system_prompt=system_prompt,
        name="decision-agent",
        retries=retries,
        model_settings={
            "temperature": temperature,
            "max_tokens": max_tokens
        },
        output_type=RoutingDecision,
    )
    return agent
