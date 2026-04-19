import re

import httpx

from solutions.models import RoutingDecision
from src import EDGE_MODELS
from src.model_registry import MODEL_REGISTRY
from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

DEFAULT_MODEL_ASSIGNMENTS = {
    "intent": "gemma-3n-e4b",
    "mission": "gemma-3n-e4b",
    "latency": "gemma-3n-e4b",
    "decision": "gemma-3n-e4b",
}

AGENT_ROLES = ["intent", "mission", "latency", "decision"]


SYSTEM_PROMPT_NOT_ALLOW_MODEL = ["gemma-3-4b", "gemma-3n-e4b", "gemma-3-12b", "gemma-3-27b"]

# Models that don't support tool use / function calling on OpenRouter.
# These cannot be used as pipeline agents (which rely on pydantic-ai structured output),
# but can still be selected as final routing targets.
TOOL_USE_NOT_SUPPORTED_MODELS = ["gemma-3-4b", "gemma-3n-e4b", "gemma-3-12b", "gemma-3-27b", "nemotron-nano"]

def check_available_models() -> list[str]:
    """
    Checks whether all registry models are available on OpenRouter.
    """
    available = []
    with httpx.Client(timeout=10.0) as client:
        for key, cfg in MODEL_REGISTRY.items():
            try:
                resp = client.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": cfg.model_id,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                    },
                )
                if resp.status_code == 200:
                    available.append(key)
            except Exception:
                pass
    return available if available else list(MODEL_REGISTRY.keys())


def check_deployment_available(decision: RoutingDecision) -> RoutingDecision:
    """
    Checks whether a deployment is available for the chosen model.
    If a model is available on the edge, use edge.
    """
    model_key = decision.model_key
    if decision.deployment == "cloud":
        if model_key in EDGE_MODELS:
            decision.deployment = "edge"
        return decision
    elif decision.deployment == "edge":
        model_key = decision.model_key
        if model_key not in EDGE_MODELS:
            decision.deployment = "cloud"
        return decision
    else:
        raise ValueError(f"Unknown deployment {decision.deployment}")


def _parse_json_output(text: str, output_type):
    """
    Extract JSON from model text response and validate against the pydantic model.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.strip().rstrip("`")
    # Find the first JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return output_type.model_validate_json(match.group())
    raise ValueError(f"Could not extract JSON from model response: {text[:200]}")