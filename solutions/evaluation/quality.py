"""
Patched Quality Evaluation Module (LLM-as-a-Judge).

Copied from src/quality.py with one key fix:
- Uses JSON-in-prompt instead of output_type for structured output.
- This allows models WITHOUT tool/function calling support to serve as evaluators.
- Default evaluator changed from trinity-mini (offline) to gemma-3-27b (available).
"""

import json
import asyncio
import logging
import random
from typing import Dict, List

from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError

from src.model_registry import MODEL_REGISTRY
from solutions.utils import _parse_json_output, SYSTEM_PROMPT_NOT_ALLOW_MODEL

logger = logging.getLogger(__name__)

# Default evaluator model — must be available on OpenRouter and not require tool use
DEFAULT_EVALUATOR_MODEL = "gemma-3-27b"


# ── Models (unchanged from src/quality.py) ──────────────────────────────────

class QualityDimension(BaseModel):
    """Individual quality dimension score"""
    dimension: str = Field(description="Name of the quality dimension")
    score: float = Field(ge=0, le=10, description="Score from 0-10")
    reasoning: str = Field(description="Brief explanation for the score")


class QualityEvaluation(BaseModel):
    """Complete quality evaluation result"""
    overall_score: float = Field(ge=0, le=10, description="Overall quality score")
    dimensions: List[QualityDimension] = Field(description="Individual dimension scores")
    summary: str = Field(description="Brief summary of the evaluation")
    model_used: str = Field(description="Model that generated the response")
    evaluator_model: str = Field(description="Model used for evaluation")


class QualityEvaluationResult(BaseModel):
    """Structured output for quality evaluation"""
    overall_score: float = Field(ge=0, le=10, description="Overall quality score from 0-10")
    dimensions: List[QualityDimension] = Field(description="Individual dimension scores")
    summary: str = Field(description="Brief 2-3 sentence summary of the evaluation")


class BatchQualityEvaluationResult(BaseModel):
    """Structured output for batch evaluation of multiple items."""
    evaluations: List[QualityEvaluationResult] = Field(
        description="Evaluation results in the same order as the input items"
    )


# ── Prompts (unchanged from src/quality.py) ─────────────────────────────────

DEFAULT_EVALUATION_PROMPT = """
You are an expert evaluator assessing the quality of AI assistant responses.

Evaluate responses on these dimensions (0-10 scale):

1. **Accuracy**: Is the information factually correct?
2. **Relevance**: Does it directly address the query?
3. **Completeness**: Does it cover all important aspects?
4. **Clarity**: Is it well-organized and easy to understand?
5. **Helpfulness**: Would this response satisfy the user's need?

Provide an overall score (0-10) that reflects the weighted importance of these dimensions,
along with individual dimension scores and brief reasoning for each.

Be objective and consistent in your evaluations. Consider the context and difficulty
of the original query when assigning scores.
"""

SINGLE_EVAL_EXAMPLE = """
## Example

Input:
## Original Query
What is 2+2?

## Response to Evaluate
4.

Output:
{
  "overall_score": 6.0,
  "dimensions": [
    {"dimension": "accuracy", "score": 10.0, "reasoning": "Correct arithmetic result."},
    {"dimension": "relevance", "score": 10.0, "reasoning": "Directly answers the question."},
    {"dimension": "completeness", "score": 4.0, "reasoning": "No working shown, no context; bare numeric answer."},
    {"dimension": "clarity", "score": 8.0, "reasoning": "Unambiguous but terse."},
    {"dimension": "helpfulness", "score": 5.0, "reasoning": "Answers the question but offers nothing beyond it."}
  ],
  "summary": "Correct but shallow one-token answer. Adequate for trivial arithmetic, not exemplary."
}
"""

BATCH_EVAL_EXAMPLE = """
## Example

Input:
## Item 1
**Query:** Explain gravity in one sentence.
**Response:** Gravity is a force that attracts objects with mass toward each other.

## Item 2
**Query:** Write a Python function to reverse a string.
**Response:** ```python
def reverse(s): return s[::-1]
```

Output:
{
  "evaluations": [
    {
      "overall_score": 7.0,
      "dimensions": [
        {"dimension": "accuracy", "score": 8.0, "reasoning": "Correct at a layperson level; omits that it is curvature of spacetime in GR."},
        {"dimension": "relevance", "score": 10.0, "reasoning": "Directly answers the prompt."},
        {"dimension": "completeness", "score": 6.0, "reasoning": "One sentence as requested; could mention mass-dependence more precisely."},
        {"dimension": "clarity", "score": 9.0, "reasoning": "Plain, readable prose."},
        {"dimension": "helpfulness", "score": 7.0, "reasoning": "Useful as a quick intuition but not deep."}
      ],
      "summary": "Solid one-sentence explanation; accurate but intentionally simplified."
    },
    {
      "overall_score": 6.0,
      "dimensions": [
        {"dimension": "accuracy", "score": 9.0, "reasoning": "Slice works for strings; returns reversed copy correctly."},
        {"dimension": "relevance", "score": 10.0, "reasoning": "Matches the request."},
        {"dimension": "completeness", "score": 4.0, "reasoning": "No type hints, no docstring, no handling of non-str input."},
        {"dimension": "clarity", "score": 7.0, "reasoning": "Readable but unnamed parameter and no explanation."},
        {"dimension": "helpfulness", "score": 6.0, "reasoning": "Works for the common case; minimal pedagogical value."}
      ],
      "summary": "Functional one-liner that solves the task but lacks polish expected of production code."
    }
  ]
}
"""


EVALUATOR_SYSTEM_PROMPT = """You are a strict expert evaluator assessing AI assistant responses. Use the FULL 0-10 scale.

## Score Calibration (USE THE FULL RANGE)
- 9-10: Exceptional. Flawless, comprehensive, expert-level. Reserve for truly outstanding responses.
- 7-8: Good. Solid answer with minor gaps or room for improvement.
- 5-6: Adequate. Gets the job done but has notable weaknesses.
- 3-4: Poor. Significant issues, missing key information, or partially incorrect.
- 1-2: Very poor. Mostly wrong, unhelpful, or misses the point.
- 0: Completely wrong or harmful.

## Dimensions to Evaluate
1. accuracy - Factually correct? Any errors or hallucinations? (Be strict - any factual error drops this significantly)
2. relevance - Directly addresses the query? Stays on topic?
3. completeness - Covers all important aspects? What's missing?
4. clarity - Well-organized? Easy to understand?
5. helpfulness - Would this actually help the user?

## Critical Instructions
- DO NOT default to high scores. Most responses are 5-8, not 9-10.
- A "correct but shallow" answer is 5-6, not 8-9.
- Deduct points for: verbosity, missing edge cases, lack of examples when helpful, generic advice.
- Perfect 10s should be rare - the response must be genuinely exceptional."""


# ── Agent creation (PATCHED: no output_type, uses JSON-in-prompt) ───────────

def _model_string(model_key: str) -> str:
    return f"openrouter:{MODEL_REGISTRY[model_key].model_id}"


def _build_schema_prompt(output_type: type, example: str | None = None) -> str:
    """Build a prompt suffix that tells the model to respond with JSON matching the schema.

    When `example` is provided, it is appended after the schema as a one-shot
    demonstration of the expected JSON shape. Examples use mid-range scores to
    avoid anchoring the evaluator toward the top of the scale.
    """
    schema = json.dumps(output_type.model_json_schema(), indent=2)
    parts = [
        "\n\nYou MUST respond with ONLY valid JSON matching this schema:",
        schema,
        "Do not include markdown fences, backticks, or any text outside the JSON.",
    ]
    if example:
        parts.append(example)
    return "\n".join(parts)


def create_evaluator_agent(
    evaluator_model: str = DEFAULT_EVALUATOR_MODEL,
    schema_type: type = QualityEvaluationResult,
    example: str | None = SINGLE_EVAL_EXAMPLE,
) -> Agent:
    """Create a pydantic-ai Agent for quality evaluation WITHOUT tool use.

    Instead of output_type, the JSON schema is embedded in the system prompt
    and the response is parsed manually.
    For models that don't support system prompts, the prompt is stored
    on the agent and prepended to user messages at call time.
    """
    evaluator_config = MODEL_REGISTRY.get(evaluator_model)
    if not evaluator_config:
        raise ValueError(f"Unknown evaluator model: {evaluator_model}")

    system_prompt = EVALUATOR_SYSTEM_PROMPT + _build_schema_prompt(schema_type, example=example)

    if evaluator_model in SYSTEM_PROMPT_NOT_ALLOW_MODEL:
        agent = Agent(model=_model_string(evaluator_model))
        agent._prepend_prompt = system_prompt
    else:
        agent = Agent(model=_model_string(evaluator_model), system_prompt=system_prompt)
        agent._prepend_prompt = None

    return agent


# Cache for evaluator agents
_evaluator_agents: Dict[str, Agent] = {}


def get_evaluator_agent(evaluator_model: str = DEFAULT_EVALUATOR_MODEL) -> Agent:
    """Get or create a cached evaluator agent."""
    if evaluator_model not in _evaluator_agents:
        _evaluator_agents[evaluator_model] = create_evaluator_agent(evaluator_model)
    return _evaluator_agents[evaluator_model]


# ── 429 retry helper ────────────────────────────────────────────────────────

async def _run_agent_with_retry(
    agent: Agent,
    user_prompt: str,
    output_type: type,
    max_retries: int = 3,
    parse_retries: int = 3,
    base_delay: float = 3.0,
    max_delay: float = 15.0,
):
    """Run agent with 429 retry (outer) and JSON parse retry (inner).

    Handles models that don't support system prompts by prepending
    the stored system prompt to the user message.
    """
    # Prepend system prompt for models that don't support it natively
    prepend = getattr(agent, '_prepend_prompt', None)
    if prepend:
        user_prompt = f"{prepend}\n\n{user_prompt}"

    last_429 = None
    for http_attempt in range(max_retries):
        try:
            last_parse_error = None
            prompt = user_prompt
            for parse_attempt in range(parse_retries):
                result = await agent.run(user_prompt=prompt)
                try:
                    parsed = _parse_json_output(result.output, output_type)
                    return parsed
                except (ValueError, ValidationError) as e:
                    last_parse_error = e
                    logger.warning(
                        "Eval JSON parse attempt %d/%d failed: %s",
                        parse_attempt + 1, parse_retries, e,
                    )
                    prompt = (
                        f"{user_prompt}\n\n"
                        f"Your previous response was not valid JSON.\n"
                        f"Error: {e}\n\n"
                        f"Please respond ONLY with valid JSON matching the required schema."
                    )
                    continue
            raise last_parse_error
        except Exception as e:
            if isinstance(e, ModelHTTPError) and e.status_code == 429:
                last_429 = e
                delay = min(base_delay * (2 ** http_attempt), max_delay) + random.uniform(0, base_delay)
                logger.warning("Evaluator rate limited (429), attempt %d/%d, waiting %.1fs", http_attempt + 1, max_retries, delay)
                await asyncio.sleep(delay)
                continue
            raise
    raise last_429


# ── Single evaluation ───────────────────────────────────────────────────────

async def evaluate_quality(
    query: str,
    response: str,
    model_key: str,
    evaluator_model: str = DEFAULT_EVALUATOR_MODEL,
) -> QualityEvaluation:
    """Evaluate response quality using LLM-as-a-judge (no tool use required)."""
    agent = get_evaluator_agent(evaluator_model)

    user_prompt = f"""Evaluate the following response:

## Original Query
{query}

## Response to Evaluate
{response}

Provide scores for accuracy, relevance, completeness, clarity, and helpfulness."""

    parsed = await _run_agent_with_retry(agent, user_prompt, QualityEvaluationResult)

    return QualityEvaluation(
        overall_score=parsed.overall_score,
        dimensions=parsed.dimensions,
        summary=parsed.summary,
        model_used=model_key,
        evaluator_model=evaluator_model,
    )


# ── Batch evaluation ────────────────────────────────────────────────────────

def _create_batch_evaluator_agent(evaluator_model: str) -> Agent:
    """Create a batch evaluator agent WITHOUT tool use."""
    return create_evaluator_agent(
        evaluator_model,
        schema_type=BatchQualityEvaluationResult,
        example=BATCH_EVAL_EXAMPLE,
    )


async def _evaluate_batch_chunk(
    items: List[tuple],
    evaluator_model: str,
) -> List[QualityEvaluation]:
    """Evaluate a small batch of items."""
    user_prompt = "Evaluate each of the following responses:\n\n"
    for i, (query, response, _) in enumerate(items, 1):
        user_prompt += f"## Item {i}\n"
        user_prompt += f"**Query:** {query}\n"
        user_prompt += f"**Response:** {response}\n\n"

    user_prompt += f"\nProvide exactly {len(items)} evaluations in the same order as the items above."

    agent = _create_batch_evaluator_agent(evaluator_model)
    parsed = await _run_agent_with_retry(agent, user_prompt, BatchQualityEvaluationResult)

    # Validate count
    if len(parsed.evaluations) != len(items):
        raise ValueError(
            f"Expected {len(items)} evaluations but got {len(parsed.evaluations)}"
        )

    return [
        QualityEvaluation(
            overall_score=e.overall_score,
            dimensions=e.dimensions,
            summary=e.summary,
            model_used=items[i][2],
            evaluator_model=evaluator_model,
        )
        for i, e in enumerate(parsed.evaluations)
    ]


async def evaluate_quality_batch(
    items: List[tuple],
    evaluator_model: str = DEFAULT_EVALUATOR_MODEL,
    chunk_size: int = 5,
) -> List[QualityEvaluation]:
    """Evaluate multiple query-response pairs in batched API calls."""
    if not items:
        return []

    all_results = []
    for i in range(0, len(items), chunk_size):
        chunk = items[i:i + chunk_size]
        chunk_num = (i // chunk_size) + 1
        total_chunks = (len(items) + chunk_size - 1) // chunk_size
        print(f"    Evaluating batch {chunk_num}/{total_chunks} ({len(chunk)} items)...")

        chunk_results = await _evaluate_batch_chunk(chunk, evaluator_model)
        all_results.extend(chunk_results)

    return all_results
