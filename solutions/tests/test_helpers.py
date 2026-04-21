import pytest
from pydantic import BaseModel, ValidationError

from solutions.models import RoutingDecision
from solutions.pipelines import _pick_model
from solutions.utils import check_deployment_available, _parse_json_output
from src.model_registry import MODEL_REGISTRY, ModelTier, EDGE_MODELS


ALL_MODEL_KEYS = list(MODEL_REGISTRY.keys())


class _TinyModel(BaseModel):
    name: str
    value: int


class TestDeploymentValidation:
    def test_edge_model_on_cloud_gets_corrected_to_edge(self):
        """SMALL models on cloud should be corrected to edge."""
        for model_key in EDGE_MODELS:
            decision = RoutingDecision(
                model_key=model_key, deployment="cloud", confidence=0.9, reasoning="test"
            )
            result = check_deployment_available(decision)
            assert result.deployment == "edge"

    def test_non_edge_model_on_edge_gets_corrected_to_cloud(self):
        """Non-SMALL models on edge should be corrected to cloud."""
        non_edge_models = [k for k in MODEL_REGISTRY if k not in EDGE_MODELS]
        for model_key in non_edge_models:
            decision = RoutingDecision(
                model_key=model_key, deployment="edge", confidence=0.9, reasoning="test"
            )
            result = check_deployment_available(decision)
            assert result.deployment == "cloud"

    def test_valid_edge_deployment_unchanged(self):
        for model_key in EDGE_MODELS:
            decision = RoutingDecision(
                model_key=model_key, deployment="edge", confidence=0.9, reasoning="test"
            )
            result = check_deployment_available(decision)
            assert result.deployment == "edge"

    def test_valid_cloud_deployment_for_large_model_unchanged(self):
        decision = RoutingDecision(
            model_key="llama-3.3-70b", deployment="cloud", confidence=0.9, reasoning="test"
        )
        result = check_deployment_available(decision)
        assert result.deployment == "cloud"


class TestPickModel:
    def test_picks_cheapest_in_preferred_tier(self):
        model_key, deployment = _pick_model(
            ALL_MODEL_KEYS, [ModelTier.SMALL, ModelTier.MEDIUM]
        )
        assert MODEL_REGISTRY[model_key].tier == ModelTier.SMALL

    def test_prefers_edge_when_requested(self):
        model_key, deployment = _pick_model(
            ALL_MODEL_KEYS, [ModelTier.SMALL], prefer_edge=True
        )
        assert deployment == "edge"
        assert model_key in EDGE_MODELS

    def test_falls_through_tiers(self):
        """If preferred tier is unavailable, falls to next tier."""
        medium_and_large = [
            k for k in ALL_MODEL_KEYS
            if MODEL_REGISTRY[k].tier in (ModelTier.MEDIUM, ModelTier.LARGE)
        ]
        model_key, deployment = _pick_model(
            medium_and_large, [ModelTier.SMALL, ModelTier.MEDIUM]
        )
        assert MODEL_REGISTRY[model_key].tier == ModelTier.MEDIUM

    def test_ultimate_fallback(self):
        """If no preferred tier matches, picks cheapest available."""
        small_only = [k for k in ALL_MODEL_KEYS if MODEL_REGISTRY[k].tier == ModelTier.SMALL]
        model_key, deployment = _pick_model(
            small_only, [ModelTier.REASONING, ModelTier.LARGE]
        )
        assert model_key in small_only


class TestParseJsonOutput:
    def test_parses_plain_json(self):
        result = _parse_json_output('{"name": "x", "value": 1}', _TinyModel)
        assert result == _TinyModel(name="x", value=1)

    def test_strips_json_markdown_fence(self):
        text = '```json\n{"name": "x", "value": 1}\n```'
        result = _parse_json_output(text, _TinyModel)
        assert result == _TinyModel(name="x", value=1)

    def test_strips_plain_markdown_fence(self):
        text = '```\n{"name": "x", "value": 1}\n```'
        result = _parse_json_output(text, _TinyModel)
        assert result == _TinyModel(name="x", value=1)

    def test_extracts_json_from_surrounding_prose(self):
        text = 'Sure, here is the answer: {"name": "x", "value": 1}. Hope that helps!'
        result = _parse_json_output(text, _TinyModel)
        assert result == _TinyModel(name="x", value=1)

    def test_raises_value_error_when_no_json_object(self):
        with pytest.raises(ValueError):
            _parse_json_output("no json in this response", _TinyModel)

    def test_raises_validation_error_on_schema_mismatch(self):
        with pytest.raises(ValidationError):
            _parse_json_output('{"name": "x"}', _TinyModel)
