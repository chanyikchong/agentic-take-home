from typing import Optional, Literal

from pydantic import BaseModel, Field
from enum import Enum


class MetaRoutingDecision(BaseModel):
    intent_model: str = Field(..., description="Model for intent classification.")
    mission_model: str = Field(..., description="Model for mission scoring.")
    latency_model: str = Field(..., description="Model for latency scoring.")
    decision_model: str = Field(..., description="Model for decision scoring.")
    reasoning: str = Field(..., description="Reasoning for intent classification. Keep it tighten")


class QueryIntent(str, Enum):
    SIMPLE_FACTUAL = "simple_factual"
    EXPLANATION = "explanation"
    ANALYSIS = "analysis"
    REASONING = "reasoning"
    CODING = "coding"


class IntentClassification(BaseModel):
    intent: QueryIntent
    confidence: float = Field(..., ge=0, le=1, description="The confidence score of the intent classification.")
    reasoning: str = Field(..., description="The reasoning of the classification. Keep it tighten")


class MissionCriticality(BaseModel):
    score: float = Field(..., ge=0, le=1, description="The score of the mission criticality. 0 = low stakes, 1 = must be accurate")
    confidence: float = Field(..., ge=0, le=1, description="The confidence of giving this score")
    reasoning: str = Field(..., description="The reasoning of the mission criticality. Keep it tighten")


class LatencyCriticality(BaseModel):
    score: float = Field(..., ge=0, le=1, description="The score of the latency criticality. 0 = can wait, 1 = needs instant response")
    confidence: float = Field(..., ge=0, le=1, description="The confidence of giving this score")
    reasoning: str = Field(..., description="The reasoning of the latency criticality. Keep it tighten")


class RoutingDecision(BaseModel):
    model_key: str = Field(..., description="The ID of the model to route to.")
    deployment: Literal["edge", "cloud"] = Field(..., description="The deployment of the model. Select from 'edge' or 'cloud'.")
    confidence: float = Field(..., ge=0, le=1, description="The confidence of giving this score")
    reasoning: Optional[str] = Field(None, description="The reasoning of the routing decision. Keep it tighten")
