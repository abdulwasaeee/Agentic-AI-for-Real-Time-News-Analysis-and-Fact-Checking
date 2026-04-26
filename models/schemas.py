"""Frozen JSON interface contracts for the Fact-Check Agent.

These are the single source of truth for all agent handoffs.
Do not change field names without versioning and notifying downstream consumers.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class EntityRef(BaseModel):
    """A named entity attached to a claim."""

    entity_id: str
    name: str
    entity_type: str  # "person" | "organization" | "country" | "location" | "event" | "product"
    sentiment: str    # "positive" | "negative" | "neutral"


class FactCheckInput(BaseModel):
    """Input handed to the LangGraph graph via graph.invoke({"input": ...})."""

    claim_id: str
    claim_text: str
    entities: list[EntityRef]
    source_url: str
    article_id: str
    image_caption: Optional[str] = None
    image_url: Optional[str] = None
    timestamp: datetime
    prefetched_chunks: list[str] = Field(default_factory=list)


class FactCheckOutput(BaseModel):
    """Output emitted by emit_output node and written to MemoryAgent."""

    verdict_id: str
    claim_id: str
    verdict: str                      # "supported" | "refuted" | "misleading"
    confidence_score: int = Field(ge=0, le=100)
    evidence_links: list[str]
    reasoning: str
    bias_score: float = Field(ge=0.0, le=1.0)
    cross_modal_flag: bool = False
    cross_modal_explanation: Optional[str] = None
    last_verified_at: Optional[datetime] = None
    revalidation_needed: bool = False


class SimilarClaim(BaseModel):
    claim_id: str
    claim_text: str
    verdict_label: Optional[str] = None
    verdict_confidence: Optional[float] = None
    distance: float
    verified_at: Optional[datetime] = None


class MemoryQueryRequest(BaseModel):
    claim_text: str
    top_k: int = 5


class MemoryQueryResponse(BaseModel):
    results: list[SimilarClaim]
    max_confidence: float = 0.0
