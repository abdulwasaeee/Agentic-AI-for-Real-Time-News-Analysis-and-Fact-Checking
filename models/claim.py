"""Claim and Entity data models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class MentionSentiment(BaseModel):
    entity_id: str
    name: str
    entity_type: str  # "person", "organization", "country", etc.
    sentiment: str    # "positive", "negative", "neutral"


class Claim(BaseModel):
    claim_id: str
    article_id: str
    claim_text: str
    claim_type: Optional[str] = None  # "statistical", "attribution", "causal", "predictive"
    extracted_at: datetime
    status: str = "pending"  # "pending", "verified", "expired"
    entities: list[MentionSentiment] = Field(default_factory=list)


class Entity(BaseModel):
    entity_id: str
    name: str
    entity_type: str
    current_credibility: float = Field(default=0.5, ge=0.0, le=1.0)
    total_claims: int = 0
    accurate_claims: int = 0
    first_seen: datetime
    last_seen: datetime
