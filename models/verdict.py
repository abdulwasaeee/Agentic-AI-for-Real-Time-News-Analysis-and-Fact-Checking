"""Verdict data model (written by Fact-Check Agent)."""

from datetime import datetime

from pydantic import BaseModel, Field


class Verdict(BaseModel):
    verdict_id: str
    claim_id: str
    label: str  # "supported", "refuted", "misleading"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_summary: str
    bias_score: float = Field(ge=0.0, le=1.0)
    image_mismatch: bool = False
    verified_at: datetime
