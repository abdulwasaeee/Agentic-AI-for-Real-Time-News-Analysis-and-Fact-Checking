"""LangGraph state schema for the Fact-Check Agent graph."""
from datetime import datetime
from typing import Optional, TypedDict

from models.schemas import FactCheckInput, FactCheckOutput, MemoryQueryResponse


class FactCheckState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────
    input: FactCheckInput

    # ── Memory query results ──────────────────────────────────────────────
    memory_results: Optional[MemoryQueryResponse]
    entity_context: list[dict]

    # ── Routing ───────────────────────────────────────────────────────────
    route: Optional[str]
    revalidation_needed: Optional[bool]
    retrieval_gate_needed: Optional[bool]

    # ── Intermediate data ─────────────────────────────────────────────────
    retrieved_chunks: list[str]
    sub_claims: list[str]
    debate_transcript: Optional[str]

    # ── Source credibility ────────────────────────────────────────────────
    source_credibility: Optional[dict]

    # ── Cross-modal ───────────────────────────────────────────────────────
    cross_modal_flag: bool
    cross_modal_explanation: Optional[str]
    clip_similarity_score: Optional[float]

    # ── Freshness ─────────────────────────────────────────────────────────
    last_verified_at: Optional[datetime]

    # ── Final output ──────────────────────────────────────────────────────
    output: Optional[FactCheckOutput]


INITIAL_STATE: dict = {
    "memory_results":          None,
    "entity_context":          [],
    "route":                   None,
    "revalidation_needed":     None,
    "retrieval_gate_needed":   None,
    "retrieved_chunks":        [],
    "sub_claims":              [],
    "debate_transcript":       None,
    "source_credibility":      None,
    "cross_modal_flag":        False,
    "cross_modal_explanation": None,
    "clip_similarity_score":   None,
    "last_verified_at":        None,
    "output":                  None,
}
