"""Routing functions for the LangGraph conditional edges."""
from config import settings
from orchestration.state import FactCheckState

CACHE_CONFIDENCE_THRESHOLD = 0.80


def router(state: FactCheckState) -> str:
    memory_results = state.get("memory_results")
    if memory_results and memory_results.max_confidence >= CACHE_CONFIDENCE_THRESHOLD:
        return "cache"
    return "live_search"


def freshness_router(state: FactCheckState) -> str:
    if state.get("revalidation_needed") is False:
        return "fresh"
    return "stale"


def retrieval_gate_router(state: FactCheckState) -> str:
    if state.get("retrieval_gate_needed", True):
        return "needed"
    return "skip"


def debate_check(state: FactCheckState) -> str:
    if settings.use_debate:
        output = state.get("output")
        if output and output.confidence_score < settings.debate_confidence_threshold:
            return "debate"
    return "skip"
