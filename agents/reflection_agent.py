"""Reflection Agent — maintains per-source, topic-conditioned credibility signals.

READ  (query_source_credibility) — called by query_memory node before verdict synthesis.
WRITE (update_source_credibility) — called by write_memory node after verdict synthesis.
"""
import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    from agents.memory_agent import MemoryAgent

logger = logging.getLogger(__name__)

_DEFAULT_K = 20
_MIN_SAMPLES = 2


def source_id_from_url(source_url: str) -> str:
    domain = urlparse(source_url).netloc or source_url
    return f"src_{domain.replace('.', '_')}"


def credibility_signal(verdict_label: str, confidence_score: int) -> float:
    c = confidence_score / 100.0
    if verdict_label == "supported":
        return c
    if verdict_label == "refuted":
        return 1.0 - c
    return 0.5


def query_source_credibility(claim_text: str, source_url: str, memory: "MemoryAgent",
                               k: int = _DEFAULT_K) -> dict:
    source_id = source_id_from_url(source_url)
    try:
        results = memory.query_source_credibility(claim_text=claim_text, source_id=source_id, k=k)
    except Exception as exc:
        logger.warning("query_source_credibility failed for %s: %s", source_id, exc)
        return {"credibility_mean": None, "bias_mean": None, "bias_std": None, "sample_count": 0}

    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if len(metadatas) < _MIN_SAMPLES:
        return {"credibility_mean": None, "bias_mean": None, "bias_std": None, "sample_count": len(metadatas)}

    weights = [1.0 / (d + 1e-6) for d in distances]
    W = sum(weights)
    cred_mean = sum(w * m["credibility"] for w, m in zip(weights, metadatas)) / W
    bias_mean = sum(w * m["bias"]        for w, m in zip(weights, metadatas)) / W
    bias_var  = sum(w * (m["bias"] - bias_mean) ** 2 for w, m in zip(weights, metadatas)) / W
    bias_std  = math.sqrt(bias_var)

    return {
        "credibility_mean": round(cred_mean, 4),
        "bias_mean":        round(bias_mean, 4),
        "bias_std":         round(bias_std, 4),
        "sample_count":     len(metadatas),
    }


def update_source_credibility(claim_text: str, source_url: str, verdict_id: str,
                               verdict_label: str, confidence_score: int, bias_score: float,
                               memory: "MemoryAgent") -> None:
    source_id   = source_id_from_url(source_url)
    credibility = credibility_signal(verdict_label, confidence_score)
    point_id    = f"sc_{verdict_id}"
    created_at  = datetime.now(timezone.utc).isoformat()
    try:
        memory.add_source_credibility_point(
            point_id=point_id, claim_text=claim_text, topic_text=claim_text,
            source_id=source_id, credibility=credibility, bias=bias_score,
            verdict_label=verdict_label, verdict_id=verdict_id, created_at=created_at,
        )
    except Exception as exc:
        logger.error("update_source_credibility failed for %s: %s", source_id, exc)
