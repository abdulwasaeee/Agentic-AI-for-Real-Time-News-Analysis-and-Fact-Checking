"""Retrieval reranking: Reciprocal Rank Fusion + optional cross-encoder."""
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

_RRF_K = 60


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], id_key: str = "claim_id") -> list[dict]:
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            key = item[id_key]
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            if key not in items:
                items[key] = item
    merged = sorted(items.values(), key=lambda x: scores[x[id_key]], reverse=True)
    for item in merged:
        item["rrf_score"] = scores[item[id_key]]
    return merged


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def cross_encoder_rerank(query: str, candidates: list[dict], model_name: str,
                          top_k: int, text_key: str = "claim_text") -> list[dict]:
    if not candidates:
        return candidates
    try:
        model = _load_cross_encoder(model_name)
        pairs = [(query, c[text_key]) for c in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        result = []
        for score, item in ranked[:top_k]:
            item = dict(item)
            item["cross_encoder_score"] = float(score)
            result.append(item)
        return result
    except Exception as e:
        logger.error("Cross-encoder reranking failed: %s — returning original order", e)
        return candidates[:top_k]


def rerank_candidates(query: str, vector_results: list[dict], graph_results: list[dict],
                       use_cross_encoder: bool, cross_encoder_model: str, top_k: int) -> list[dict]:
    lists_to_merge = [l for l in [vector_results, graph_results] if l]
    if not lists_to_merge:
        return []
    if len(lists_to_merge) == 1:
        merged = lists_to_merge[0][:top_k]
    else:
        merged = reciprocal_rank_fusion(lists_to_merge)
    if use_cross_encoder:
        return cross_encoder_rerank(query, merged, cross_encoder_model, top_k)
    return merged[:top_k]
