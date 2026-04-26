"""Memory Agent facade — the single public API for all agents.

All agents import MemoryAgent and call its methods. They never interact
with VectorStore or GraphStore directly.

Usage:
    from agents.memory_agent import MemoryAgent, get_memory, close_memory
    from config import settings

    memory = get_memory()   # singleton — call at startup, reuse everywhere
"""

import logging
from datetime import datetime
from typing import Optional

from config import Settings, settings as _default_settings
from memory.embeddings import EmbeddingHelper
from memory.graph_db import GraphStore
from memory.vector_db import VectorStore
from models.caption import ImageCaption
from models.credibility import CredibilitySnapshot, Prediction
from models.pipeline import PreprocessingOutput
from models.verdict import Verdict

logger = logging.getLogger(__name__)


class MemoryAgent:
    def __init__(self, cfg: Settings = _default_settings):
        self._embeddings = EmbeddingHelper(
            api_key=cfg.openai_api_key,
            model=cfg.embedding_model,
        )
        self._vector = VectorStore(
            api_key=cfg.chroma_api_key,
            tenant=cfg.chroma_tenant,
            database=cfg.chroma_database,
            host=cfg.chroma_host,
            port=cfg.chroma_port,
        )
        self._graph = GraphStore(
            uri=cfg.neo4j_uri,
            user=cfg.neo4j_user,
            password=cfg.neo4j_password,
        )
        logger.info("MemoryAgent initialized")

    def close(self) -> None:
        self._graph.close()

    def init_schema(self) -> None:
        self._graph.init_schema()

    # ── Primary Write: Ingest Preprocessed Output ───────────────────────

    def ingest_preprocessed(self, output: PreprocessingOutput) -> bool:
        """Ingest a preprocessed article into both databases. Returns False if duplicate."""
        if self._vector.check_content_hash_exists(output.article.content_hash):
            logger.info("Duplicate article skipped: %s", output.article.article_id)
            return False

        texts_to_embed, text_labels = [], []
        texts_to_embed.append(output.article.body_snippet)
        text_labels.append(("article", 0))
        for i, claim in enumerate(output.claims):
            texts_to_embed.append(claim.claim_text)
            text_labels.append(("claim", i))
        if output.image_caption:
            texts_to_embed.append(output.image_caption.vlm_caption)
            text_labels.append(("caption", 0))

        embeddings = self._embeddings.embed_batch(texts_to_embed)

        article_embedding = None
        claim_embeddings: list[list[float]] = []
        caption_embedding = None
        for idx, (label_type, _) in enumerate(text_labels):
            if label_type == "article":
                article_embedding = embeddings[idx]
            elif label_type == "claim":
                claim_embeddings.append(embeddings[idx])
            elif label_type == "caption":
                caption_embedding = embeddings[idx]

        self._graph.merge_source(
            source_id=output.source.source_id, name=output.source.name,
            domain=output.source.domain, category=output.source.category,
            base_credibility=output.source.base_credibility,
        )
        self._graph.create_article(
            article_id=output.article.article_id, title=output.article.title,
            url=output.article.url, source_id=output.article.source_id,
            published_at=output.article.published_at, ingested_at=output.article.ingested_at,
            content_hash=output.article.content_hash,
        )
        claims_data = [
            {
                "claim_id": c.claim_id, "claim_text": c.claim_text,
                "claim_type": c.claim_type or "", "extracted_at": c.extracted_at.isoformat(),
                "status": c.status,
                "entities": [{"entity_id": e.entity_id, "name": e.name,
                               "entity_type": e.entity_type, "sentiment": e.sentiment}
                              for e in c.entities],
            }
            for c in output.claims
        ]
        self._graph.create_claims_with_entities(claims_data, output.article.article_id)
        if output.image_caption:
            self._graph.create_image_caption(
                caption_id=output.image_caption.caption_id,
                article_id=output.image_caption.article_id,
                image_url=output.image_caption.image_url,
                vlm_caption=output.image_caption.vlm_caption,
            )

        self._vector.upsert_article(
            article_id=output.article.article_id, embedding=article_embedding,
            document=output.article.body_snippet, source_id=output.article.source_id,
            domain=output.source.category, content_hash=output.article.content_hash,
            published_at=output.article.published_at.isoformat(),
        )
        for i, claim in enumerate(output.claims):
            self._vector.upsert_claim(
                claim_id=claim.claim_id, embedding=claim_embeddings[i],
                document=claim.claim_text, article_id=claim.article_id,
                source_id=output.article.source_id, status=claim.status,
                extracted_at=claim.extracted_at.isoformat(),
            )
        if output.image_caption and caption_embedding:
            self._vector.upsert_caption(
                caption_id=output.image_caption.caption_id, embedding=caption_embedding,
                document=output.image_caption.vlm_caption,
                article_id=output.image_caption.article_id,
                image_url=output.image_caption.image_url,
            )

        logger.info("Ingested article %s with %d claims", output.article.article_id, len(output.claims))
        return True

    # ── Teammate Write Methods ──────────────────────────────────────────

    def add_verdict(self, verdict: Verdict) -> None:
        claim_results = self._vector.get_claims_by_ids([verdict.claim_id])
        claim_text = claim_results["documents"][0] if claim_results["documents"] else ""
        embedding = self._embeddings.embed(f"{claim_text} [verdict: {verdict.label}]")
        self._vector.upsert_verdict(
            verdict_id=verdict.verdict_id, embedding=embedding,
            document=verdict.evidence_summary, claim_id=verdict.claim_id,
            label=verdict.label, confidence=verdict.confidence,
            bias_score=verdict.bias_score, image_mismatch=verdict.image_mismatch,
            verified_at=verdict.verified_at.isoformat(),
        )
        self._graph.create_verdict(
            verdict_id=verdict.verdict_id, claim_id=verdict.claim_id,
            label=verdict.label, confidence=verdict.confidence,
            evidence_summary=verdict.evidence_summary, bias_score=verdict.bias_score,
            image_mismatch=verdict.image_mismatch, verified_at=verdict.verified_at,
        )

    def ensure_entity_exists(self, name: str) -> str:
        """Create entity node if it doesn't exist. Returns entity_id."""
        return self._graph.ensure_entity_exists(name)

    def backfill_mentions_for_entity(self, name: str) -> int:
        """Link existing claims that mention this entity by name. Returns link count."""
        entity_id = self._graph.ensure_entity_exists(name)
        return self._graph.backfill_mentions_for_entity(entity_id, name)

    def auto_store_claim_with_entities(self, claim_id: str, claim_text: str,
                                       article_id: str, entity_dicts: list[dict]) -> None:
        """Store a claim in ChromaDB + create Claim/Entity nodes in Neo4j.

        Called automatically after every fact-check so entities are tracked
        even when claims come directly from the frontend (no preprocessing pipeline).
        """
        from datetime import timezone
        try:
            embedding = self._embeddings.embed(claim_text)
            self._vector.upsert_claim(
                claim_id=claim_id, embedding=embedding, document=claim_text,
                article_id=article_id, source_id="src_frontend",
                status="verified",
                extracted_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            logger.warning("ChromaDB claim upsert skipped: %s", e)
        self._graph.auto_store_claim_with_entities(claim_id, claim_text, article_id, entity_dicts)

    def add_credibility_snapshot(self, snapshot: CredibilitySnapshot) -> None:
        self._graph.create_snapshot(
            snapshot_id=snapshot.snapshot_id, entity_id=snapshot.entity_id,
            credibility_score=snapshot.credibility_score,
            sentiment_score=snapshot.sentiment_score,
            snapshot_at=snapshot.snapshot_at,
        )

    def update_entity(self, entity_id: str, **updates) -> None:
        self._graph.update_entity(entity_id, updates)

    def add_prediction(self, prediction: Prediction) -> None:
        self._graph.create_prediction(
            prediction_id=prediction.prediction_id, entity_id=prediction.entity_id,
            prediction_text=prediction.prediction_text, confidence=prediction.confidence,
            predicted_at=prediction.predicted_at, deadline=prediction.deadline,
            outcome=prediction.outcome,
        )

    def resolve_prediction(self, prediction_id: str, outcome: str) -> None:
        self._graph.resolve_prediction(prediction_id, outcome)

    # ── Reflection Agent write/read (Task 2) ────────────────────────────

    def add_source_credibility_point(self, point_id: str, claim_text: str, topic_text: str,
                                      source_id: str, credibility: float, bias: float,
                                      verdict_label: str, verdict_id: str, created_at: str) -> None:
        """Append one (source, topic, credibility, bias) observation."""
        embedding = self._embeddings.embed(topic_text)
        self._vector.upsert_source_credibility_point(
            point_id=point_id, embedding=embedding, document=topic_text,
            source_id=source_id, credibility=credibility, bias=bias,
            verdict_label=verdict_label, verdict_id=verdict_id, created_at=created_at,
        )

    def query_source_credibility(self, claim_text: str, source_id: str, k: int = 20) -> dict:
        """Retrieve k nearest (source, topic) credibility observations."""
        embedding = self._embeddings.embed(claim_text)
        return self._vector.query_source_credibility(embedding, source_id=source_id, k=k)

    # ── Query Methods ───────────────────────────────────────────────────

    def search_similar_claims(self, text: str, top_k: int = 5) -> dict:
        embedding = self._embeddings.embed(text)
        return self._vector.search_similar_claims(embedding, top_k)

    def check_duplicate(self, content_hash: str) -> bool:
        return self._vector.check_content_hash_exists(content_hash)

    def get_claims_by_ids(self, ids: list[str]) -> dict:
        return self._vector.get_claims_by_ids(ids)

    def get_caption_by_article(self, article_id: str) -> dict:
        return self._vector.get_caption_by_article(article_id)

    def get_verdict_by_claim(self, claim_id: str) -> dict:
        return self._vector.get_verdict_by_claim(claim_id)

    def get_entity_context(self, claim_id: str) -> list[dict]:
        return self._graph.get_entity_context(claim_id)

    def get_entity_claims(self, entity_id: str, since: Optional[datetime] = None) -> list[dict]:
        return self._graph.get_entity_claims(entity_id, since)

    def get_entity_snapshots(self, entity_id: str, limit: int = 20) -> list[dict]:
        return self._graph.get_entity_snapshots(entity_id, limit)

    def get_source_credibility(self, article_id: str) -> Optional[float]:
        return self._graph.get_source_credibility(article_id)

    def get_trending_entities(self, since: datetime, limit: int = 10) -> list[dict]:
        return self._graph.get_trending_entities(since, limit)

    def get_expired_predictions(self) -> list[dict]:
        return self._graph.get_expired_predictions()

    # ── Extended queries (for Entity Tracker, Prediction Agent, frontend) ─

    def get_entity_by_name(self, name: str) -> Optional[dict]:
        return self._graph.get_entity_by_name(name)

    def get_claim_count_for_entity(self, entity_id: str, since: datetime) -> int:
        return self._graph.get_claim_count_for_entity(entity_id, since)

    def get_predictions_for_entity(self, entity_id: str, include_resolved: bool = False) -> list[dict]:
        return self._graph.get_predictions_for_entity(entity_id, include_resolved)

    def update_verdict_with_feedback(self, verdict_id: str, correct_label: str,
                                      correct_confidence: float, feedback_note: str = "") -> None:
        """Override a verdict with human-corrected values (human-in-the-loop)."""
        self._graph.update_verdict_with_feedback(verdict_id, correct_label, correct_confidence, feedback_note)
        try:
            self._vector.update_verdict_metadata(verdict_id, correct_label, correct_confidence)
        except Exception as e:
            logger.warning("ChromaDB verdict patch skipped: %s", e)

    def find_human_verdict_for_claim(self, claim_text: str, threshold: float = 0.70) -> dict | None:
        """Return a human-corrected verdict for a semantically similar claim, or None."""
        try:
            embedding = self._embeddings.embed(claim_text)
            return self._vector.find_human_verdict_by_embedding(embedding, threshold=threshold)
        except Exception as e:
            logger.warning("Human verdict lookup failed: %s", e)
            return None

    # ── GraphRAG methods (for Task 2 use_graph_rag=True path) ───────────

    def get_entity_ids_for_claims(self, claim_ids: list[str]) -> list[dict]:
        return self._graph.get_entity_ids_for_claims(claim_ids)

    def get_graph_claims_for_entities(self, entity_ids: list[str]) -> list[dict]:
        return self._graph.get_graph_claims_for_entities(entity_ids)


# ── Process-level singleton ─────────────────────────────────────────────

_memory: Optional[MemoryAgent] = None


def get_memory() -> MemoryAgent:
    """Return (or create) the process-level MemoryAgent singleton."""
    global _memory
    if _memory is None:
        logger.info("Initialising MemoryAgent singleton")
        _memory = MemoryAgent(_default_settings)
    return _memory


def close_memory() -> None:
    """Close the MemoryAgent and its Neo4j driver. Call at process shutdown."""
    global _memory
    if _memory is not None:
        _memory.close()
        _memory = None
        logger.info("MemoryAgent closed")
