"""
Entity Tracker Agent
Task 3 — Full-Stack & Evaluation Engineer

WHAT THIS DOES:
- Reads from Graph DB: all claims mentioning a specific entity + their verdicts
- Computes a credibility score and sentiment score for a time window
- Writes a new CredibilitySnapshot node back to Graph DB
- Runs periodically (e.g., every hour via cron or manual trigger)

HOW IT CONNECTS:
- Reads from:  Graph DB (Neo4j) — CLAIM, VERDICT, MENTIONS edges  (via MemoryAgent)
- Writes to:   Graph DB — new CREDIBILITY_SNAPSHOT node, updated ENTITY fields

RUN: python -m agents.entity_tracker
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from agents.memory_agent import MemoryAgent, get_memory
from id_utils import make_id
from models.credibility import CredibilitySnapshot

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SCORING LOGIC (original from Task 3 — unchanged)
# ─────────────────────────────────────────────

def compute_credibility_score(claims: list[dict]) -> float:
    """
    Credibility score = weighted accuracy of claims in this window.

    Formula:
        For each claim:
            "supported"  → contributes +1.0 * confidence
            "misleading" → contributes +0.4 * confidence  (partial credit)
            "refuted"    → contributes  0.0

        credibility = sum(positive_contributions) / sum(all_weights)

    Result is always in [0, 1].
    """
    if not claims:
        return 0.5

    total_weight = sum(c["verdict_confidence"] for c in claims if c.get("verdict_confidence"))
    positive_sum = 0.0

    for c in claims:
        conf  = c.get("verdict_confidence") or 0.0
        label = c.get("verdict_label", "")
        if label == "supported":
            positive_sum += 1.0 * conf
        elif label == "misleading":
            positive_sum += 0.4 * conf
        # "refuted" contributes 0

    credibility = positive_sum / total_weight if total_weight > 0 else 0.5
    return round(credibility, 4)


def compute_sentiment_score(claims: list[dict]) -> float:
    """
    Sentiment score in [-1, +1].

    Maps "positive" → +1, "neutral" → 0, "negative" → -1.
    Takes a simple average across all claims in the window.
    """
    if not claims:
        return 0.0

    sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    scores = [sentiment_map.get(c.get("sentiment", "neutral"), 0.0) for c in claims]
    return round(sum(scores) / len(scores), 4)


# ─────────────────────────────────────────────
# MAIN AGENT FUNCTION
# ─────────────────────────────────────────────

def run_entity_tracker(entity_name: str, window_hours: int = 24,
                       memory: Optional[MemoryAgent] = None) -> Optional[CredibilitySnapshot]:
    """
    Main function of the Entity Tracker Agent.

    Steps:
    1. Find the entity in Graph DB by name
    2. Pull all verified claims mentioning it (within the time window)
    3. Compute credibility score and sentiment score
    4. Write a new CredibilitySnapshot node to Graph DB
    5. Update the Entity node's aggregate fields

    Args:
        entity_name  : The name of the entity to track (e.g., "Tesla")
        window_hours : How far back to look for claims (default: last 24 hours)
        memory       : MemoryAgent instance (uses singleton if None)
    """
    if memory is None:
        memory = get_memory()

    print(f"\n{'='*60}")
    print(f"  Entity Tracker Agent — Running for: '{entity_name}'")
    print(f"  Window: last {window_hours} hours")
    print(f"{'='*60}")

    # ── Step 1: Find the entity ──
    print("\n[1/5] Looking up entity in Graph DB...")
    entity_dict = memory.get_entity_by_name(entity_name)
    if entity_dict is None:
        print(f"  ERROR: Entity '{entity_name}' not found in Graph DB. Skipping.")
        return None

    entity_id          = entity_dict["entity_id"]
    current_credibility = entity_dict.get("current_credibility", 0.5)
    total_claims        = entity_dict.get("total_claims") or 0
    accurate_claims     = entity_dict.get("accurate_claims") or 0
    print(f"  Found: {entity_dict['name']} (id: {entity_id})")

    # ── Step 2: Pull recent claims ──
    since = datetime.now() - timedelta(hours=window_hours)
    print(f"\n[2/5] Querying claims since {since.strftime('%Y-%m-%d %H:%M')}...")
    # Returns: [{claim_id, claim_text, verdict_label, verdict_confidence, sentiment}, ...]
    claims = memory.get_entity_claims(entity_id, since=since)
    print(f"  Found {len(claims)} verified claims in this window")

    if not claims:
        print("  No claims in this window. No snapshot will be written.")
        return None

    label_counts = {"supported": 0, "refuted": 0, "misleading": 0}
    for c in claims:
        label = c.get("verdict_label", "")
        if label in label_counts:
            label_counts[label] += 1
    print(f"  Breakdown: {label_counts}")

    # ── Step 3: Compute scores ──
    print("\n[3/5] Computing scores...")
    credibility_score = compute_credibility_score(claims)
    sentiment_score   = compute_sentiment_score(claims)
    print(f"  Credibility score : {credibility_score:.4f}")
    print(f"  Sentiment score   : {sentiment_score:.4f}")

    drift = credibility_score - current_credibility
    if abs(drift) > 0.1:
        direction = "DROP" if drift < 0 else "RISE"
        print(f"  ALERT: Significant credibility {direction} detected! "
              f"({current_credibility:.2f} → {credibility_score:.2f})")

    # ── Step 4: Write snapshot ──
    print("\n[4/5] Writing CredibilitySnapshot to Graph DB...")
    snapshot = CredibilitySnapshot(
        snapshot_id       = make_id("snap_"),
        entity_id         = entity_id,
        credibility_score = credibility_score,
        sentiment_score   = sentiment_score,
        snapshot_at       = datetime.now(),
    )
    memory.add_credibility_snapshot(snapshot)
    print(f"  Snapshot {snapshot.snapshot_id} written.")

    # ── Step 5: Update entity aggregates ──
    print("\n[5/5] Updating Entity aggregate fields in Graph DB...")
    new_total    = total_claims + len(claims)
    new_accurate = accurate_claims + label_counts["supported"]
    memory.update_entity(
        entity_id,
        total_claims        = new_total,
        accurate_claims     = new_accurate,
        current_credibility = credibility_score,
        last_seen           = datetime.now().isoformat(),
    )
    print(f"  total_claims={new_total}, accurate_claims={new_accurate}, "
          f"current_credibility={credibility_score:.4f}")

    print(f"\nEntity Tracker complete for '{entity_name}'")
    print(f"   Snapshot ID: {snapshot.snapshot_id}")
    print(f"   Final credibility: {credibility_score:.2%}")

    return snapshot


def run_batch_tracker(entity_names: list[str], window_hours: int = 24,
                      memory: Optional[MemoryAgent] = None) -> dict:
    """Run the Entity Tracker for a list of entities."""
    if memory is None:
        memory = get_memory()

    print(f"\n{'#'*60}")
    print(f"  BATCH TRACKER — {len(entity_names)} entities")
    print(f"{'#'*60}")

    results = {}
    for name in entity_names:
        snapshot = run_entity_tracker(name, window_hours, memory)
        results[name] = snapshot

    print(f"\n{'#'*60}")
    print(f"  BATCH COMPLETE")
    for name, snap in results.items():
        if snap:
            print(f"  OK {name}: credibility={snap.credibility_score:.2%}")
        else:
            print(f"  SKIP {name}: no data")
    print(f"{'#'*60}\n")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_entity_tracker("Tesla", window_hours=24)
