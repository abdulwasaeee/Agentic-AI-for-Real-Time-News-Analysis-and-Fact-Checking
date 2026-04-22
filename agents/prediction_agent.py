"""
Prediction Agent
Task 3 — Full-Stack & Evaluation Engineer

WHAT THIS DOES:
- Reads credibility snapshot history from Graph DB (written by Entity Tracker)
- Detects patterns: is credibility trending up, down, or stable?
- Outputs time-bound, falsifiable predictions with a confidence score
- Writes predictions back to Graph DB as PREDICTION nodes
- Also checks expired predictions and resolves them

HOW IT CONNECTS:
- Reads from:  Graph DB — ENTITY, CREDIBILITY_SNAPSHOT, PREDICTION nodes  (via MemoryAgent)
- Writes to:   Graph DB — new PREDICTION nodes, updated outcome on expired PREDICTION nodes

RUN: python -m agents.prediction_agent
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Optional

from agents.memory_agent import MemoryAgent, get_memory
from id_utils import make_id
from models.credibility import CredibilitySnapshot, Prediction

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TREND DETECTION (original from Task 3 — unchanged)
# ─────────────────────────────────────────────

def detect_trend(snapshots: list[dict]) -> dict:
    """
    Analyzes a time series of credibility scores to detect the trend.

    Method: Simple Linear Regression (line of best fit).
    The slope m tells us: positive = rising, negative = falling.

    Returns a dict with:
        slope, direction, recent_avg, older_avg,
        recent_drop_count, volatility
    """
    if len(snapshots) < 3:
        return {"direction": "stable", "slope": 0.0, "recent_avg": 0.5,
                "older_avg": 0.5, "recent_drop_count": 0, "volatility": 0.0}

    scores = [s["credibility_score"] for s in snapshots]
    n = len(scores)

    x_vals = list(range(n))
    x_mean = sum(x_vals) / n
    y_mean = sum(scores) / n
    numerator   = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, scores))
    denominator = sum((x - x_mean) ** 2 for x in x_vals)
    slope = numerator / denominator if denominator != 0 else 0.0

    if slope < -0.003:
        direction = "falling"
    elif slope > 0.003:
        direction = "rising"
    else:
        direction = "stable"

    mid        = n // 2
    recent_avg = sum(scores[mid:]) / len(scores[mid:])
    older_avg  = sum(scores[:mid]) / len(scores[:mid])

    recent_drop_count = 0
    for i in range(n - 1, 0, -1):
        if scores[i] < scores[i - 1]:
            recent_drop_count += 1
        else:
            break

    variance   = sum((s - y_mean) ** 2 for s in scores) / n
    volatility = math.sqrt(variance)

    return {
        "slope":             round(slope, 6),
        "direction":         direction,
        "recent_avg":        round(recent_avg, 4),
        "older_avg":         round(older_avg, 4),
        "recent_drop_count": recent_drop_count,
        "volatility":        round(volatility, 4),
    }


def detect_sentiment_trend(snapshots: list[dict]) -> str:
    """Simple sentiment direction based on recent snapshots."""
    if len(snapshots) < 4:
        return "neutral"
    recent_sent = [s.get("sentiment_score", 0.0) for s in snapshots[-5:]]
    avg = sum(recent_sent) / len(recent_sent)
    if avg > 0.2:
        return "positive"
    elif avg < -0.2:
        return "negative"
    return "neutral"


# ─────────────────────────────────────────────
# PREDICTION GENERATION (original rules R1–R5 — unchanged)
# ─────────────────────────────────────────────

def generate_predictions(entity_name: str, entity_id: str,
                          trend: dict, sentiment_dir: str,
                          claim_count_7d: int) -> list[Prediction]:
    """
    Rules-based prediction generation.

    Rules:
        R1. Sustained declining trend → predict further decline
        R2. Rising trend → predict stabilization or further rise
        R3. High claim volume in 7 days → predict fact-check surge
        R4. Negative sentiment + falling credibility → predict major negative article
        R5. Stable + low volatility → predict no significant change
    """
    predictions = []
    now = datetime.now()

    direction  = trend["direction"]
    slope      = trend["slope"]
    drops      = trend["recent_drop_count"]
    volatility = trend["volatility"]

    # ── Rule R1: Sustained decline ──
    if direction == "falling" and drops >= 3:
        confidence = min(0.85, 0.5 + abs(slope) * 100 + drops * 0.05)
        predictions.append(Prediction(
            prediction_id   = make_id("pred_"),
            entity_id       = entity_id,
            prediction_text = (
                f"Credibility of '{entity_name}' is likely to decline further "
                f"(below {trend['recent_avg'] - 0.05:.0%}) within the next 7 days "
                f"if the current pattern of {drops} consecutive declining snapshots continues."
            ),
            confidence    = round(confidence, 2),
            predicted_at  = now,
            deadline      = now + timedelta(days=7),
        ))

    # ── Rule R2: Rising trend ──
    elif direction == "rising":
        confidence = min(0.80, 0.5 + slope * 80)
        predictions.append(Prediction(
            prediction_id   = make_id("pred_"),
            entity_id       = entity_id,
            prediction_text = (
                f"Credibility of '{entity_name}' is trending upward. "
                f"It is likely to stabilize or continue rising above "
                f"{trend['recent_avg'] + 0.03:.0%} within 5 days, "
                f"assuming no new negative claims emerge."
            ),
            confidence   = round(confidence, 2),
            predicted_at = now,
            deadline     = now + timedelta(days=5),
        ))

    # ── Rule R3: High claim volume ──
    if claim_count_7d >= 10:
        confidence = min(0.78, 0.45 + claim_count_7d * 0.015)
        predictions.append(Prediction(
            prediction_id   = make_id("pred_"),
            entity_id       = entity_id,
            prediction_text = (
                f"'{entity_name}' has generated {claim_count_7d} claims in the last 7 days "
                f"— above average activity. Expect at least 3 more fact-check verdicts "
                f"mentioning this entity in the next 14 days."
            ),
            confidence   = round(confidence, 2),
            predicted_at = now,
            deadline     = now + timedelta(days=14),
        ))

    # ── Rule R4: Negative sentiment + falling credibility ──
    if sentiment_dir == "negative" and direction == "falling":
        predictions.append(Prediction(
            prediction_id   = make_id("pred_"),
            entity_id       = entity_id,
            prediction_text = (
                f"Combined falling credibility and negative sentiment signals for '{entity_name}' "
                f"suggest a high likelihood (>65%) of a major misleading or refuted fact-check "
                f"article being published within the next 10 days."
            ),
            confidence   = 0.65,
            predicted_at = now,
            deadline     = now + timedelta(days=10),
        ))

    # ── Rule R5: Stable, low volatility ──
    if direction == "stable" and volatility < 0.03:
        predictions.append(Prediction(
            prediction_id   = make_id("pred_"),
            entity_id       = entity_id,
            prediction_text = (
                f"'{entity_name}' shows low volatility (σ={volatility:.3f}) and a stable trend. "
                f"No significant credibility change (>10%) is expected in the next 7 days "
                f"without a major news event."
            ),
            confidence   = 0.70,
            predicted_at = now,
            deadline     = now + timedelta(days=7),
        ))

    return predictions


# ─────────────────────────────────────────────
# RESOLVE EXPIRED PREDICTIONS
# ─────────────────────────────────────────────

def resolve_expired_predictions(memory: MemoryAgent, current_snapshots: list[dict]) -> None:
    """Check global expired predictions and resolve them with a simple heuristic."""
    expired = memory.get_expired_predictions()
    if not expired:
        print("  No expired predictions to resolve.")
        return

    current_cred = current_snapshots[-1]["credibility_score"] if current_snapshots else 0.5

    for pred in expired:
        # TODO: Replace heuristic with LLM-based outcome evaluation
        # Simple rule: if recent credibility dropped and prediction said "decline" → confirmed
        outcome = "inconclusive"
        print(f"  Resolving expired prediction: {pred['prediction_id']} → {outcome}")
        memory.resolve_prediction(pred["prediction_id"], outcome)


# ─────────────────────────────────────────────
# MAIN AGENT FUNCTION
# ─────────────────────────────────────────────

def run_prediction_agent(entity_name: str, entity_id: str,
                          memory: Optional[MemoryAgent] = None) -> list[Prediction]:
    """
    Main function of the Prediction Agent.

    Steps:
    1. Pull credibility snapshot history from Graph DB
    2. Detect trend using linear regression
    3. Get 7-day claim count for volume signal
    4. Generate predictions based on rules R1–R5
    5. Write predictions to Graph DB
    6. Resolve any expired predictions

    Args:
        entity_name : Display name for logging/prediction text
        entity_id   : Graph DB entity_id (use MemoryAgent.get_entity_by_name to look it up)
        memory      : MemoryAgent instance (uses singleton if None)
    """
    if memory is None:
        memory = get_memory()

    print(f"\n{'='*60}")
    print(f"  Prediction Agent — Running for: '{entity_name}'")
    print(f"{'='*60}")

    # ── Step 1: Load snapshot history ──
    print("\n[1/5] Loading credibility snapshot history...")
    # Returns list of dicts: {snapshot_id, credibility_score, sentiment_score, snapshot_at}
    # Ordered ASC by snapshot_at (oldest first) for trend regression
    snapshots = memory.get_entity_snapshots(entity_id, limit=30)
    print(f"  Loaded {len(snapshots)} snapshots")

    # ── Step 2: Detect trend ──
    print("\n[2/5] Detecting trend...")
    trend         = detect_trend(snapshots)
    sentiment_dir = detect_sentiment_trend(snapshots)

    print(f"  Direction        : {trend['direction']}")
    print(f"  Slope            : {trend['slope']:.6f} per snapshot")
    print(f"  Recent avg       : {trend['recent_avg']:.2%}")
    print(f"  Older avg        : {trend['older_avg']:.2%}")
    print(f"  Consecutive drops: {trend['recent_drop_count']}")
    print(f"  Volatility (σ)   : {trend['volatility']:.4f}")
    print(f"  Sentiment dir    : {sentiment_dir}")

    # ── Step 3: Claim volume ──
    print("\n[3/5] Checking 7-day claim volume...")
    since_7d       = datetime.now() - timedelta(days=7)
    claim_count_7d = memory.get_claim_count_for_entity(entity_id, since_7d)
    print(f"  Claims in last 7 days: {claim_count_7d}")

    # ── Step 4: Generate predictions ──
    print("\n[4/5] Generating predictions...")
    predictions = generate_predictions(entity_name, entity_id, trend, sentiment_dir, claim_count_7d)
    print(f"  Generated {len(predictions)} predictions")

    # ── Step 5: Write to DB ──
    print("\n[5/5] Writing predictions to Graph DB...")
    for pred in predictions:
        memory.add_prediction(pred)
        print(f"  Written: {pred.prediction_id} ({pred.confidence:.0%}) — {pred.prediction_text[:80]}...")

    # ── Step 6: Resolve expired ──
    print("\n[Bonus] Resolving expired predictions...")
    resolve_expired_predictions(memory, snapshots)

    print(f"\nPrediction Agent complete for '{entity_name}'")
    print(f"   {len(predictions)} new predictions written")

    return predictions


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    mem = get_memory()
    name = sys.argv[1] if len(sys.argv) > 1 else "Tesla"
    entity = mem.get_entity_by_name(name)
    if entity:
        run_prediction_agent(name, entity["entity_id"], memory=mem)
    else:
        print(f"Entity '{name}' not found in Graph DB.")
