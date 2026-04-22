"""
Seed Demo Data Script
=====================
Populates Neo4j with realistic demo entities, claims, verdicts,
credibility snapshots, and predictions so the full dashboard works.

Run once before starting the app:
    python seed_demo_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta, timezone
from agents.memory_agent import get_memory
from id_utils import make_id
from models.credibility import CredibilitySnapshot, Prediction

NOW = datetime.now(timezone.utc)


# ─────────────────────────────────────────────
# Demo entities with realistic credibility arcs
# ─────────────────────────────────────────────

ENTITIES = [
    {
        "entity_id": "ent_trump_001",
        "name": "Trump",
        "entity_type": "person",
        # Declining arc — lots of refuted claims recently
        "snapshots": [0.62, 0.59, 0.55, 0.52, 0.50, 0.47, 0.43, 0.40],
        "claims": [
            {"text": "Trump claimed the 2020 election was stolen.",        "label": "refuted",    "conf": 0.91},
            {"text": "Trump signed the CARES Act stimulus checks.",         "label": "supported",  "conf": 0.95},
            {"text": "Trump was the first president impeached twice.",      "label": "supported",  "conf": 0.97},
            {"text": "Trump was killed by Iran.",                           "label": "refuted",    "conf": 0.98},
            {"text": "Trump imposed tariffs on Chinese goods.",             "label": "supported",  "conf": 0.93},
            {"text": "Trump claimed COVID would disappear like a miracle.", "label": "misleading", "conf": 0.80},
        ],
    },
    {
        "entity_id": "ent_tesla_001",
        "name": "Tesla",
        "entity_type": "organization",
        # Rising arc — mostly supported claims
        "snapshots": [0.58, 0.61, 0.63, 0.66, 0.68, 0.71, 0.74, 0.76],
        "claims": [
            {"text": "Tesla delivered over 1.8 million vehicles in 2023.",  "label": "supported",  "conf": 0.94},
            {"text": "Tesla recalled 2 million vehicles over autopilot.",    "label": "supported",  "conf": 0.96},
            {"text": "Tesla became the most valuable car company ever.",     "label": "supported",  "conf": 0.88},
            {"text": "Tesla's Full Self-Driving is fully autonomous.",       "label": "refuted",    "conf": 0.85},
            {"text": "Tesla has zero quality control issues.",               "label": "refuted",    "conf": 0.90},
        ],
    },
    {
        "entity_id": "ent_who_001",
        "name": "WHO",
        "entity_type": "organization",
        # Stable arc
        "snapshots": [0.72, 0.71, 0.73, 0.70, 0.72, 0.71, 0.73, 0.72],
        "claims": [
            {"text": "WHO declared COVID-19 a pandemic in March 2020.",     "label": "supported",  "conf": 0.99},
            {"text": "WHO recommended masks for general public early on.",   "label": "misleading", "conf": 0.75},
            {"text": "WHO is funded primarily by private donors.",           "label": "misleading", "conf": 0.70},
            {"text": "WHO eradicated smallpox globally.",                    "label": "supported",  "conf": 0.98},
        ],
    },
    {
        "entity_id": "ent_musk_001",
        "name": "Elon Musk",
        "entity_type": "person",
        # Volatile arc
        "snapshots": [0.70, 0.65, 0.72, 0.60, 0.68, 0.55, 0.62, 0.50],
        "claims": [
            {"text": "Elon Musk bought Twitter for $44 billion.",           "label": "supported",  "conf": 0.99},
            {"text": "Elon Musk is the richest person in the world.",       "label": "supported",  "conf": 0.85},
            {"text": "Elon Musk claimed he would put humans on Mars by 2024.", "label": "refuted", "conf": 0.88},
            {"text": "Elon Musk promoted a cryptocurrency that crashed.",    "label": "supported",  "conf": 0.82},
        ],
    },
]


def seed():
    print("Connecting to Neo4j...")
    memory = get_memory()

    print("Initializing schema (constraints + indexes)...")
    memory.init_schema()
    print("  Schema ready.")

    for entity in ENTITIES:
        eid    = entity["entity_id"]
        ename  = entity["name"]
        etype  = entity["entity_type"]
        snaps  = entity["snapshots"]
        claims = entity["claims"]

        print(f"\n── Seeding: {ename} ──")

        # ── 1. Write Entity node ──────────────────────────────────────
        with memory._graph._driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {entity_id: $eid})
                SET e.name               = $name,
                    e.entity_type        = $etype,
                    e.current_credibility = $cred,
                    e.total_claims       = $total,
                    e.accurate_claims    = $accurate,
                    e.first_seen         = $first_seen,
                    e.last_seen          = $last_seen
                """,
                eid        = eid,
                name       = ename,
                etype      = etype,
                cred       = snaps[-1],
                total      = len(claims),
                accurate   = sum(1 for c in claims if c["label"] == "supported"),
                first_seen = (NOW - timedelta(days=30)).isoformat(),
                last_seen  = NOW.isoformat(),
            )
        print(f"  Entity node written.")

        # ── 2. Write Claims + Verdicts ────────────────────────────────
        source_id  = f"src_demo_{eid}"
        article_id = make_id("art_")

        # Write a dummy source + article so foreign keys are satisfied
        with memory._graph._driver.session() as session:
            session.run(
                """
                MERGE (s:Source {source_id: $sid})
                SET s.name = 'Demo Source', s.domain = 'demo.factguard.ai',
                    s.category = 'demo', s.base_credibility = 0.5
                """,
                sid=source_id,
            )
            session.run(
                """
                MERGE (a:Article {article_id: $aid})
                SET a.title = $title, a.url = $url,
                    a.source_id = $sid, a.ingested_at = $now,
                    a.published_at = $now, a.content_hash = $hash,
                    a.body_snippet = $snippet
                """,
                aid     = article_id,
                title   = f"Demo article about {ename}",
                url     = f"https://demo.factguard.ai/{eid}",
                sid     = source_id,
                now     = NOW.isoformat(),
                hash    = make_id("hash_"),
                snippet = f"Demo content for {ename}",
            )

        for i, claim in enumerate(claims):
            claim_id   = make_id("clm_")
            verdict_id = make_id("vrd_")
            claim_time = NOW - timedelta(hours=(len(claims) - i) * 3)

            with memory._graph._driver.session() as session:
                # Write Claim
                session.run(
                    """
                    MERGE (c:Claim {claim_id: $cid})
                    SET c.claim_text  = $text,
                        c.article_id  = $aid,
                        c.extracted_at = $ts,
                        c.status      = 'verified'
                    """,
                    cid  = claim_id,
                    text = claim["text"],
                    aid  = article_id,
                    ts   = claim_time.isoformat(),
                )
                # Write Verdict
                session.run(
                    """
                    MERGE (v:Verdict {verdict_id: $vid})
                    SET v.claim_id          = $cid,
                        v.verdict_label     = $label,
                        v.verdict_confidence = $conf,
                        v.verified_at       = $ts,
                        v.reasoning         = $reason
                    """,
                    vid    = verdict_id,
                    cid    = claim_id,
                    label  = claim["label"],
                    conf   = claim["conf"],
                    ts     = claim_time.isoformat(),
                    reason = f"Demo verdict for: {claim['text'][:60]}",
                )
                # Link: Claim → Verdict
                session.run(
                    """
                    MATCH (c:Claim {claim_id: $cid})
                    MATCH (v:Verdict {verdict_id: $vid})
                    MERGE (c)-[:VERIFIED_AS]->(v)
                    """,
                    cid=claim_id, vid=verdict_id,
                )
                # Link: Entity → Claim
                session.run(
                    """
                    MATCH (e:Entity {entity_id: $eid})
                    MATCH (c:Claim  {claim_id:  $cid})
                    MERGE (e)-[:MENTIONS]->(c)
                    """,
                    eid=eid, cid=claim_id,
                )
                # Link: Article → Claim
                session.run(
                    """
                    MATCH (a:Article {article_id: $aid})
                    MATCH (c:Claim   {claim_id:   $cid})
                    MERGE (a)-[:CONTAINS]->(c)
                    """,
                    aid=article_id, cid=claim_id,
                )

        print(f"  {len(claims)} claims + verdicts written.")

        # ── 3. Write Credibility Snapshots ────────────────────────────
        sentiments = [0.1, -0.1, 0.0, -0.2, 0.1, -0.3, -0.1, -0.2]
        for i, score in enumerate(snaps):
            days_ago  = len(snaps) - i
            snap_time = NOW - timedelta(days=days_ago)
            snapshot  = CredibilitySnapshot(
                snapshot_id       = make_id("snap_"),
                entity_id         = eid,
                credibility_score = score,
                sentiment_score   = sentiments[i % len(sentiments)],
                snapshot_at       = snap_time,
            )
            memory.add_credibility_snapshot(snapshot)

        print(f"  {len(snaps)} credibility snapshots written.")

        # ── 4. Write Predictions ──────────────────────────────────────
        trend_dir = "falling" if snaps[-1] < snaps[0] else "rising" if snaps[-1] > snaps[0] else "stable"

        if trend_dir == "falling":
            pred_text = (
                f"Credibility of '{ename}' is likely to decline further "
                f"(below {snaps[-1] - 0.05:.0%}) within the next 7 days "
                f"based on a sustained declining trend."
            )
            confidence = 0.74
            deadline   = NOW + timedelta(days=7)
        elif trend_dir == "rising":
            pred_text = (
                f"Credibility of '{ename}' is trending upward and likely to "
                f"stabilize or continue rising above {snaps[-1] + 0.03:.0%} within 5 days."
            )
            confidence = 0.68
            deadline   = NOW + timedelta(days=5)
        else:
            pred_text = (
                f"'{ename}' shows a stable trend. No significant credibility "
                f"change (>10%) is expected in the next 7 days."
            )
            confidence = 0.70
            deadline   = NOW + timedelta(days=7)

        prediction = Prediction(
            prediction_id   = make_id("pred_"),
            entity_id       = eid,
            prediction_text = pred_text,
            confidence      = confidence,
            predicted_at    = NOW,
            deadline        = deadline,
        )
        memory.add_prediction(prediction)
        print(f"  1 prediction written ({trend_dir} trend).")

    print("\n" + "="*50)
    print("  Seed complete! 4 entities loaded:")
    for e in ENTITIES:
        print(f"    - {e['name']} ({len(e['snapshots'])} snapshots, {len(e['claims'])} claims)")
    print("\n  Now run:  streamlit run frontend/app.py")
    print("  Then search for: Trump, Tesla, WHO, or Elon Musk")
    print("="*50)

    memory.close()


if __name__ == "__main__":
    seed()
