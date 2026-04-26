"""Quick Neo4j diagnostic — run this to see exactly what's stored."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timezone
from agents.memory_agent import get_memory
from id_utils import make_id

memory = get_memory()
driver = memory._graph._driver

# ── 1. Node counts ────────────────────────────────────────────────────────────
print("\n── Node counts ──────────────────────────────")
with driver.session() as s:
    for label in ["Entity", "Claim", "Verdict", "CredibilitySnapshot", "Prediction", "Article", "Source"]:
        n = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
        print(f"  {label:25s}: {n}")

# ── 2. Entities ───────────────────────────────────────────────────────────────
print("\n── Entities in DB ───────────────────────────")
with driver.session() as s:
    rows = list(s.run("MATCH (e:Entity) RETURN e.name AS name, e.entity_id AS eid, e.current_credibility AS cred LIMIT 20"))
    if rows:
        for r in rows:
            print(f"  name={r['name']!r:20s}  id={r['eid']!r:25s}  cred={r['cred']}")
    else:
        print("  NONE")

# ── 3. Recent Claims ──────────────────────────────────────────────────────────
print("\n── Recent Claims (last 5) ───────────────────")
with driver.session() as s:
    rows = list(s.run("""
        MATCH (c:Claim)
        RETURN c.claim_id AS cid, c.claim_text AS text, c.status AS status,
               c.extracted_at AS ts
        ORDER BY c.extracted_at DESC LIMIT 5
    """))
    if rows:
        for r in rows:
            print(f"  [{r['status']}] {str(r['text'])[:70]!r}  @ {r['ts']}")
    else:
        print("  NONE")

# ── 4. Recent Verdicts ────────────────────────────────────────────────────────
print("\n── Recent Verdicts (last 5) ─────────────────")
with driver.session() as s:
    rows = list(s.run("""
        MATCH (v:Verdict)
        RETURN v.verdict_id AS vid,
               coalesce(v.label, v.verdict_label, 'UNKNOWN') AS label,
               coalesce(v.confidence, v.verdict_confidence, 0) AS conf,
               v.verified_at AS ts
        ORDER BY v.verified_at DESC LIMIT 5
    """))
    if rows:
        for r in rows:
            print(f"  {str(r['label']):12s}  conf={r['conf']}  @ {r['ts']}")
    else:
        print("  NONE")

# ── 5. Claim→Entity links ─────────────────────────────────────────────────────
print("\n── Claim→Entity links ───────────────────────")
with driver.session() as s:
    rows = list(s.run("""
        MATCH (c:Claim)-[:MENTIONS]->(e:Entity)
        RETURN c.claim_id AS cid, e.name AS ename LIMIT 10
    """))
    if rows:
        for r in rows:
            print(f"  Claim {r['cid']} → Entity {r['ename']!r}")
    else:
        print("  NONE — no Claim→Entity MENTIONS links")

# ── 6. Claim→Verdict links ────────────────────────────────────────────────────
print("\n── Claim→Verdict links ──────────────────────")
with driver.session() as s:
    rows = list(s.run("""
        MATCH (c:Claim)-[:VERIFIED_AS]->(v:Verdict)
        RETURN c.claim_id AS cid,
               coalesce(v.label, v.verdict_label, 'UNKNOWN') AS label LIMIT 10
    """))
    if rows:
        for r in rows:
            print(f"  Claim {r['cid']} → Verdict {r['label']!r}")
    else:
        print("  NONE — no Claim→Verdict links")

# ── 7. Direct write test ──────────────────────────────────────────────────────
print("\n── Direct write test ────────────────────────")
try:
    test_eid = "ent_test_apple"
    with driver.session() as s:
        s.run("""
            MERGE (e:Entity {entity_id: $eid})
            SET e.name='Apple', e.entity_type='organization',
                e.current_credibility=0.5, e.total_claims=0,
                e.accurate_claims=0,
                e.first_seen=datetime(), e.last_seen=datetime()
        """, eid=test_eid)
    print("  Write OK — test Entity 'Apple' merged")

    # Verify it's there
    with driver.session() as s:
        r = s.run("MATCH (e:Entity {entity_id: $eid}) RETURN e.name AS n", eid=test_eid).single()
        print(f"  Read back: name={r['n']!r}" if r else "  ERROR: could not read back!")
except Exception as e:
    print(f"  WRITE FAILED: {e}")

# ── 8. Check settings ─────────────────────────────────────────────────────────
print("\n── Settings check ───────────────────────────")
from config import settings
print(f"  dry_run     : {settings.dry_run}")
print(f"  offline_mode: {settings.offline_mode}")
print(f"  neo4j_uri   : {settings.neo4j_uri}")
print(f"  openai_key  : {'SET' if settings.openai_api_key else 'MISSING'}")
print(f"  tavily_key  : {'SET' if settings.tavily_api_key else 'MISSING'}")

print("\nDone.\n")
memory.close()
