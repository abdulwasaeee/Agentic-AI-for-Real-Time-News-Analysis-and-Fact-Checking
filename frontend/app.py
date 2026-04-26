"""
FactGuard - Agentic AI Fact Checking Dashboard
Task 3: Full-Stack & Evaluation Engineer

Run with: streamlit run frontend/app.py
"""

import sys
import os

# Ensure the project root is on sys.path so imports like `from agents...` work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import time
import threading
from datetime import datetime, timedelta, timezone

# ─────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FactGuard | AI Fact Checker",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — original UI design preserved exactly
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* Main container */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* Hero title */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #f0f4ff;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    font-size: 1rem;
    color: #64748b;
    margin-bottom: 2rem;
}

/* Verdict cards */
.verdict-card {
    background: linear-gradient(135deg, #1e2535 0%, #151b2e 100%);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.verdict-supported {
    border-left: 4px solid #10b981;
    box-shadow: 0 0 20px rgba(16,185,129,0.08);
}
.verdict-refuted {
    border-left: 4px solid #ef4444;
    box-shadow: 0 0 20px rgba(239,68,68,0.08);
}
.verdict-misleading {
    border-left: 4px solid #f59e0b;
    box-shadow: 0 0 20px rgba(245,158,11,0.08);
}

/* Badge */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-supported { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-refuted { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-misleading { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2535;
}

/* Info metric */
.metric-box {
    background: #1e2535;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 2px;
}

/* Source pill */
.source-pill {
    display: inline-block;
    background: #1e2535;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.8rem;
    margin: 4px;
    color: #94a3b8;
}

/* Tab styling override */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #1e2535;
    padding: 6px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    background: #2d3748 !important;
    color: #a78bfa !important;
}

/* Input styling */
.stTextArea textarea, .stTextInput input {
    background: #1e2535 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.6rem 2rem;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    opacity: 0.85;
}

/* Image mismatch warning */
.img-mismatch-warning {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #fca5a5;
    font-size: 0.85rem;
}
.img-match-ok {
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #6ee7b7;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# BACKEND INITIALIZATION (lazy — avoids startup crash if DBs not running)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _init_memory():
    """Initialize MemoryAgent singleton once per Streamlit session."""
    try:
        from agents.memory_agent import get_memory
        return get_memory(), None
    except Exception as e:
        return None, str(e)


def _get_memory():
    memory, err = _init_memory()
    return memory


# ─────────────────────────────────────────────
# REAL BACKEND FUNCTIONS
# ─────────────────────────────────────────────

def _run_entity_tracker_background(claim_text_or_name: str, direct_name: str = "") -> None:
    """
    Run the Entity Tracker in a background thread after every fact-check.

    Accepts either a claim text (extracts entities via spaCy) or a direct
    entity name string. Always includes `direct_name` if provided.
    """
    def _worker():
        try:
            from agents.entity_tracker import run_entity_tracker
            from agents.memory_agent import get_memory

            print(f"\n[bg_tracker] thread started  claim_text_or_name={claim_text_or_name!r}  direct_name={direct_name!r}")
            entity_names: set[str] = set()

            # Include direct entity name if given (e.g. from entity search box)
            # Also ensure the entity node exists in Neo4j — spaCy won't create
            # nodes for common nouns like "apple" or "bitcoin".
            if direct_name:
                clean = direct_name.strip()
                entity_names.add(clean)
                print(f"[bg_tracker] direct_name provided → adding {clean!r}")
                try:
                    memory = get_memory()
                    memory.ensure_entity_exists(clean)
                    print(f"[bg_tracker] ensure_entity_exists OK for {clean!r}")
                    linked = memory.backfill_mentions_for_entity(clean)
                    print(f"[bg_tracker] backfill_mentions linked {linked} existing claim(s) → entity {clean!r}")
                except Exception as _ex:
                    print(f"[bg_tracker] ensure/backfill FAILED: {_ex}")

            # Also extract from claim text via spaCy
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
                doc = nlp(claim_text_or_name)
                spacy_hits = [(e.text, e.label_) for e in doc.ents
                              if e.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "NORP", "FAC"}
                              and len(e.text.strip()) > 1]
                print(f"[bg_tracker] spaCy NER on claim text → {spacy_hits}")
                for ent_text, _ in spacy_hits:
                    entity_names.add(ent_text.strip())
            except Exception as _ex:
                print(f"[bg_tracker] spaCy failed: {_ex}")

            print(f"[bg_tracker] entity_names to track: {entity_names}")
            if not entity_names:
                print("[bg_tracker] nothing to track — exiting thread")
                return

            memory = get_memory()
            for name in entity_names:
                print(f"[bg_tracker] → run_entity_tracker({name!r})")
                run_entity_tracker(name, window_hours=720, memory=memory)
            print("[bg_tracker] thread done")
        except Exception as _ex:
            import traceback
            print(f"[bg_tracker] UNHANDLED ERROR: {_ex}")
            traceback.print_exc()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def _scrape_article(url: str) -> dict:
    """Fetch og:title, og:description, og:image from a URL. Returns dict with keys: title, description, image_url."""
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (compatible; FactGuard/1.0)"}
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        def og(prop):
            tag = soup.find("meta", property=f"og:{prop}") or soup.find("meta", attrs={"name": f"og:{prop}"})
            return tag["content"].strip() if tag and tag.get("content") else ""
        title   = og("title") or (soup.title.string.strip() if soup.title else "")
        desc    = og("description")
        img_url = og("image")
        if not desc:
            # fall back to first <p>
            p = soup.find("p")
            desc = p.get_text().strip()[:400] if p else ""
        return {"title": title, "description": desc, "image_url": img_url}
    except Exception as e:
        print(f"[scrape_article] failed for {url}: {e}")
        return {"title": "", "description": "", "image_url": ""}


def get_real_verdict(query: str) -> dict:
    """Run the LangGraph fact-check pipeline, with URL scraping and human cache lookup."""
    claim_text = query
    image_url  = ""
    source_url = "https://unknown.source"

    # ── URL detection: scrape article ──────────────────────────────────
    if query.strip().startswith("http"):
        scraped = _scrape_article(query.strip())
        title   = scraped["title"]
        desc    = scraped["description"]
        image_url  = scraped["image_url"] or ""
        source_url = query.strip()
        claim_text = f"{title}. {desc}".strip(" .") if title or desc else query
        print(f"[get_real_verdict] URL scraped → claim_text={claim_text[:120]!r}  image_url={image_url!r}")

    # ── Human verdict cache ─────────────────────────────────────────────
    mem = _get_memory()
    if mem is not None:
        try:
            cached = mem.find_human_verdict_for_claim(claim_text)
            print(f"[get_real_verdict] human_cache result: {cached}")
            if cached:
                print(f"[get_real_verdict] human_cache HIT → label={cached['label']}  conf={cached['confidence']}")
                st.session_state["_last_claim_text"] = claim_text
                st.session_state["_last_image_url"]  = image_url
                return {
                    "label":            cached["label"],
                    "confidence":       float(cached["confidence"]),
                    "claim_text":       claim_text[:200],
                    "evidence_summary": "(Human-corrected verdict from memory)",
                    "bias_score":       float(cached.get("bias_score", 0.5)),
                    "image_mismatch":   False,
                    "image_url":        image_url,
                    "vlm_caption":      "",
                    "sources":          [],
                    "charged_phrases":  [],
                    "verdict_id":       cached.get("verdict_id", ""),
                    "claim_id":         cached.get("claim_id", ""),
                    "_from_cache":      True,
                }
        except Exception as _ce:
            print(f"[get_real_verdict] human_cache MISS (exception): {_ce}")

    # ── Run pipeline ────────────────────────────────────────────────────
    try:
        from agents.fact_check_agent import fact_check_claim
        output = fact_check_claim(
            claim_text,
            source_url=source_url,
            image_url=image_url or None,
        )

        st.session_state["_last_claim_text"] = claim_text
        st.session_state["_last_image_url"]  = image_url

        # Store claim + entities in memory for entity tracker
        if mem is not None:
            try:
                import spacy as _spacy
                _nlp = _spacy.load("en_core_web_sm")
                _doc = _nlp(claim_text)
                _valid_labels = {"PERSON", "ORG", "GPE", "PRODUCT", "NORP", "FAC"}
                from id_utils import make_id
                entity_dicts = [
                    {"entity_id": make_id("ent_"), "name": e.text.strip(),
                     "entity_type": e.label_.lower(), "sentiment": "neutral"}
                    for e in _doc.ents if e.label_ in _valid_labels and len(e.text.strip()) > 1
                ]
                # Deduplicate by name
                seen_names: set[str] = set()
                unique_entities = []
                for ed in entity_dicts:
                    if ed["name"] not in seen_names:
                        seen_names.add(ed["name"])
                        unique_entities.append(ed)
                if unique_entities:
                    st.session_state["_auto_detected_entities"] = [e["name"] for e in unique_entities]
                    print(f"[get_real_verdict] auto_store_claim_with_entities  claim_id={output.claim_id!r}  entities={[e['name'] for e in unique_entities]}")
                    mem.auto_store_claim_with_entities(
                        claim_id=output.claim_id,
                        claim_text=claim_text,
                        article_id=make_id("art_"),
                        entity_dicts=unique_entities,
                    )
            except Exception as _ee:
                print(f"[get_real_verdict] entity auto-store skipped: {_ee}")

        return {
            "label":            output.verdict,
            "confidence":       output.confidence_score / 100,
            "claim_text":       claim_text[:200],
            "evidence_summary": output.reasoning,
            "bias_score":       output.bias_score,
            "image_mismatch":   output.cross_modal_flag,
            "image_url":        image_url,
            "vlm_caption":      output.cross_modal_explanation or "",
            "sources":          [{"name": url.split("/")[2] if len(url.split("/")) > 2 else url,
                                  "url": url, "credibility": 0.5}
                                 for url in (output.evidence_links or [])[:4]],
            "charged_phrases":  [],
            "verdict_id":       output.verdict_id,
            "claim_id":         output.claim_id,
            "_from_cache":      False,
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {
            "label":            "misleading",
            "confidence":       0.0,
            "claim_text":       claim_text[:200],
            "evidence_summary": f"Pipeline error: {e}",
            "bias_score":       0.5,
            "image_mismatch":   False,
            "image_url":        image_url,
            "vlm_caption":      "",
            "sources":          [],
            "charged_phrases":  [],
            "verdict_id":       "",
            "claim_id":         "",
            "_from_cache":      False,
        }


def get_real_entity_history(entity_name: str) -> pd.DataFrame:
    """
    Query Graph DB for CredibilitySnapshot history for a given entity.

    Returns a DataFrame with columns: date, credibility_score, sentiment_score.
    Falls back to an empty DataFrame if the entity is not found.
    """
    memory = _get_memory()
    if memory is None:
        return _empty_entity_df()

    try:
        print(f"\n[get_entity_history] looking up entity {entity_name!r}")
        entity_dict = memory.get_entity_by_name(entity_name)
        print(f"[get_entity_history] get_entity_by_name result: {entity_dict}")
        if entity_dict is None:
            print(f"[get_entity_history] entity NOT found in Neo4j")
            st.info(f"No data yet for '{entity_name}'. Fact-check a claim mentioning them first — the tracker runs automatically.")
            return _empty_entity_df()

        print(f"[get_entity_history] entity found: id={entity_dict['entity_id']!r}")
        snapshots = memory.get_entity_snapshots(entity_dict["entity_id"], limit=60)
        print(f"[get_entity_history] snapshots returned: {len(snapshots)}")
        if not snapshots:
            print(f"[get_entity_history] entity exists but has 0 snapshots")
            st.info(f"'{entity_name}' was found but has no credibility snapshots yet. Check a claim mentioning them and the chart will populate.")
            return _empty_entity_df()

        rows = []
        for snap in snapshots:
            snap_at = snap["snapshot_at"]
            # Neo4j returns neo4j.time.DateTime — convert to Python datetime
            if hasattr(snap_at, "to_native"):
                snap_at = snap_at.to_native()
            rows.append({
                "date":             snap_at,
                "credibility_score": snap["credibility_score"],
                "sentiment_score":   snap["sentiment_score"],
            })

        return pd.DataFrame(rows)

    except Exception as e:
        st.warning(f"Could not load entity history: {e}")
        return _empty_entity_df()


def _empty_entity_df() -> pd.DataFrame:
    return pd.DataFrame({"date": [], "credibility_score": [], "sentiment_score": []})


def get_real_predictions(entity_name: str) -> list:
    """
    Query Graph DB for pending Prediction nodes for the given entity.

    Falls back to an empty list if not available.
    """
    memory = _get_memory()
    if memory is None:
        return []

    try:
        entity_dict = memory.get_entity_by_name(entity_name)
        if entity_dict is None:
            return []

        preds = memory.get_predictions_for_entity(entity_dict["entity_id"], include_resolved=False)
        result = []
        for p in preds:
            deadline = p.get("deadline")
            if hasattr(deadline, "to_native"):
                deadline = deadline.to_native()
            deadline_str = deadline.strftime("%Y-%m-%d") if deadline else "—"
            result.append({
                "text":       p["prediction_text"],
                "confidence": float(p.get("confidence") or 0.0),
                "deadline":   deadline_str,
                "outcome":    p.get("outcome"),
            })
        return result

    except Exception as e:
        st.warning(f"Could not load predictions: {e}")
        return []


# ─────────────────────────────────────────────
# HELPER FUNCTIONS — original UI helpers preserved
# ─────────────────────────────────────────────

def render_verdict_badge(label: str) -> str:
    icons = {"supported": "✓", "refuted": "✗", "misleading": "⚠"}
    icon = icons.get(label, "?")
    return f'<span class="badge badge-{label}">{icon} {label.upper()}</span>'


def render_confidence_gauge(confidence: float, label: str):
    color_map = {"supported": "#10b981", "refuted": "#ef4444", "misleading": "#f59e0b"}
    color = color_map.get(label, "#a78bfa")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%", "font": {"size": 32, "color": "#e2e8f0", "family": "Space Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#475569", "size": 10}},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "#1e2535",
            "bordercolor": "#2d3748",
            "steps": [
                {"range": [0, 40], "color": "rgba(239,68,68,0.08)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.08)"},
                {"range": [70, 100], "color": "rgba(16,185,129,0.08)"},
            ],
            "threshold": {"line": {"color": color, "width": 2}, "thickness": 0.8, "value": confidence * 100}
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e2e8f0"}
    )
    return fig


def render_credibility_chart(df: pd.DataFrame, entity_name: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["credibility_score"],
        mode="lines+markers", name="Credibility",
        line=dict(color="#a78bfa", width=2.5, shape="spline"),
        marker=dict(size=4, color="#a78bfa"),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.06)"
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["sentiment_score"],
        mode="lines", name="Sentiment",
        line=dict(color="#38bdf8", width=1.5, dash="dot", shape="spline"),
        yaxis="y2"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="DM Sans"),
        title=dict(text=f"Credibility Drift — {entity_name}", font=dict(color="#e2e8f0", size=14)),
        xaxis=dict(gridcolor="#1e2535", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#1e2535", showgrid=True, zeroline=False,
                   range=[0, 1], title=dict(text="Credibility", font=dict(color="#a78bfa"))),
        yaxis2=dict(overlaying="y", side="right", range=[-1, 1],
                    title=dict(text="Sentiment", font=dict(color="#38bdf8")), showgrid=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2d3748"),
        height=300, margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified"
    )
    return fig


# ─────────────────────────────────────────────
# APP LAYOUT — original UI design preserved exactly
# ─────────────────────────────────────────────

# Header
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:0.5rem;">
    <span style="font-size:2.5rem;">🛡️</span>
    <div>
        <div class="hero-title">FactGuard</div>
        <div class="hero-subtitle">Multi-Agent AI Fact-Checking System · Powered by LLMs + Knowledge Graph</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Input Section ──
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_input = st.text_area(
        "Paste a news claim, article URL, or text snippet",
        placeholder="e.g. '500,000 Tesla vehicles were recalled due to brake defects' or paste a full article URL...",
        height=100,
        label_visibility="collapsed"
    )
with col_btn:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    run_btn = st.button("⚡ VERIFY", use_container_width=True)

# ── Entity search ──
# _auto_entity holds the last auto-detected entity name.
# We pass it as value= so Streamlit doesn't complain about modifying
# a keyed widget after instantiation.
if "_auto_entity" not in st.session_state:
    st.session_state["_auto_entity"] = ""

entity_col, _ = st.columns([3, 7])
with entity_col:
    entity_query = st.text_input(
        "Track an entity",
        value=st.session_state["_auto_entity"],
        placeholder="e.g. Elon Musk, Tesla, WHO...",
        label_visibility="visible",
    )
    # Sync whatever the user typed back into session_state so the background
    # tracker can read it when the Verify button fires.
    if entity_query.strip():
        st.session_state["_auto_entity"] = entity_query.strip()
        print(f"[entity_box] user typed entity: {entity_query.strip()!r}")

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RUN PIPELINE on button click
# ─────────────────────────────────────────────
if run_btn and user_input.strip():
    # Guardrail check
    try:
        from agents.input_guardrail import check_input
        _guard = check_input(user_input)
        if _guard["blocked"]:
            st.error(f"⚠️ **Input blocked** [{_guard['risk']} risk]: {_guard['reason']}")
            st.stop()
    except Exception:
        pass

    st.session_state["_auto_detected_entities"] = []  # clear stale entities from last run
    with st.spinner("🔍 Agents working: Scraping → Preprocessing → Fact-Checking..."):
        result = get_real_verdict(user_input)

    # Persist result so widget interactions don't wipe it
    st.session_state["_last_result"]     = result
    st.session_state["_last_user_input"] = user_input

    # Auto-fill entity box if user didn't type one
    _user_typed_entity = entity_query.strip()
    if not _user_typed_entity:
        _claim_for_ner = st.session_state.get("_last_claim_text", user_input)
        _auto_entity = ""
        try:
            import spacy as _spacy
            _nlp = _spacy.load("en_core_web_sm")
            _doc = _nlp(_claim_for_ner)
            _valid_labels = {"PERSON", "ORG", "GPE", "PRODUCT", "NORP", "FAC"}
            _found = [e.text.strip() for e in _doc.ents
                      if e.label_ in _valid_labels and len(e.text.strip()) > 1]
            if _found:
                _auto_entity = _found[0]
        except Exception:
            pass

        if not _auto_entity and user_input.strip().startswith("http"):
            try:
                from urllib.parse import urlparse
                _path = urlparse(user_input).path
                _words = [w for w in _path.replace("/", "-").split("-")
                          if len(w) > 3 and w.isalpha()]
                if _words:
                    _auto_entity = _words[0].capitalize()
            except Exception:
                pass

        if _auto_entity:
            st.session_state["_auto_entity"] = _auto_entity

    # Run entity tracker in background
    _tracker_entity = st.session_state.get("_auto_entity", "").strip()
    if _tracker_entity:
        _run_entity_tracker_background("", direct_name=_tracker_entity)


# ─────────────────────────────────────────────
# RESULTS — rendered from session state so they survive widget reruns
# ─────────────────────────────────────────────
result = st.session_state.get("_last_result")
if result is not None:
    _user_input_display = st.session_state.get("_last_user_input", "")
    _image_url   = st.session_state.get("_last_image_url", result.get("image_url", ""))
    _claim_text  = st.session_state.get("_last_claim_text", result.get("claim_text", ""))
    _auto_dets   = st.session_state.get("_auto_detected_entities", [])
    _tracker_ent = st.session_state.get("_auto_entity", "").strip()

    # Pills banner for tracked entities
    _all_tracked = []
    if _tracker_ent:
        _all_tracked.append(_tracker_ent)
    for _ae in _auto_dets:
        if _ae not in _all_tracked:
            _all_tracked.append(_ae)
    if _all_tracked:
        _pills_html = " ".join(
            f'<span style="background:#1e2535;border:1px solid #4f46e5;border-radius:999px;'
            f'padding:3px 10px;font-size:0.75rem;color:#a78bfa;margin:2px;">📍 {e}</span>'
            for e in _all_tracked[:6]
        )
        st.markdown(f'<div style="margin-bottom:0.8rem">Tracking: {_pills_html}</div>',
                    unsafe_allow_html=True)

    # ── 2 TABS ──
    tab1, tab2 = st.tabs([
        "📋  Fact Verdict",
        "📈  Entity & Trend"
    ])

    # ─────────────────────
    # TAB 1: FACT VERDICT
    # ─────────────────────
    with tab1:
        st.markdown('<div class="section-header">Verification Result</div>', unsafe_allow_html=True)

        left, right = st.columns([3, 2])

        with left:
            label = result["label"]
            st.markdown(f"""
            <div class="verdict-card verdict-{label}">
                <div style="margin-bottom:0.8rem">{render_verdict_badge(label)}</div>
                <div style="font-size:0.85rem; color:#94a3b8; margin-bottom:0.8rem; font-style:italic;">
                    "{result['claim_text']}"
                </div>
                <div style="font-size:0.9rem; color:#cbd5e1; line-height:1.7;">
                    {result['evidence_summary']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Verified sources
            st.markdown('<div class="section-header" style="margin-top:1rem">Verified Sources</div>', unsafe_allow_html=True)
            if result["sources"]:
                source_html = ""
                for src in result["sources"]:
                    cred_color = "#10b981" if src["credibility"] > 0.7 else "#ef4444"
                    source_html += f'<span class="source-pill">🔗 {src["name"]} <span style="color:{cred_color}; font-weight:600;">{src["credibility"]:.0%}</span></span>'
                st.markdown(source_html, unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#475569; font-size:0.85rem;">No external sources retrieved.</span>', unsafe_allow_html=True)

            # ── Human Feedback Form ─────────────────────────────────────
            _verdict_id = result.get("verdict_id", "")
            if _verdict_id:
                st.markdown('<div class="section-header" style="margin-top:1.2rem">Correct This Verdict</div>', unsafe_allow_html=True)
                with st.form(key="feedback_form"):
                    _fb_label = st.radio(
                        "Correct label",
                        ["supported", "refuted", "misleading"],
                        index=["supported", "refuted", "misleading"].index(result["label"])
                            if result["label"] in ["supported", "refuted", "misleading"] else 2,
                        horizontal=True,
                    )
                    _fb_conf = st.slider("Correct confidence", 0, 100,
                                         int(result["confidence"] * 100), step=5)
                    _fb_note = st.text_input("Note (optional)", placeholder="Why is this verdict incorrect?")
                    _fb_submit = st.form_submit_button("Submit Correction")

                if _fb_submit:
                    _mem = _get_memory()
                    if _mem is not None:
                        try:
                            _mem.update_verdict_with_feedback(
                                verdict_id=_verdict_id,
                                correct_label=_fb_label,
                                correct_confidence=_fb_conf / 100,
                                feedback_note=_fb_note,
                            )
                            # Patch session state so UI reflects immediately
                            st.session_state["_last_result"]["label"]      = _fb_label
                            st.session_state["_last_result"]["confidence"] = _fb_conf / 100
                            st.success(f"Correction saved → {_fb_label.upper()} at {_fb_conf}%")
                            st.rerun()
                        except Exception as _fe:
                            st.error(f"Could not save correction: {_fe}")
                    else:
                        st.warning("Memory agent not available.")

        with right:
            st.markdown('<div class="section-header">Confidence Score</div>', unsafe_allow_html=True)
            st.plotly_chart(render_confidence_gauge(result["confidence"], label),
                            use_container_width=True, config={"displayModeBar": False})

            # ── Image Cross-Check ───────────────────────────────────────
            st.markdown('<div class="section-header" style="margin-top:0.5rem">Image Cross-Check</div>', unsafe_allow_html=True)
            if _image_url:
                if result["image_mismatch"]:
                    explanation = result.get("vlm_caption") or "The article image does not match the described event context."
                    st.markdown(f"""
                    <div class="img-mismatch-warning">
                        ⚠️ <strong>Image Mismatch Detected</strong><br>
                        {explanation}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    caption_text = result.get("vlm_caption") or "Image content aligns with the article claim."
                    st.markdown(f"""
                    <div class="img-match-ok">
                        ✓ <strong>Image Consistent</strong><br>
                        {caption_text}
                    </div>
                    """, unsafe_allow_html=True)
                st.image(_image_url,
                         caption=result["vlm_caption"] if result.get("vlm_caption") else "Article image",
                         use_container_width=True)
            else:
                st.markdown("""
                <div style="color:#475569; font-size:0.85rem; padding:0.5rem 0;">
                    🖼️ No image detected in this article.
                </div>
                """, unsafe_allow_html=True)

            # ── Snapshot counter ────────────────────────────────────────
            _snap_entity = _tracker_ent or (_auto_dets[0] if _auto_dets else "")
            if _snap_entity:
                _mem2 = _get_memory()
                if _mem2 is not None:
                    try:
                        _edict = _mem2.get_entity_by_name(_snap_entity)
                        if _edict:
                            _snaps = _mem2.get_entity_snapshots(_edict["entity_id"], limit=5)
                            _snap_count = len(_snaps)
                            _needed = 3
                            st.markdown(f"""
                            <div style="margin-top:0.8rem; font-size:0.8rem; color:#94a3b8;">
                                📊 SNAPSHOTS FOUND {_snap_count} / {_needed} needed for credibility graph
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(min(_snap_count / _needed, 1.0))
                    except Exception:
                        pass

    # ─────────────────────
    # TAB 2: ENTITY & TREND
    # ─────────────────────
    with tab2:
        entity_name = entity_query.strip() if entity_query.strip() else \
                      (_auto_dets[0] if _auto_dets else _tracker_ent or "Tesla")
        st.markdown(f'<div class="section-header">Entity Profile — {entity_name}</div>', unsafe_allow_html=True)

        df = get_real_entity_history(entity_name)

        if df.empty:
            st.info(f"No credibility history yet for '{entity_name}'. Check a claim mentioning them and it will appear here automatically.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            current_cred = df["credibility_score"].iloc[-1]
            cred_change  = df["credibility_score"].iloc[-1] - df["credibility_score"].iloc[0]
            avg_sent     = df["sentiment_score"].mean()

            with m1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color:{'#10b981' if current_cred > 0.6 else '#ef4444'};">
                        {current_cred:.0%}
                    </div>
                    <div class="metric-label">Current Credibility</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                arrow = "↓" if cred_change < 0 else "↑"
                col = "#ef4444" if cred_change < 0 else "#10b981"
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color:{col};">{arrow} {abs(cred_change):.0%}</div>
                    <div class="metric-label">Drift ({len(df)}-snapshot span)</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                sent_label = "Positive" if avg_sent > 0.2 else ("Negative" if avg_sent < -0.1 else "Neutral")
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value" style="font-size:1.4rem;">{sent_label}</div>
                    <div class="metric-label">Avg Sentiment</div>
                </div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{len(df)}</div>
                    <div class="metric-label">Snapshots Tracked</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.plotly_chart(render_credibility_chart(df, entity_name),
                            use_container_width=True, config={"displayModeBar": False})

        # Predictions
        st.markdown('<div class="section-header" style="margin-top:0.5rem">AI Predictions</div>', unsafe_allow_html=True)
        predictions = get_real_predictions(entity_name)

        if not predictions:
            st.info(f"No predictions yet for '{entity_name}'. They generate automatically after credibility history builds up.")
        else:
            for pred in predictions:
                conf_color = "#10b981" if pred["confidence"] > 0.7 else "#f59e0b"
                st.markdown(f"""
                <div class="verdict-card" style="margin-bottom:0.7rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                        <span style="font-size:0.75rem; color:#64748b; font-family:'Space Mono',monospace;">
                            📅 Deadline: {pred['deadline']}
                        </span>
                        <span style="font-size:0.75rem; font-weight:700; color:{conf_color}; font-family:'Space Mono',monospace;">
                            {pred['confidence']:.0%} confidence
                        </span>
                    </div>
                    <div style="color:#cbd5e1; font-size:0.9rem;">{pred['text']}</div>
                </div>
                """, unsafe_allow_html=True)

elif run_btn and not user_input.strip():
    st.warning("Please enter a news claim or article URL to verify.")

# ─────────────────────────────────────────────
# EMPTY STATE (no query yet)
# ─────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div style="text-align:center; padding:4rem 0; color:#334155;">
        <div style="font-size:4rem; margin-bottom:1rem;">🛡️</div>
        <div style="font-family:'Space Mono',monospace; font-size:1rem; color:#475569;">
            Paste a claim above and click VERIFY to begin fact-checking
        </div>
        <div style="margin-top:1rem; font-size:0.8rem; color:#334155;">
            Powered by 6 specialized AI agents · Real-time web search · Knowledge Graph memory
        </div>
    </div>
    """, unsafe_allow_html=True)
