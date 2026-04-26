"""All LangGraph node functions for the Fact-Check Agent graph.

Each node is a pure function: (state) -> dict of partial state updates.
Nodes that need MemoryAgent receive it as a second argument via closure (see graph.py).
"""
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import llm_factory as _llm_factory

from agents.reflection_agent import query_source_credibility, update_source_credibility
from tools.cross_modal_tool import check_cross_modal
from tools.freshness_tool import check_freshness
from tools.live_search_tool import format_search_context, search_live
from tools.rag_tool import format_rag_context, retrieve_similar_claims
from tools.reranker import rerank_candidates
from models.schemas import FactCheckOutput, MemoryQueryResponse, SimilarClaim
from orchestration.state import FactCheckState
from prompts import (
    ADVOCATE_PROMPT, ARBITER_PROMPT, DECOMPOSITION_PROMPT,
    IS_RETRIEVAL_NEEDED_PROMPT, VERDICT_SYNTHESIS_PROMPT,
)

if TYPE_CHECKING:
    from agents.memory_agent import MemoryAgent

logger = logging.getLogger(__name__)


# ── Node: receive_claim ───────────────────────────────────────────────────────

def receive_claim(state: FactCheckState) -> dict:
    prefetched = list(state["input"].prefetched_chunks)
    return {
        "memory_results":          None,
        "entity_context":          [],
        "route":                   None,
        "revalidation_needed":     None,
        "retrieval_gate_needed":   None,
        "retrieved_chunks":        prefetched,
        "sub_claims":              [],
        "debate_transcript":       None,
        "source_credibility":      None,
        "cross_modal_flag":        False,
        "cross_modal_explanation": None,
        "clip_similarity_score":   None,
        "last_verified_at":        None,
        "output":                  None,
    }


# ── Node: query_memory ────────────────────────────────────────────────────────

def query_memory(state: FactCheckState, memory: "MemoryAgent", settings=None) -> dict:
    from config import settings as _settings
    if settings is None:
        settings = _settings

    if settings.offline_mode:
        return {
            "memory_results":     MemoryQueryResponse(results=[], max_confidence=0.0),
            "entity_context":     [],
            "source_credibility": {},
            "last_verified_at":   None,
        }

    inp = state["input"]
    vector_results = retrieve_similar_claims(inp.claim_text, memory)
    entity_ctx     = memory.get_entity_context(inp.claim_id)

    graph_results: list[dict] = []
    if settings.use_graph_rag and vector_results:
        claim_ids  = [c["claim_id"] for c in vector_results]
        entity_ids = [e["entity_id"] for e in memory.get_entity_ids_for_claims(claim_ids)]
        if entity_ids:
            graph_results = memory.get_graph_claims_for_entities(entity_ids)

    reranked = rerank_candidates(
        query              = inp.claim_text,
        vector_results     = vector_results,
        graph_results      = graph_results,
        use_cross_encoder  = settings.use_cross_encoder,
        cross_encoder_model= settings.cross_encoder_model,
        top_k              = settings.reranker_top_k,
    )

    results = [
        SimilarClaim(
            claim_id           = c["claim_id"],
            claim_text         = c["claim_text"],
            verdict_label      = c.get("verdict_label"),
            verdict_confidence = c.get("verdict_confidence"),
            distance           = c.get("distance", 0.0),
            verified_at        = c.get("verified_at"),
        )
        for c in reranked
    ]

    max_confidence = max(
        (r.verdict_confidence for r in results if r.verdict_confidence is not None),
        default=0.0,
    )
    best = next((r for r in results if r.verdict_confidence == max_confidence and r.verified_at), None)
    last_verified_at = best.verified_at if best else None

    source_cred = query_source_credibility(
        claim_text=inp.claim_text, source_url=inp.source_url, memory=memory,
    )

    return {
        "memory_results":     MemoryQueryResponse(results=results, max_confidence=max_confidence),
        "entity_context":     entity_ctx,
        "last_verified_at":   last_verified_at,
        "source_credibility": source_cred,
    }


# ── Node: return_cached ───────────────────────────────────────────────────────

def return_cached(state: FactCheckState) -> dict:
    best = next((r for r in state["memory_results"].results if r.verdict_label), None)
    chunk = (
        f"[CACHE HIT] Similar verified claim: \"{best.claim_text}\" "
        f"— Prior verdict: {best.verdict_label} ({best.verdict_confidence:.0%} confidence)"
        if best else "[CACHE HIT] Prior verdict found but no claim text available."
    )
    return {"retrieved_chunks": [chunk], "route": "cache"}


# ── Node: freshness_check ────────────────────────────────────────────────────

def freshness_check(state: FactCheckState, settings) -> dict:
    last_verified_at = state.get("last_verified_at")
    if not last_verified_at:
        return {"revalidation_needed": True}

    memory_results = state["memory_results"]
    best = next((r for r in memory_results.results if r.verdict_label and r.verified_at), None)
    if not best:
        return {"revalidation_needed": True}

    result = check_freshness(
        claim_text         = state["input"].claim_text,
        verdict_label      = best.verdict_label,
        verdict_confidence = best.verdict_confidence or 0.0,
        last_verified_at   = last_verified_at,
        api_key            = settings.openai_api_key,
        model              = _llm_factory.llm_model_name(),
    )
    return {"revalidation_needed": result["revalidate"]}


# ── Node: live_search ─────────────────────────────────────────────────────────

def live_search(state: FactCheckState, settings) -> dict:
    if state.get("retrieved_chunks"):
        return {"route": "live_search"}
    results         = search_live(state["input"].claim_text, api_key=settings.tavily_api_key)
    context, _links = format_search_context(results)
    return {"retrieved_chunks": [context], "route": "live_search"}


# ── Node: rag_retrieval ───────────────────────────────────────────────────────

def rag_retrieval(state: FactCheckState) -> dict:
    rag_context = format_rag_context([r.model_dump() for r in state["memory_results"].results])
    return {"retrieved_chunks": list(state["retrieved_chunks"]) + [rag_context]}


# ── Node: retrieval_gate ──────────────────────────────────────────────────────

def retrieval_gate(state: FactCheckState, settings) -> dict:
    if not settings.use_retrieval_gate:
        return {"retrieval_gate_needed": True}
    inp = state["input"]
    memory_context = ""
    if state.get("memory_results") and state["memory_results"].results:
        memory_context = format_rag_context([r.model_dump() for r in state["memory_results"].results])
    prompt = IS_RETRIEVAL_NEEDED_PROMPT.format(claim_text=inp.claim_text)
    if memory_context:
        prompt += f"\n\nEXISTING CONTEXT FROM MEMORY:\n{memory_context}"
    client = _llm_factory.make_llm_client()
    try:
        response = client.chat.completions.create(
            model=_llm_factory.llm_model_name(),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        return {"retrieval_gate_needed": bool(result.get("retrieval_needed", True))}
    except Exception as e:
        logger.error("retrieval_gate failed: %s", e)
        return {"retrieval_gate_needed": True}


# ── Node: synthesize_verdict ──────────────────────────────────────────────────

def synthesize_verdict(state: FactCheckState, settings) -> dict:
    from id_utils import make_id

    inp            = state["input"]
    evidence_block = "\n\n".join(state["retrieved_chunks"]) or "No evidence retrieved."
    cred_lines     = [f"Source: {inp.source_url}"]

    sc = state.get("source_credibility") or {}
    cred_mean    = sc.get("credibility_mean")
    bias_mean    = sc.get("bias_mean")
    bias_std     = sc.get("bias_std")
    sample_count = sc.get("sample_count", 0)

    if cred_mean is not None and sample_count >= 2:
        cred_lines.append(f"Source credibility: {cred_mean:.0%} (based on {sample_count} past verdicts)")
        cred_lines.append(f"Source bias: {bias_mean:.2f} ± {bias_std:.2f}")
    elif sample_count == 1:
        cred_lines.append("Source credibility: only 1 prior verdict — insufficient for reliable estimate")
    else:
        cred_lines.append("Source credibility: no prior verdicts for this source")

    if state["entity_context"]:
        cred_lines.append("Entity credibility context:")
        for e in state["entity_context"]:
            cred_lines.append(
                f"  - {e['name']} ({e['entity_type']}): "
                f"credibility {e.get('current_credibility', 0.5):.2f}, "
                f"sentiment: {e.get('sentiment', 'neutral')}"
            )
    source_credibility_note = "\n".join(cred_lines)

    prompt = VERDICT_SYNTHESIS_PROMPT.format(
        claim_text=inp.claim_text,
        evidence_block=evidence_block,
        source_credibility_note=source_credibility_note,
    )

    client = _llm_factory.make_llm_client()
    try:
        response = client.chat.completions.create(
            model=_llm_factory.llm_model_name(),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error("Verdict synthesis failed: %s", e)
        result = {"verdict": "misleading", "confidence_score": 0, "bias_score": 0.5,
                  "reasoning": f"Synthesis failed: {e}", "evidence_links": []}

    output = FactCheckOutput(
        verdict_id       = make_id("vrd_"),
        claim_id         = inp.claim_id,
        verdict          = result.get("verdict", "misleading"),
        confidence_score = int(result.get("confidence_score", 0)),
        evidence_links   = result.get("evidence_links", []),
        reasoning        = result.get("reasoning", ""),
        bias_score       = float(result.get("bias_score", 0.5)),
        cross_modal_flag = False,
    )
    return {"output": output}


# ── Node: decompose_claim ─────────────────────────────────────────────────────

def decompose_claim(state: FactCheckState, settings) -> dict:
    if not settings.use_claim_decomposition:
        return {}
    inp = state["input"]
    prompt = DECOMPOSITION_PROMPT.format(claim_text=inp.claim_text)
    client = _llm_factory.make_llm_client()
    try:
        response = client.chat.completions.create(
            model=_llm_factory.llm_model_name(),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        sub_claims = [sc["text"] for sc in result.get("sub_claims", []) if sc.get("verifiable", True)]
        return {"sub_claims": sub_claims}
    except Exception as e:
        logger.error("decompose_claim failed: %s", e)
        return {}


# ── Node: multi_agent_debate ──────────────────────────────────────────────────

def multi_agent_debate(state: FactCheckState, settings) -> dict:
    from id_utils import make_id

    inp            = state["input"]
    output         = state.get("output")
    evidence_block = "\n\n".join(state["retrieved_chunks"]) or "No evidence retrieved."
    client         = _llm_factory.make_llm_client()

    def _call(prompt_text: str) -> str:
        resp = client.chat.completions.create(
            model=_llm_factory.llm_model_name(),
            messages=[{"role": "user", "content": prompt_text}], temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    try:
        argument_for     = _call(ADVOCATE_PROMPT.format(position="TRUE (supported)", position_adj="supporting",
                                                         claim_text=inp.claim_text, evidence_block=evidence_block))
        argument_against = _call(ADVOCATE_PROMPT.format(position="FALSE (refuted)", position_adj="refuting",
                                                         claim_text=inp.claim_text, evidence_block=evidence_block))
        arbiter_raw      = _call(ARBITER_PROMPT.format(claim_text=inp.claim_text,
                                                        argument_for=argument_for,
                                                        argument_against=argument_against))
        if arbiter_raw.startswith("```"):
            arbiter_raw = arbiter_raw.split("```")[1].lstrip("json").strip()
        result = json.loads(arbiter_raw)
        transcript = (f"=== FOR ===\n{argument_for}\n\n=== AGAINST ===\n{argument_against}\n\n"
                      f"=== ARBITER ===\n{arbiter_raw}")
        if output:
            updated_output = output.model_copy(update={
                "verdict":          result.get("verdict", output.verdict),
                "confidence_score": int(result.get("confidence_score", output.confidence_score)),
                "bias_score":       float(result.get("bias_score", output.bias_score)),
                "reasoning":        result.get("reasoning", output.reasoning),
                "evidence_links":   result.get("evidence_links", output.evidence_links),
            })
        else:
            updated_output = output
        return {"output": updated_output, "debate_transcript": transcript}
    except Exception as e:
        logger.error("multi_agent_debate failed: %s", e)
        return {}


# ── Node: cross_modal_check ───────────────────────────────────────────────────

def cross_modal_check(state: FactCheckState, settings) -> dict:
    inp    = state["input"]
    result = check_cross_modal(
        claim_text    = inp.claim_text,
        image_caption = inp.image_caption,
        api_key       = settings.openai_api_key,
        model         = _llm_factory.llm_model_name(),
        image_url     = getattr(inp, "image_url", None),
    )
    current_output: Optional[FactCheckOutput] = state.get("output")
    updated_output = None
    if current_output:
        updated_output = current_output.model_copy(update={
            "cross_modal_flag":        result["flag"],
            "cross_modal_explanation": result["explanation"],
        })
    return {
        "cross_modal_flag":        result["flag"],
        "cross_modal_explanation": result["explanation"],
        "clip_similarity_score":   result.get("siglip_score"),
        "output":                  updated_output or current_output,
    }


# ── Node: write_memory ────────────────────────────────────────────────────────

def write_memory(state: FactCheckState, memory: "MemoryAgent") -> dict:
    from config import settings as _settings
    if _settings.dry_run or _settings.offline_mode:
        return {}

    from models.verdict import Verdict

    output: Optional[FactCheckOutput] = state.get("output")
    if not output:
        return {}

    evidence_summary = output.reasoning
    if output.evidence_links:
        evidence_summary += "\n\nSources: " + " | ".join(output.evidence_links)

    # ── STEP 1: Store Claim + Entity nodes FIRST ─────────────────────────
    # create_verdict uses MATCH (c:Claim), so the Claim node must exist
    # before we write the verdict, otherwise the link is silently lost.
    print(f"\n{'='*60}")
    print(f"[write_memory] claim_id={state['input'].claim_id}")
    print(f"[write_memory] claim_text={state['input'].claim_text[:80]!r}")
    try:
        import spacy as _spacy
        _nlp = _spacy.load("en_core_web_sm")
        _claim_text = state["input"].claim_text

        # For URLs, run NER on the URL path words, not the raw URL string
        _text_for_ner = _claim_text
        if _claim_text.strip().startswith("http"):
            try:
                from urllib.parse import urlparse as _urlparse
                _path = _urlparse(_claim_text).path
                _text_for_ner = _path.replace("/", " ").replace("-", " ").replace("_", " ")
                print(f"[write_memory] URL claim — NER text from path: {_text_for_ner!r}")
            except Exception:
                pass

        _doc = _nlp(_text_for_ner)
        _all_spacy_ents = [(e.text, e.label_) for e in _doc.ents]
        print(f"[write_memory] spaCy found {len(_all_spacy_ents)} raw ents: {_all_spacy_ents}")

        _label_map = {
            "PERSON": "person", "ORG": "organization", "GPE": "country",
            "LOC": "location", "PRODUCT": "product", "EVENT": "event",
            "NORP": "organization", "FAC": "location",
        }
        entity_dicts: list[dict] = []
        seen_names: set[str] = set()

        for ent in _doc.ents:
            if ent.label_ in _label_map and ent.text.strip() not in seen_names:
                seen_names.add(ent.text.strip())
                entity_dicts.append({
                    "entity_id":   f"ent_{ent.text.strip().lower().replace(' ', '_')}",
                    "name":        ent.text.strip(),
                    "entity_type": _label_map[ent.label_],
                })

        # Also include entities from the Preprocessing Agent (full pipeline)
        _pipeline_ents = state["input"].entities
        print(f"[write_memory] pipeline entities from input: {[e.name for e in _pipeline_ents]}")
        for e in _pipeline_ents:
            if e.name not in seen_names:
                seen_names.add(e.name)
                entity_dicts.append({
                    "entity_id":   e.entity_id,
                    "name":        e.name,
                    "entity_type": e.entity_type,
                })

        print(f"[write_memory] FINAL entity_dicts to store ({len(entity_dicts)} entities):")
        for ed in entity_dicts:
            print(f"  → {ed['name']!r}  id={ed['entity_id']!r}  type={ed['entity_type']!r}")
        if not entity_dicts:
            print("[write_memory] WARNING: no entities extracted — Claim will be stored but with no MENTIONS links")

        print(f"[write_memory] calling auto_store_claim_with_entities ...")
        memory.auto_store_claim_with_entities(
            claim_id     = state["input"].claim_id,
            claim_text   = _claim_text,
            article_id   = state["input"].article_id,
            entity_dicts = entity_dicts,
        )
        print(f"[write_memory] auto_store_claim_with_entities OK")
    except Exception as _e:
        import traceback
        print(f"[write_memory] ENTITY STORE FAILED: {_e}")
        traceback.print_exc()
        logger.warning("Entity auto-extraction skipped: %s", _e)

    # ── STEP 2: Write verdict (Claim node now exists, MATCH will succeed) ─
    print(f"[write_memory] writing verdict: label={output.verdict!r}  conf={output.confidence_score}")
    verdict = Verdict(
        verdict_id       = output.verdict_id,
        claim_id         = output.claim_id,
        label            = output.verdict,
        confidence       = output.confidence_score / 100,
        evidence_summary = evidence_summary,
        bias_score       = output.bias_score,
        image_mismatch   = output.cross_modal_flag,
        verified_at      = datetime.now(timezone.utc),
    )
    memory.add_verdict(verdict)
    print(f"[write_memory] verdict written OK  verdict_id={output.verdict_id}")

    update_source_credibility(
        claim_text       = state["input"].claim_text,
        source_url       = state["input"].source_url,
        verdict_id       = output.verdict_id,
        verdict_label    = output.verdict,
        confidence_score = output.confidence_score,
        bias_score       = output.bias_score,
        memory           = memory,
    )
    return {}


# ── Node: emit_output ─────────────────────────────────────────────────────────

def emit_output(state: FactCheckState) -> dict:
    current_output: Optional[FactCheckOutput] = state.get("output")
    if not current_output:
        return {}
    updated = current_output.model_copy(update={
        "last_verified_at":    state.get("last_verified_at"),
        "revalidation_needed": bool(state.get("revalidation_needed")),
    })
    return {"output": updated}
