"""Freshness tool — classifies whether a cached verdict needs live re-verification."""
import json
import logging
from datetime import datetime, timezone

import llm_factory as _llm_factory
from prompts import FRESHNESS_CHECK_PROMPT

logger = logging.getLogger(__name__)

_REACT_SYSTEM = """\
You are deciding whether a cached fact-check verdict needs live re-verification.
Use search only when necessary. After reasoning, return JSON with keys:
revalidate (bool), reason (str), claim_category (str).
"""

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_current_info",
        "description": "Search the web for current information about a topic",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _check_freshness_single(claim_text: str, verdict_label: str, verdict_confidence: float,
                              time_since_verified_days: int, model: str) -> dict:
    prompt = FRESHNESS_CHECK_PROMPT.format(
        claim_text=claim_text, verdict_label=verdict_label,
        verdict_confidence=verdict_confidence,
        time_since_verified_days=time_since_verified_days,
    )
    client = _llm_factory.make_llm_client()
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def _check_freshness_react(claim_text: str, verdict_label: str, verdict_confidence: float,
                            time_since_verified_days: int, model: str, tavily_api_key: str) -> dict:
    from tools.live_search_tool import search_live
    user_msg = FRESHNESS_CHECK_PROMPT.format(
        claim_text=claim_text, verdict_label=verdict_label,
        verdict_confidence=verdict_confidence,
        time_since_verified_days=time_since_verified_days,
    )
    messages = [{"role": "system", "content": _REACT_SYSTEM}, {"role": "user", "content": user_msg}]
    client = _llm_factory.make_llm_client()
    for _ in range(3):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=[_SEARCH_TOOL], tool_choice="auto", temperature=0,
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]})
            for tc in msg.tool_calls:
                query = json.loads(tc.function.arguments).get("query", claim_text)
                try:
                    results = search_live(query, api_key=tavily_api_key)
                    snippets = "\n".join(r.get("content", "")[:300] for r in results[:3])
                except Exception as e:
                    snippets = f"Search unavailable: {e}"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": snippets or "No results."})
        else:
            raw = (msg.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            return json.loads(raw)
    raise ValueError("ReAct freshness loop did not converge")


def check_freshness(claim_text: str, verdict_label: str, verdict_confidence: float,
                    last_verified_at: datetime, api_key: str, model: str) -> dict:
    from config import settings
    now = datetime.now(timezone.utc)
    if last_verified_at.tzinfo is None:
        last_verified_at = last_verified_at.replace(tzinfo=timezone.utc)
    days_old = (now - last_verified_at).days
    try:
        if settings.use_freshness_react:
            result = _check_freshness_react(claim_text, verdict_label, verdict_confidence,
                                             days_old, model, settings.tavily_api_key)
        else:
            result = _check_freshness_single(claim_text, verdict_label, verdict_confidence,
                                              days_old, model)
        return {
            "revalidate":     bool(result.get("revalidate", False)),
            "reason":         result.get("reason", ""),
            "claim_category": result.get("claim_category", "unknown"),
        }
    except Exception as e:
        logger.warning("freshness_tool failed (%s) — defaulting to revalidate=True", e)
        return {"revalidate": True, "reason": str(e), "claim_category": "unknown"}
