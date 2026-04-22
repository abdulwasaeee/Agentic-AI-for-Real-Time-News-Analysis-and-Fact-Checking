"""Live search tool — queries Tavily for current web evidence about a claim."""
import logging

from tavily import TavilyClient

logger = logging.getLogger(__name__)

_MIN_DISTINCT_DOMAINS = 3


def search_live(claim_text: str, api_key: str, max_results: int = 5) -> list[dict]:
    client = TavilyClient(api_key=api_key)
    results = _run_search(client, claim_text, max_results)
    if _count_distinct_domains(results) < _MIN_DISTINCT_DOMAINS:
        results = _run_search(client, f"fact check: {claim_text}", max_results + 3)
    return results


def _run_search(client: TavilyClient, query: str, max_results: int) -> list[dict]:
    try:
        response = client.search(query=query, max_results=max_results, search_depth="advanced")
        return response.get("results", [])
    except Exception as e:
        logger.error("Tavily search failed: %s", e)
        return []


def _count_distinct_domains(results: list[dict]) -> int:
    domains = set()
    for r in results:
        parts = r.get("url", "").split("/")
        if len(parts) >= 3:
            domains.add(parts[2])
    return len(domains)


def format_search_context(results: list[dict]) -> tuple[str, list[str]]:
    if not results:
        return "[LIVE SEARCH] No results found.", []
    lines = ["[LIVE SEARCH RESULTS]"]
    evidence_links: list[str] = []
    for r in results:
        url     = r.get("url", "")
        title   = r.get("title", "Untitled")
        content = (r.get("content") or r.get("snippet") or "")[:300]
        score   = r.get("score")
        score_str = f" (relevance: {score:.2f})" if score else ""
        lines.append(f"- [{title}]({url}){score_str}: {content}")
        if url:
            evidence_links.append(url)
    return "\n".join(lines), evidence_links
