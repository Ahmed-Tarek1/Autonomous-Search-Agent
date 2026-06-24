"""
agents/search.py — Search Agent (ReAct loop)
================================================
Reads:  state["sub_questions"], state["question"]
Writes: state["search_results"], state["reasoning_trace"]

Contract:
  - search_results: List[SearchResult] — no duplicate URLs
  - Each SearchResult: {url, title, snippet, source (domain)}
  - reasoning_trace: plain List[str], one entry per sub-question ("[P2] ...")
    LangGraph appends via operator.add.

Run in isolation:
    python agents/search.py
"""
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, List, Tuple
from urllib.parse import urlparse

import yaml

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from state import ResearchState, SearchResult, mock_state


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "shared_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


try:
    configs = _load_config()
except (FileNotFoundError, yaml.YAMLError) as e:
    raise RuntimeError(f"Failed to load shared_config.yaml: {e}") from e

MAX_ITERATIONS     = configs.get("MAX_ITERATIONS", 5)
MIN_SOURCES        = configs.get("MIN_SOURCES", 3)
MIN_DOMAINS        = configs.get("MIN_DOMAINS", 2)
TAVILY_MAX_RESULTS = configs.get("TAVILY_MAX_RESULTS", 5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _field(result: Any, key: str) -> Any:
    """
    Access a field from a SearchResult regardless of whether it is a
    TypedDict (subscript) or a Pydantic model / dataclass (attribute).
    FIX #3: Eliminates TypeError when SearchResult is not a plain dict.
    """
    try:
        return result[key]
    except (TypeError, KeyError):
        return getattr(result, key)


def _truncate(text: str, max_len: int = 50) -> str:
    """Truncate with ellipsis only when necessary. FIX #11 (ellipsis regression)."""
    return textwrap.shorten(text, width=max_len, placeholder="...")


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _make_llm() -> ChatGroq:
    """
    Instantiate a fresh LLM client per call.
    """
    return ChatGroq(
        model=configs["MAIN_MODEL"],
        temperature=configs.get("SEARCH_TEMPERATURE", 0),
        api_key=os.getenv("GROQ_API_KEY", ""),
    )


# ---------------------------------------------------------------------------
# ReAct system prompt factory
# ---------------------------------------------------------------------------

def _make_react_system() -> str:
    """
    Build the ReAct prompt at call time.
    """
    return configs["REACT_SYSTEM"].format(
        min_sources=MIN_SOURCES, min_domains=MIN_DOMAINS
    )


# ---------------------------------------------------------------------------
# Tool — Tavily client
# ---------------------------------------------------------------------------

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def _plain_domain(url: str) -> str:
    """
    Extract a plain domain, stripping Markdown link syntax
    (e.g. '[example.com](https://example.com)') that Tavily occasionally returns.
    """
    url = _MD_LINK_RE.sub(r"\1", url).strip()
    return urlparse(url).netloc or url


@tool
def web_search(query: str) -> List[dict]:
    """Search the web for information. Returns title, url, and snippet per result."""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))
        response = client.search(query, max_results=TAVILY_MAX_RESULTS)
        return [
            {
                "url": r["url"],
                "title": r["title"],
                "snippet": r.get("content", ""),
                "source": _plain_domain(r["url"]),
            }
            for r in response.get("results", [])
        ]
    except Exception as e:
        print(f"  [search error] {e}")
        return []


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------

def _react_loop(sub_question: str) -> Tuple[List[SearchResult], List[str]]:
    """
    Run a ReAct loop for a single sub-question.
    Returns (results, error_notes) — error_notes surfaces in the trace.
    """
    llm = _make_llm()
    react_system = _make_react_system()

    all_results: List[SearchResult] = []
    error_notes: List[str] = []
    query = sub_question

    for iteration in range(MAX_ITERATIONS):
        print(f"    [iter {iteration + 1}] query: {query}")
        raw = web_search.invoke({"query": query})

        if not raw:
            note = f"iter {iteration + 1}: search returned no results for '{query}'"
            print(f"    [warn] {note}")
            error_notes.append(note)

        for r in raw:
            if not any(_field(x, "url") == r["url"] for x in all_results):
                all_results.append(SearchResult(**r))

        domains = {_field(r, "source") for r in all_results}
        stopping_prompt = (
            f"Sub-question: {sub_question}\n"
            f"Results so far: {len(all_results)} from domains: {list(domains)}"
        )

        decision = llm.invoke([
            SystemMessage(content=react_system),
            HumanMessage(content=stopping_prompt),
        ]).content.strip()

        print(f"    [decision] {decision}")

        if decision.upper() == "STOP":
            break
        elif decision.upper().startswith("SEARCH:"):
            query = decision[7:].strip()
        else:
            note = f"iter {iteration + 1}: unrecognised LLM decision: {decision!r}"
            print(f"    [warn] {note}")
            error_notes.append(note)
            break

    return all_results, error_notes


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def search_agent(state: ResearchState) -> dict:
    """
    LangGraph node: search_agent
    Runs a ReAct search loop per sub-question, deduplicates, writes results.
    """
    all_results: List[SearchResult] = []
    trace_entries: List[str] = []

    for sub_q in state["sub_questions"]:
        print(f"Searching for: {sub_q}")
        results, error_notes = _react_loop(sub_q)

        before = len(all_results)
        for r in results:
            if not any(_field(x, "url") == _field(r, "url") for x in all_results):
                all_results.append(r)

        added = len(all_results) - before
        domains = {_field(r, "source") for r in results}
        label = _truncate(sub_q)
        entry = f"'{label}' → {added} new results from {list(domains)}"
        if error_notes:
            entry += f" [errors: {'; '.join(error_notes)}]"
        trace_entries.append(entry)

    return {
        "search_results": all_results,
        "reasoning_trace": trace_entries,
    }


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    
    test_state: ResearchState = {
        **mock_state(),
        "sub_questions": [
            "What does research say about intermittent fasting and weight loss?",
        ],
        "search_results": [],
        "reasoning_trace": [],
    }

    result = search_agent(test_state)
    print(f"\nTotal results: {len(result['search_results'])}")
    for r in result["search_results"]:
        print(f"  - [{_field(r, 'source')}] {str(_field(r, 'title'))[:60]}")
    print("Trace:", result["reasoning_trace"])
