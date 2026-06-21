"""
agents/p2_search.py — Search Agent (ReAct loop)
================================================
Owner: Person 2
Reads:  state["sub_questions"], state["question"]
Writes: state["search_results"], state["reasoning_trace"]

Fix v2: TavilyClient instantiated lazily inside the tool function,
        not at module import time — prevents crash when key is missing.

Run in isolation:
    python agents/p2_search.py
"""

import os
from typing import List
from urllib.parse import urlparse

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

from state import ResearchState, SearchResult, mock_state

# ---------------------------------------------------------------------------
# LLM client — instantiated at import (key must exist, but no network call yet)
# ---------------------------------------------------------------------------
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY", ""),
)

MAX_ITERATIONS = 3
MIN_SOURCES = 3
MIN_DOMAINS = 2

# ---------------------------------------------------------------------------
# Tool — Tavily client created lazily inside the function body
# so importing this module never raises even if TAVILY_API_KEY is unset.
# ---------------------------------------------------------------------------

@tool
def web_search(query: str) -> List[dict]:
    """Search the web for information. Returns title, url, and snippet per result."""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))
        response = client.search(query, max_results=5)
        return [
            {
                "url": r["url"],
                "title": r["title"],
                "snippet": r.get("content", ""),
                "source": urlparse(r["url"]).netloc,
            }
            for r in response.get("results", [])
        ]
    except Exception as e:
        print(f"  [search error] {e}")
        return []


REACT_SYSTEM = """You are a search agent operating in a ReAct loop.
For each step you receive:
- The sub-question you are researching
- The results found so far (count and domains)

Respond with exactly one of:
- STOP  (if you have >= {min_sources} results from >= {min_domains} different domains)
- SEARCH: <your refined query>  (to run another search)

No explanation. No other text.""".format(
    min_sources=MIN_SOURCES, min_domains=MIN_DOMAINS
)


def _react_loop(sub_question: str) -> List[SearchResult]:
    """Run a ReAct loop for a single sub-question. Returns search results."""
    all_results: List[SearchResult] = []
    query = sub_question

    for iteration in range(MAX_ITERATIONS):
        print(f"    [iter {iteration + 1}] query: {query}")
        raw = web_search.invoke({"query": query})

        for r in raw:
            if not any(x["url"] == r["url"] for x in all_results):
                all_results.append(SearchResult(**r))

        domains = {r["source"] for r in all_results}
        stopping_prompt = (
            f"Sub-question: {sub_question}\n"
            f"Results so far: {len(all_results)} from domains: {list(domains)}"
        )

        decision = llm.invoke([
            SystemMessage(content=REACT_SYSTEM),
            HumanMessage(content=stopping_prompt),
        ]).content.strip()

        print(f"    [decision] {decision}")

        if decision.upper() == "STOP":
            break
        if decision.upper().startswith("SEARCH:"):
            query = decision[7:].strip()

    return all_results


def search_agent(state: ResearchState) -> ResearchState:
    """
    LangGraph node — P2.
    Runs a ReAct search loop per sub-question, deduplicates, writes results.
    """
    all_results: List[SearchResult] = []
    trace_entries = []

    for sub_q in state["sub_questions"]:
        print(f"  [P2] Searching for: {sub_q}")
        results = _react_loop(sub_q)
        before = len(all_results)

        for r in results:
            if not any(x["url"] == r["url"] for x in all_results):
                all_results.append(r)

        added = len(all_results) - before
        domains = {r["source"] for r in results}
        trace_entries.append(
            f"[P2] '{sub_q[:50]}...' → {added} new results from {list(domains)}"
        )

    return {
        **state,
        "search_results": all_results,
        "reasoning_trace": trace_entries,  # LangGraph appends via operator.add
    }


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    state = mock_state()
    state["sub_questions"] = [
        "What does research say about intermittent fasting and weight loss?",
    ]
    state["search_results"] = []

    result = search_agent(state)
    print(f"\nTotal results: {len(result['search_results'])}")
    for r in result["search_results"]:
        print(f"  - [{r['source']}] {r['title'][:60]}")
