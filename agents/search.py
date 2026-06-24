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
import sys
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from state import ResearchState, SearchResult, mock_state


def load_variables():
    config_path = Path(__file__).resolve().parent.parent / "shared_config.yaml"
    with open(config_path, "r") as file:
        configs = yaml.safe_load(file)
    load_dotenv()
    return configs


configs = load_variables()

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------
llm = ChatGroq(
    model=configs["MAIN_MODEL"],
    temperature=configs["SEARCH_TEMPERATURE"],
    api_key=os.getenv("GROQ_API_KEY", ""),
)

MAX_ITERATIONS = configs["MAX_ITERATIONS"]
MIN_SOURCES = configs["MIN_SOURCES"]
MIN_DOMAINS = configs["MIN_DOMAINS"]
TAVILY_MAX_RESULTS = configs["TAVILY_MAX_RESULTS"]

# ---------------------------------------------------------------------------
# Tool — Tavily client
# ---------------------------------------------------------------------------

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
                "source": urlparse(r["url"]).netloc,
            }
            for r in response.get("results", [])
        ]
    except Exception as e:
        print(f"  [search error] {e}")
        return []
    

REACT_SYSTEM = configs["REACT_SYSTEM"].format(
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
        else:
            print(f"    [warn] unrecognized decision, stopping early: {decision!r}")
            break

    return all_results


def search_agent(state: ResearchState) -> ResearchState:
    """
    LangGraph node: search_agent
    Runs a ReAct search loop per sub-question, deduplicates, writes results.
    """
    all_results: List[SearchResult] = []
    trace_entries = []

    for sub_q in state["sub_questions"]:
        print(f"Searching for: {sub_q}")
        results = _react_loop(sub_q)
        before = len(all_results)

        for r in results:
            if not any(x["url"] == r["url"] for x in all_results):
                all_results.append(r)

        added = len(all_results) - before
        domains = {r["source"] for r in results}
        trace_entries.append(
            f"'{sub_q[:50]}...' → {added} new results from {list(domains)}"
        )

    return {
        "search_results": all_results,
        "reasoning_trace": trace_entries,
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
    print("Trace:", result["reasoning_trace"])
