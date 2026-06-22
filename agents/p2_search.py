"""
agents/p2_search.py — Search Agent (ReAct loop)
================================================
Owner: Person 2
Reads:  state["sub_questions"], state["question"]
Writes: state["search_results"], state["reasoning_trace"]

Contract:
  - search_results: List[SearchResult] — no duplicate URLs
  - Each SearchResult: {url, title, snippet, source (domain)}
  - reasoning_trace: plain List[str], one entry per sub-question ("[P2] ...")
    LangGraph appends via operator.add.

Run in isolation:
    python agents/p2_search.py
"""

from state import ResearchState, SearchResult, mock_state


def search_agent(state: ResearchState) -> ResearchState:
    """
    TODO (Person 2): Implement ReAct loop using Tavily.
    - For each sub-question, run a search loop (max 3 iterations):
        1. Call web_search tool (Tavily, max_results=5)
        2. Deduplicate by URL across all sub-questions
        3. Ask LLM: STOP (if >= 3 results from >= 2 domains) or SEARCH: <refined query>
    - Collect all results into a single deduplicated list
    - Return one reasoning_trace entry per sub-question
    """
    # Mock output so pipeline runs end-to-end from Day 1
    return {
        **state,
        "search_results": [],
        "reasoning_trace": ["[P2] STUB — replace with real Tavily ReAct implementation"],
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