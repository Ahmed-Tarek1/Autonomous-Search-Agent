"""
pipeline.py — LangGraph Pipeline
==================================
Owner: Person 1.
Wires all 6 nodes into the ResearchState graph.

Fix v2:
  - load_dotenv() called at top before any LangChain import
  - Graph built lazily (get_pipeline()) instead of at module import time —
    prevents all LLM clients from loading when tests merely import this file.

Usage:
    from pipeline import run_pipeline
    final_state = run_pipeline("What are the health effects of intermittent fasting?")
"""

from dotenv import load_dotenv
load_dotenv()  # must be before any langchain/anthropic import

import time
from langgraph.graph import StateGraph, END

from state import ResearchState
from agents import (
    decompose_query,
    search_agent,
    retrieve_passages,
    detect_conflicts,
    route_on_conflict,
    synthesize_report,
    evaluate_state,
)

# ---------------------------------------------------------------------------
# Lazy graph construction — avoids loading all LLM clients at import time
# ---------------------------------------------------------------------------

_pipeline = None


def get_pipeline():
    """Build and cache the compiled LangGraph pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    graph = StateGraph(ResearchState)

    graph.add_node("decompose",           decompose_query)    # P1
    graph.add_node("search",              search_agent)       # P2
    graph.add_node("retrieve",            retrieve_passages)  # P3
    graph.add_node("detect",              detect_conflicts)   # P4
    graph.add_node("synthesize_normal",   synthesize_report)  # P5 — no conflicts
    graph.add_node("synthesize_warning",  synthesize_report)  # P5 — with conflicts

    graph.set_entry_point("decompose")

    graph.add_edge("decompose", "search")
    graph.add_edge("search",    "retrieve")
    graph.add_edge("retrieve",  "detect")

    # Only conditional edge — driven by P4's conflict_report.has_conflicts flag
    graph.add_conditional_edges(
        "detect",
        route_on_conflict,
        {
            "synthesize_normal":  "synthesize_normal",
            "synthesize_warning": "synthesize_warning",
        },
    )

    graph.add_edge("synthesize_normal",  END)
    graph.add_edge("synthesize_warning", END)

    _pipeline = graph.compile()
    return _pipeline


def run_pipeline(question: str, run_eval: bool = True) -> ResearchState:
    """
    Run the full research pipeline on a question.

    Args:
        question:  The research question to answer.
        run_eval:  If True, run P6 evaluation after the graph exits.

    Returns:
        The final ResearchState with all fields populated.
    """
    initial_state = ResearchState(
        question=question,
        sub_questions=[],
        reasoning_trace=[],          # starts empty; operator.add accumulates entries
        search_results=[],
        retrieved_passages=[],
        conflict_report={"has_conflicts": False, "pairs": []},
        final_report="",
        citations=[],
        unverified_claims=[],
        eval_scores=None,
    )

    print(f"\n{'='*60}")
    print(f"RESEARCH PIPELINE: {question}")
    print(f"{'='*60}")

    start = time.time()
    final_state = get_pipeline().invoke(initial_state)
    latency = time.time() - start

    print(f"\n[pipeline] Completed in {latency:.1f}s")

    if run_eval:
        scores = evaluate_state(final_state, latency_seconds=latency)
        final_state = {**final_state, "eval_scores": scores}

    return final_state


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = run_pipeline(
        "What are the health effects of intermittent fasting?",
        run_eval=True,
    )

    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(result["final_report"])

    print("\nCITATIONS")
    for c in result["citations"]:
        print(f"  {c}")

    print(f"\nEVAL SCORES: {result['eval_scores']}")
    print(f"\nREASONING TRACE ({len(result['reasoning_trace'])} steps):")
    for entry in result["reasoning_trace"]:
        print(f"  {entry}")
