"""
agents/evaluator.py — Evaluator
====================================
Runs OUTSIDE the LangGraph graph — takes final state as input.

Contract:
  - evaluate_state(state, latency_seconds) → dict with keys:
      faithfulness, answer_relevancy, hallucination_rate,
      unverified_claim_count, source_count, report_length_chars, latency_seconds
  - hallucination_rate = unverified_claims / total_claims (0.0 if no report)
  - run_benchmark(pipeline_fn, n_runs) → runs 5 benchmark questions n_runs each,
      prints mean ± std per metric
  - pipeline.py calls evaluate_state(final_state, latency_seconds=latency)
    — this signature must stay stable.

Usage:
    from agents.evaluator import evaluate_state
    scores = evaluate_state(final_state, latency_seconds=12.5)

Run benchmark suite:
    python agents/evaluator.py
"""

from typing import Optional
from state import ResearchState, mock_state

BENCHMARK_QUESTIONS = [
    "What are the health effects of intermittent fasting?",
    "Does coffee improve cognitive performance?",
    "What causes the placebo effect?",
    "What are the environmental impacts of electric vehicles?",
    "Is social media use linked to depression in teenagers?",
]


def evaluate_state(
    state: ResearchState,
    latency_seconds: Optional[float] = None,
) -> dict:
    """
    TODO (Person 6): Implement real evaluation.
    - compute_ragas_scores(state): run RAGAS faithfulness + answer_relevancy
      (wrap in try/except — returns None values if ragas not installed)
    - compute_hallucination_rate(state): call LLM to count total claims,
      divide len(unverified_claims) by that count
    - Return all 7 keys in the dict below — pipeline.py and tests depend on them
    """
    # Mock output so pipeline runs end-to-end from Day 1
    return {
        "faithfulness": None,
        "answer_relevancy": None,
        "hallucination_rate": 0.0,
        "unverified_claim_count": len(state.get("unverified_claims", [])),
        "source_count": len(state.get("retrieved_passages", [])),
        "report_length_chars": len(state.get("final_report", "")),
        "latency_seconds": latency_seconds,
    }


def run_benchmark(pipeline_fn, n_runs: int = 3):
    """
    TODO (Person 6): Implement benchmark runner.
    - For each question in BENCHMARK_QUESTIONS, run pipeline_fn n_runs times
    - Call evaluate_state on each result
    - Print mean ± std for faithfulness, answer_relevancy,
      hallucination_rate, latency_seconds
    """
    print("[P6] STUB — benchmark runner not yet implemented")
    for q in BENCHMARK_QUESTIONS:
        print(f"  Would run: {q}")


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    state = mock_state()
    state["final_report"] = (
        "## Intermittent Fasting and Health\n\n"
        "Research shows IF leads to 3-8% weight loss [Source 1]. "
        "It also improves insulin sensitivity [Source 2].\n\n"
        "## Sources\n"
        "[1] IF and weight loss — https://pubmed.ncbi.nlm.nih.gov/example1\n"
        "[2] Metabolic Effects — https://www.nejm.org/example2"
    )
    state["unverified_claims"] = []

    scores = evaluate_state(state, latency_seconds=12.5)
    print("Scores:", json.dumps(scores, indent=2))