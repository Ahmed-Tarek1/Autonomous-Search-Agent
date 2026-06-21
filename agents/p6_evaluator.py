"""
agents/p6_evaluator.py — Evaluator
====================================
Owner: Person 6
Runs OUTSIDE the LangGraph graph — takes final state as input.

Fix v2: reasoning_trace returned as plain list for operator.add.

Usage:
    from agents.p6_evaluator import evaluate_state
    scores = evaluate_state(final_state)

Run benchmark suite:
    python agents/p6_evaluator.py
"""

import json
import os
import time
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from state import ResearchState, mock_state

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY", ""),
)

COUNT_CLAIMS_SYSTEM = """Count the number of distinct factual claims in the following research report.
A factual claim is any statement that asserts something as fact (not definitions or section headers).
Return a single integer only."""

BENCHMARK_QUESTIONS = [
    "What are the health effects of intermittent fasting?",
    "Does coffee improve cognitive performance?",
    "What causes the placebo effect?",
    "What are the environmental impacts of electric vehicles?",
    "Is social media use linked to depression in teenagers?",
]


def _count_claims(report: str) -> int:
    try:
        response = llm.invoke([
            SystemMessage(content=COUNT_CLAIMS_SYSTEM),
            HumanMessage(content=report),
        ])
        return int(response.content.strip())
    except Exception:
        return len([s for s in report.split(".") if len(s.strip()) > 20])


def compute_hallucination_rate(state: ResearchState) -> float:
    report = state.get("final_report", "")
    if not report:
        return 0.0
    unverified = len(state.get("unverified_claims", []))
    total = _count_claims(report)
    return unverified / max(total, 1)


def compute_ragas_scores(state: ResearchState) -> dict:
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset

        data = Dataset.from_dict({
            "question": [state["question"]],
            "answer": [state["final_report"]],
            "contexts": [[p["text"] for p in state["retrieved_passages"]]],
        })
        scores = evaluate(data, metrics=[faithfulness, answer_relevancy])
        return {
            "faithfulness": float(scores["faithfulness"]),
            "answer_relevancy": float(scores["answer_relevancy"]),
        }
    except ImportError:
        print("  [P6] RAGAS not installed — using placeholder scores")
        return {"faithfulness": None, "answer_relevancy": None}
    except Exception as e:
        print(f"  [P6] RAGAS error: {e}")
        return {"faithfulness": None, "answer_relevancy": None}


def evaluate_state(
    state: ResearchState,
    latency_seconds: Optional[float] = None,
) -> dict:
    """Main evaluation function. Call after LangGraph pipeline exits."""
    print("  [P6] Running evaluation...")

    ragas = compute_ragas_scores(state)
    hallucination_rate = compute_hallucination_rate(state)

    scores = {
        "faithfulness": ragas["faithfulness"],
        "answer_relevancy": ragas["answer_relevancy"],
        "hallucination_rate": round(hallucination_rate, 4),
        "unverified_claim_count": len(state.get("unverified_claims", [])),
        "source_count": len(state.get("retrieved_passages", [])),
        "report_length_chars": len(state.get("final_report", "")),
        "latency_seconds": latency_seconds,
    }

    print(f"  [P6] Scores: {json.dumps(scores, indent=2)}")
    return scores


def run_benchmark(pipeline_fn, n_runs: int = 3):
    """
    Run all 5 benchmark questions through the pipeline n_runs times.
    Reports mean ± std for each metric.

    Args:
        pipeline_fn: callable that takes a question str and returns ResearchState
        n_runs: number of runs per question (for variance measurement)
    """
    import statistics

    results = {q: [] for q in BENCHMARK_QUESTIONS}

    for question in BENCHMARK_QUESTIONS:
        print(f"\n[P6] Benchmarking: {question}")
        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}")
            start = time.time()
            try:
                final_state = pipeline_fn(question)
                latency = time.time() - start
                scores = evaluate_state(final_state, latency_seconds=latency)
                results[question].append(scores)
            except Exception as e:
                print(f"  [P6] Pipeline error on run {run + 1}: {e}")

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    for question, runs in results.items():
        if not runs:
            print(f"\n{question[:60]}...\n  No successful runs.")
            continue
        print(f"\n{question[:70]}")
        for metric in ["faithfulness", "answer_relevancy", "hallucination_rate", "latency_seconds"]:
            values = [r[metric] for r in runs if r.get(metric) is not None]
            if values:
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                print(f"  {metric:25s}: {mean:.3f} ± {std:.3f}")

    return results


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    state = mock_state()
    state["final_report"] = (
        "## Intermittent Fasting and Health\n\n"
        "Research shows IF leads to 3-8% weight loss over 3-24 weeks [Source 1]. "
        "It also improves insulin sensitivity in overweight adults [Source 2].\n\n"
        "## Sources\n"
        "[1] IF and weight loss review — https://pubmed.ncbi.nlm.nih.gov/example1\n"
        "[2] Metabolic Effects — https://www.nejm.org/example2"
    )
    state["unverified_claims"] = []

    scores = evaluate_state(state, latency_seconds=12.5)
    print("\nFinal scores:", json.dumps(scores, indent=2))
