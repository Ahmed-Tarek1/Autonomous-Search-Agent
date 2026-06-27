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

import os
import re
import sys
from typing import Optional

import numpy as np
from datasets import Dataset
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness

# --- path fix so this runs from anywhere ---
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from state import ResearchState, mock_state
import yaml


#read config.yaml file
with open("configs.yaml", "r") as f:
    config = yaml.safe_load(f)
    
MAIN_MODEL = config["MAIN_MODEL"]
EMBEDDING_MODEL = config["EMBEDDING_MODEL"]

# ---------------------------------------------------------------------------
# Shared judge LLM + embeddings (initialised once at import time)
# ---------------------------------------------------------------------------
judge_llm = LangchainLLMWrapper(
    ChatGroq(model=MAIN_MODEL)
)

judge_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
)

# ---------------------------------------------------------------------------
# Benchmark questions
# ---------------------------------------------------------------------------
BENCHMARK_QUESTIONS = [
    "What are the health effects of intermittent fasting?",
    "Does coffee improve cognitive performance?",
    "What causes the placebo effect?",
    "What are the environmental impacts of electric vehicles?",
    "Is social media use linked to depression in teenagers?",
]
# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _run_ragas_in_thread(state: ResearchState) -> dict:
    """
    The actual RAGAS computation — always called inside a dedicated thread
    that owns a plain asyncio event loop.  This avoids the nest_asyncio /
    uvloop incompatibility that occurs when evaluate_state() is called from
    FastAPI's async context (which uses uvloop).
    """
    import asyncio
    # Give this thread its own vanilla asyncio loop so RAGAS / nest_asyncio
    # never sees uvloop.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        if not state.get("final_report") or not state.get("retrieved_passages"):
            print("Warning: empty report or passages — skipping RAGAS")
            return {"faithfulness": None, "answer_relevancy": None}

        data = Dataset.from_dict({
            "question": [state["question"]],
            "answer":   [state["final_report"]],
            "contexts": [[p["text"] for p in state["retrieved_passages"]]],
        })

        faith_metric = Faithfulness(llm=judge_llm)
        relevancy_metric = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)

        result = evaluate(data, metrics=[faith_metric, relevancy_metric])
        df = result.to_pandas()

        return {
            "faithfulness":     round(float(df["faithfulness"][0]), 3),
            "answer_relevancy": round(float(df["answer_relevancy"][0]), 3),
        }

    except Exception as e:
        print(f"RAGAS error: {e}")
        return {"faithfulness": None, "answer_relevancy": None}
    finally:
        loop.close()


def compute_ragas_scores(state: ResearchState) -> dict:
    """
    Run RAGAS faithfulness + answer_relevancy on the final state.
    Delegates to a background thread with its own asyncio loop to avoid
    the nest_asyncio / uvloop incompatibility when called from FastAPI.
    Returns None values if report/passages are missing or RAGAS fails.
    """
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_ragas_in_thread, state)
        return future.result()


def compute_hallucination_rate(state: ResearchState) -> float:
    """
    hallucination_rate = unverified_claims / total content sentences.
    Returns 0.0 if report is empty or no unverified claims exist.
    """
    unverified = len(state.get("unverified_claims", []))
    report     = state.get("final_report", "")

    if not report or unverified == 0:
        return 0.0

    sentences = re.split(r'(?<=[.!?])\s+', report.strip())
    content_sentences = [
        s for s in sentences
        if s and not s.startswith("#") and not s.startswith("[")
    ]
    total_claims = max(len(content_sentences), 1)
    return round(unverified / total_claims, 3)


# ---------------------------------------------------------------------------
# Public API — called by pipeline.py
# ---------------------------------------------------------------------------

def evaluate_state(
    state: ResearchState,
    latency_seconds: Optional[float] = None,
) -> dict:
    """
    Main entry point. Returns all 7 evaluation keys.
    pipeline.py signature must stay stable.
    """
    ragas_scores  = compute_ragas_scores(state)
    hallucination = compute_hallucination_rate(state)

    result = {
        "faithfulness":           ragas_scores["faithfulness"],
        "answer_relevancy":       ragas_scores["answer_relevancy"],
        "hallucination_rate":     hallucination,
        "unverified_claim_count": len(state.get("unverified_claims", [])),
        "source_count":           len(state.get("retrieved_passages", [])),
        "report_length_chars":    len(state.get("final_report", "")),
        "latency_seconds":        latency_seconds,
    }

    print("Eval scores:")
    for k, v in result.items():
        print(f"     {k}: {v}")

    return result


def run_benchmark(pipeline_fn, n_runs: int = 3):
    """
    Run all 5 benchmark questions n_runs times each.
    Prints mean ± std for faithfulness, answer_relevancy,
    hallucination_rate, and latency_seconds.
    """
    import time

    metrics = ["faithfulness", "answer_relevancy", "hallucination_rate", "latency_seconds"]
    all_results = {m: [] for m in metrics}

    for question in BENCHMARK_QUESTIONS:
        print(f"\nBenchmarking: {question}")
        for run in range(n_runs):
            start = time.time()
            final_state = pipeline_fn({"question": question})
            latency = round(time.time() - start, 2)

            scores = evaluate_state(final_state, latency_seconds=latency)
            for m in metrics:
                val = scores.get(m)
                if val is not None:
                    all_results[m].append(val)

    print("\n=== Benchmark Results ===")
    for m in metrics:
        vals = all_results[m]
        if vals:
            print(f"  {m}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
        else:
            print(f"  {m}: N/A")


# ---------------------------------------------------------------------------
# Local test — python agents/evaluator.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    state = mock_state()
    state["question"] = "What causes the placebo effect?"
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
    print("\nFinal JSON:")
    print(json.dumps(scores, indent=2))