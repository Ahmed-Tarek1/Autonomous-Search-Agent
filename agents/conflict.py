"""
agents/conflict.py - Conflict Detector
==========================================
Reads:  state["retrieved_passages"]
Writes: state["conflict_report"], state["reasoning_trace"]

Contract:
  - conflict_report: ConflictReport {has_conflicts: bool, pairs: List[ConflictPair]}
  - ConflictPair: {passage_a, passage_b, verdict, confidence (0-1), explanation}
  - Only pairs with verdict=="contradict" AND confidence >= CONFLICT_CONFIDENCE_THRESHOLD are included
  - reasoning_trace: plain List[str] with exactly 1 new entry ("...")
    LangGraph appends via operator.add.
  - route_on_conflict must remain in this file - pipeline.py imports it.

Agentic patterns used:
  - Few-shot prompting    -> LLM classifies pairs with labeled examples (conflict_prompt.py)
  - Semantic pre-filter  -> Skip trivially unrelated pairs (bag-of-words cosine similarity)
  - Parallel LLM calls   -> asyncio.gather over all pairs for speed
  - Retry with backoff   -> Handled by LLMCaller (max_retries=5)
  - Structured output    -> JSON response for reliable parsing

Run in isolation:
    python agents/conflict.py
"""

import os
import sys
import json
import time
import asyncio
import itertools
import re
from typing import List, Tuple
import yaml

# Allow running directly as: python agents/conflict.py
# if __name__ == "__main__" and "." not in sys.path:
    # sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from state import ResearchState, ConflictReport, ConflictPair, Passage, mock_state
from agents.prompts.conflict_prompt import CONFLICT_DETECTOR_PROMPT
from helpers.llm_caller import LLMCaller


# ---------------------------------------------------------------------------
# Load shared configuration from configs.yaml
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

configs = _load_config()

CONFLICT_CONFIDENCE_THRESHOLD = configs["CONFLICT_CONFIDENCE_THRESHOLD"]
CONFLICT_SIMILARITY_THRESHOLD = configs["CONFLICT_SIMILARITY_THRESHOLD"]
CONFLICT_MAX_RETRIES          = configs["CONFLICT_MAX_RETRIES"]
CONFLICT_RETRY_DELAY          = configs["CONFLICT_RETRY_DELAY"]
CONFLICT_MAX_CONCURRENT       = configs["CONFLICT_MAX_CONCURRENT"]


# ---------------------------------------------------------------------------
# Initialize the shared LLM caller (Groq) - same pattern as decomposer.py
# ---------------------------------------------------------------------------

conflict_llm: LLMCaller = LLMCaller(
    api_key=os.getenv("GROQ_API_KEY"),
    model=configs["MAIN_MODEL"],
    system_prompt=CONFLICT_DETECTOR_PROMPT,
    identifier="ConflictDetector",
    verbose=False,
)


# ---------------------------------------------------------------------------
# Utility: Lightweight cosine similarity (no ML dependencies)
# Used as a cheap pre-filter before spending LLM tokens on a pair.
# ---------------------------------------------------------------------------

def _cosine_similarity_simple(text_a: str, text_b: str) -> float:
    """Bag-of-words cosine similarity to detect obviously unrelated passage pairs."""
    def tokenize(text: str) -> dict:
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        freq: dict = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        return freq

    vec_a = tokenize(text_a)
    vec_b = tokenize(text_b)

    if not vec_a or not vec_b:
        return 0.0

    shared = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[w] * vec_b[w] for w in shared)
    norm_a = sum(v ** 2 for v in vec_a.values()) ** 0.5
    norm_b = sum(v ** 2 for v in vec_b.values()) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Core LLM call - classify a single pair using LLMCaller
# ---------------------------------------------------------------------------

def _classify_pair_sync(
    passage_a: Passage,
    passage_b: Passage,
) -> Tuple[str, float, str]:
    """
    Call Groq via LLMCaller to classify one passage pair.
    Returns (verdict, confidence, explanation).
    LLMCaller already handles retries internally (max_retries=5).
    """
    global conflict_llm

    for attempt in range(CONFLICT_MAX_RETRIES):
        try:
            raw = conflict_llm.call(
                source_a=passage_a["source"],
                text_a=passage_a["text"],
                source_b=passage_b["source"],
                text_b=passage_b["text"],
            )

            # Robust JSON extraction - handles markdown code fences if present
            json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(raw)

            verdict = str(parsed.get("verdict", "unrelated")).lower()
            if verdict not in ("contradict", "agree", "unrelated"):
                verdict = "unrelated"

            confidence = float(parsed.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))   # Clamp to [0, 1]

            explanation = str(parsed.get("explanation", "No explanation provided."))

            return verdict, confidence, explanation

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt < CONFLICT_MAX_RETRIES - 1:
                time.sleep(CONFLICT_RETRY_DELAY * (attempt + 1))
                continue
            return "unrelated", 0.0, f"Parse error after {CONFLICT_MAX_RETRIES} attempts: {e}"

        except Exception as e:
            if attempt < CONFLICT_MAX_RETRIES - 1:
                time.sleep(CONFLICT_RETRY_DELAY * (attempt + 1))
                continue
            return "unrelated", 0.0, f"LLM error after {CONFLICT_MAX_RETRIES} attempts: {e}"

    return "unrelated", 0.0, "Max retries exceeded."


# ---------------------------------------------------------------------------
# Async wrapper - enables parallel pair classification
# ---------------------------------------------------------------------------

async def _classify_pair_async(
    passage_a: Passage,
    passage_b: Passage,
    semaphore: asyncio.Semaphore,
) -> Tuple[Passage, Passage, str, float, str]:
    """
    Async wrapper around the sync LLM call.
    Uses a semaphore to limit concurrent API requests (avoid rate limits).
    """
    async with semaphore:
        loop = asyncio.get_event_loop()
        verdict, confidence, explanation = await loop.run_in_executor(
            None,   # Default thread pool executor
            _classify_pair_sync,
            passage_a,
            passage_b,
        )
    return passage_a, passage_b, verdict, confidence, explanation


# ---------------------------------------------------------------------------
# Main node function - detect_conflicts
# ---------------------------------------------------------------------------

def detect_conflicts(state: ResearchState) -> ResearchState:
    """Classify all passage pairs for contradictions and return a ConflictReport."""
    passages = state["retrieved_passages"]
    n = len(passages)

    if n < 2:
        return {
            "conflict_report": ConflictReport(has_conflicts=False, pairs=[]),
            "reasoning_trace": ["Only 1 passage retrieved - no pairs to check."],
        }

    # --- Step 1: Generate all C(n,2) pairs ---
    all_pairs = list(itertools.combinations(passages, 2))
    total_pairs = len(all_pairs)

    # --- Step 2: Semantic pre-filter ---
    # Skip pairs that share fewer than CONFLICT_SIMILARITY_THRESHOLD word overlap
    filtered_pairs = []
    skipped_count = 0
    for pa, pb in all_pairs:
        sim = _cosine_similarity_simple(pa["text"], pb["text"])
        if sim >= CONFLICT_SIMILARITY_THRESHOLD:
            filtered_pairs.append((pa, pb))
        else:
            skipped_count += 1

    if not filtered_pairs:
        return {
            "conflict_report": ConflictReport(has_conflicts=False, pairs=[]),
            "reasoning_trace": [
                f"Checked {total_pairs} pairs; all {skipped_count} skipped by "
                f"semantic pre-filter (similarity < {CONFLICT_SIMILARITY_THRESHOLD}). "
                f"No conflicts found."
            ],
        }

    # --- Step 3: Parallel LLM classification ---
    semaphore = asyncio.Semaphore(CONFLICT_MAX_CONCURRENT)

    async def run_all():
        tasks = [
            _classify_pair_async(pa, pb, semaphore)
            for pa, pb in filtered_pairs
        ]
        return await asyncio.gather(*tasks)

    # Compatible with both script and server event loop contexts
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, run_all())
                results = future.result()
        else:
            results = loop.run_until_complete(run_all())
    except RuntimeError:
        results = asyncio.run(run_all())

    # --- Step 4: Filter to confirmed contradictions ---
    contradiction_pairs: List[ConflictPair] = []
    all_verdicts = []

    for pa, pb, verdict, confidence, explanation in results:
        all_verdicts.append(verdict)
        if verdict == "contradict" and confidence >= CONFLICT_CONFIDENCE_THRESHOLD:
            contradiction_pairs.append(
                ConflictPair(
                    passage_a=pa,
                    passage_b=pb,
                    verdict=verdict,
                    confidence=confidence,
                    explanation=explanation,
                )
            )

    # --- Step 5: Assemble ConflictReport ---
    has_conflicts = len(contradiction_pairs) > 0
    conflict_report = ConflictReport(has_conflicts=has_conflicts, pairs=contradiction_pairs)

    verdict_summary = (
        f"{all_verdicts.count('contradict')} contradict, "
        f"{all_verdicts.count('agree')} agree, "
        f"{all_verdicts.count('unrelated')} unrelated"
    )
    trace_entry = (
        f"Checked {total_pairs} pairs "
        f"({skipped_count} pre-filtered, {len(filtered_pairs)} sent to LLM). "
        f"Verdicts: {verdict_summary}. "
        f"Confirmed conflicts (>={CONFLICT_CONFIDENCE_THRESHOLD} confidence): "
        f"{len(contradiction_pairs)}. "
        f"has_conflicts={has_conflicts}."
    )

    return {
        "conflict_report": conflict_report,
        "reasoning_trace": [trace_entry],
    }


# ---------------------------------------------------------------------------
# Conditional edge router - DO NOT STUB (pipeline.py imports this)
# ---------------------------------------------------------------------------

def route_on_conflict(state: ResearchState) -> str:
    """LangGraph conditional edge router - determines which synthesizer runs."""
    if state["conflict_report"]["has_conflicts"]:
        return "synthesize_warning"
    return "synthesize_normal"


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("CONFLICT DETECTOR - Isolation Test")
    print("=" * 60)
    print(f"Model:      {configs['MAIN_MODEL']}")
    print(f"Confidence threshold: {CONFLICT_CONFIDENCE_THRESHOLD}")
    print(f"Similarity threshold: {CONFLICT_SIMILARITY_THRESHOLD}")

    state = mock_state()
    result = detect_conflicts(state)
    report = result["conflict_report"]
    print(f"\n[Test] Mock state result:")
    print(f"  has_conflicts:       {report['has_conflicts']}")
    print(f"  Contradiction pairs: {len(report['pairs'])}")
    print(f"  Route ->             {route_on_conflict(result)}")
    print(f"  Trace:               {result['reasoning_trace'][-1]}")
