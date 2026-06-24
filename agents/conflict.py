"""
agents/conflict.py - Conflict Detector
==========================================
Reads:  state["retrieved_passages"]
Writes: state["conflict_report"], state["reasoning_trace"]

Contract:
  - conflict_report: ConflictReport {has_conflicts: bool, pairs: List[ConflictPair]}
  - ConflictPair: {passage_a, passage_b, verdict, confidence (0-1), explanation}
  - Only pairs with verdict=="contradict" AND confidence >= 0.75 are included
  - reasoning_trace: plain List[str] with exactly 1 new entry ("[P4] ...")
    LangGraph appends via operator.add.
  - route_on_conflict must remain in this file - pipeline.py imports it.

Agentic patterns used:
  - Few-shot prompting    -> Gemini classifies pairs with examples
  - Semantic pre-filter  -> Skip trivially unrelated pairs (cosine sim < 0.10)
  - Parallel LLM calls   -> asyncio.gather over all pairs for speed
  - Retry with backoff   -> Handles transient API errors gracefully
  - Structured output    -> JSON-mode response for reliable parsing

Run in isolation:
    python agents/conflict.py
"""

import os
import json
import time
import asyncio
import itertools
import re
from typing import List, Tuple

import sys
import os as _os
# Allow running directly as: python agents/conflict.py
if __name__ == "__main__" and "." not in sys.path:
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from state import ResearchState, ConflictReport, ConflictPair, Passage, mock_state

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFLICT_CONFIDENCE_THRESHOLD = 0.75   # Only keep pairs above this confidence
SEMANTIC_SIMILARITY_THRESHOLD = 0.10   # Skip pairs below this (likely truly unrelated)
                                        # Set low (0.10) because research snippets are short
                                        # and share few words even when on the same topic.
MAX_RETRIES = 3                         # LLM call retries on failure
RETRY_DELAY_SECONDS = 2.0              # Base delay for exponential backoff
GEMINI_MODEL = "gemini-3.1-flash-lite"      # Fast + affordable for classification tasks

# ---------------------------------------------------------------------------
# Few-shot system prompt
# This is the core "brain" of the conflict detector.
# Few-shot examples teach Gemini exactly what we mean by contradictions.
# ---------------------------------------------------------------------------

FEW_SHOT_SYSTEM = """You are a scientific fact-checking assistant. Your job is to determine whether two research passages CONTRADICT each other, AGREE with each other, or are UNRELATED.

Definitions:
- contradict: The passages make opposing factual claims about the same topic. Example: one says X increases Y, another says X decreases Y.
- agree: The passages make compatible or complementary claims about the same topic.
- unrelated: The passages discuss different topics or aspects with no direct comparison possible.

Respond ONLY with a valid JSON object - no explanation outside the JSON. Format:
{
  "verdict": "contradict" | "agree" | "unrelated",
  "confidence": <float 0.0 to 1.0>,
  "explanation": "<1-2 sentence explanation citing specific conflicting claims>"
}

--- EXAMPLES ---

EXAMPLE 1:
Passage A: "Intermittent fasting leads to significant weight loss of 3-8% over 3-24 weeks compared to baseline."
Passage B: "Randomized controlled trials show intermittent fasting produces equivalent weight loss to continuous caloric restriction with no meaningful difference."
Response:
{"verdict": "agree", "confidence": 0.72, "explanation": "Both passages confirm intermittent fasting causes weight loss. Passage B adds context that it is comparable to continuous restriction, but neither contradicts the other's core claim."}

EXAMPLE 2:
Passage A: "Intermittent fasting significantly improves insulin sensitivity in overweight adults after 12 weeks."
Passage B: "A 2023 meta-analysis of 14 RCTs found no statistically significant improvement in insulin sensitivity from intermittent fasting regimens compared to controls."
Response:
{"verdict": "contradict", "confidence": 0.91, "explanation": "Passage A claims IF improves insulin sensitivity while Passage B cites a meta-analysis finding no significant improvement - a direct factual contradiction on the same outcome measure."}

EXAMPLE 3:
Passage A: "Intermittent fasting may cause irritability, headaches, and difficulty concentrating during fasting windows."
Passage B: "Electric vehicles have lower lifetime carbon emissions than gasoline cars in most countries."
Response:
{"verdict": "unrelated", "confidence": 0.99, "explanation": "The passages discuss completely different topics: intermittent fasting side effects vs. EV environmental impact."}

EXAMPLE 4:
Passage A: "Low-carbohydrate diets produce faster short-term weight loss than low-fat diets."
Passage B: "Studies show low-fat diets are more effective for long-term weight management over 2+ years."
Response:
{"verdict": "contradict", "confidence": 0.82, "explanation": "The passages contradict on diet effectiveness for weight loss - one favors low-carb for short-term, the other favors low-fat for long-term, reflecting a genuine scientific tension."}

--- END EXAMPLES ---

Now classify the following pair of passages:"""


# ---------------------------------------------------------------------------
# Utility: Cosine similarity without heavy ML dependencies
# ---------------------------------------------------------------------------

def _cosine_similarity_simple(text_a: str, text_b: str) -> float:
    """
    Lightweight bag-of-words cosine similarity.
    Used as a cheap pre-filter to skip obviously unrelated pairs
    before spending LLM tokens on them.
    """
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
# Core LLM call - single pair classification using Gemini
# ---------------------------------------------------------------------------

def _classify_pair_sync(
    passage_a: Passage,
    passage_b: Passage,
    client,
) -> Tuple[str, float, str]:
    """
    Call Gemini to classify a passage pair.
    Returns (verdict, confidence, explanation).
    Retries up to MAX_RETRIES times on failure.
    """
    # Build the full prompt: system + user message combined
    # (Gemini uses a single prompt string, unlike Claude's system/user split)
    full_prompt = (
        FEW_SHOT_SYSTEM
        + "\n\n"
        + f"Passage A (source: {passage_a['source']}):\n"
        + f'"{passage_a["text"]}"\n\n'
        + f"Passage B (source: {passage_b['source']}):\n"
        + f'"{passage_b["text"]}"'
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_prompt,
            )
            raw = response.text.strip()

            # Robust JSON extraction - handle markdown code fences if present
            json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(raw)

            verdict = str(parsed.get("verdict", "unrelated")).lower()
            if verdict not in ("contradict", "agree", "unrelated"):
                verdict = "unrelated"

            confidence = float(parsed.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

            explanation = str(parsed.get("explanation", "No explanation provided."))

            return verdict, confidence, explanation

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
                continue
            return "unrelated", 0.0, f"Parse error after {MAX_RETRIES} attempts: {e}"

        except Exception as e:
            err_str = str(e).lower()
            # Surface auth/key errors loudly instead of silently failing
            if "api_key" in err_str or "api key" in err_str or "authentication" in err_str:
                raise RuntimeError(
                    "GEMINI_API_KEY not set or invalid. Create a .env file with:\n"
                    "  GEMINI_API_KEY=your-key-here\n"
                    "Get your key at: https://aistudio.google.com/app/apikey"
                ) from e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
                continue
            return "unrelated", 0.0, f"API error after {MAX_RETRIES} attempts: {e}"

    return "unrelated", 0.0, "Max retries exceeded."


# ---------------------------------------------------------------------------
# Async wrapper - enables parallel pair classification
# ---------------------------------------------------------------------------

async def _classify_pair_async(
    passage_a: Passage,
    passage_b: Passage,
    client,
    semaphore: asyncio.Semaphore,
) -> Tuple[Passage, Passage, str, float, str]:
    """
    Async wrapper around the sync LLM call.
    Uses a semaphore to limit concurrent API requests (avoid rate limits).
    """
    async with semaphore:
        # Run blocking I/O in a thread pool to keep event loop healthy
        loop = asyncio.get_event_loop()
        verdict, confidence, explanation = await loop.run_in_executor(
            None,   # Default thread pool
            _classify_pair_sync,
            passage_a,
            passage_b,
            client,
        )
    return passage_a, passage_b, verdict, confidence, explanation


# ---------------------------------------------------------------------------
# Main node function - detect_conflicts
# ---------------------------------------------------------------------------

def detect_conflicts(state: ResearchState) -> ResearchState:
    """
    P4 Conflict Detector - Pairwise LLM classification of retrieved passages.

    Algorithm:
    1. Generate all C(n,2) passage pairs
    2. Pre-filter: skip pairs with cosine similarity < SEMANTIC_SIMILARITY_THRESHOLD
       (they are likely unrelated topics - saves LLM tokens)
    3. Classify remaining pairs in parallel via asyncio
    4. Keep only pairs where verdict=="contradict" AND confidence >= threshold
    5. Assemble ConflictReport and return updated state

    Agentic patterns:
    - Few-shot prompting for reliable classification
    - Semantic pre-filtering (cheap heuristic before expensive LLM call)
    - Parallel async LLM calls (speed optimization)
    - Retry with exponential backoff (reliability)
    """
    passages = state["retrieved_passages"]
    n = len(passages)

    if n < 2:
        # Can't have conflicts with fewer than 2 passages
        return {
            **state,
            "conflict_report": ConflictReport(has_conflicts=False, pairs=[]),
            "reasoning_trace": ["[P4] Only 1 passage retrieved - no pairs to check."],
        }

    # --- Step 1: Initialize Gemini client ---
    try:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Create a .env file with:\n"
                "  GEMINI_API_KEY=your-key-here\n"
                "Get your key at: https://aistudio.google.com/app/apikey"
            )
        client = genai.Client(api_key=api_key)
    except ImportError:
        n_pairs = n * (n - 1) // 2
        return {
            **state,
            "conflict_report": ConflictReport(has_conflicts=False, pairs=[]),
            "reasoning_trace": [
                f"[P4] google-genai package not installed - checked {n_pairs} pairs, returning stub. "
                f"Run: pip install google-genai"
            ],
        }

    # --- Step 2: Generate all pairs ---
    all_pairs = list(itertools.combinations(passages, 2))
    total_pairs = len(all_pairs)

    # --- Step 3: Semantic pre-filter ---
    # Skip pairs that are clearly about different topics (saves LLM calls)
    filtered_pairs = []
    skipped_count = 0
    for pa, pb in all_pairs:
        sim = _cosine_similarity_simple(pa["text"], pb["text"])
        if sim >= SEMANTIC_SIMILARITY_THRESHOLD:
            filtered_pairs.append((pa, pb))
        else:
            skipped_count += 1

    if not filtered_pairs:
        return {
            **state,
            "conflict_report": ConflictReport(has_conflicts=False, pairs=[]),
            "reasoning_trace": [
                f"[P4] Checked {total_pairs} pairs; all {skipped_count} skipped by "
                f"semantic pre-filter (similarity < {SEMANTIC_SIMILARITY_THRESHOLD}). "
                f"No conflicts found."
            ],
        }

    # --- Step 4: Parallel LLM classification ---
    MAX_CONCURRENT = 3  # Respect API rate limits
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def run_all():
        tasks = [
            _classify_pair_async(pa, pb, client, semaphore)
            for pa, pb in filtered_pairs
        ]
        return await asyncio.gather(*tasks)

    # Run async event loop (compatible with both script and server contexts)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside an already-running loop (e.g., Jupyter, some servers)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, run_all())
                results = future.result()
        else:
            results = loop.run_until_complete(run_all())
    except RuntimeError:
        results = asyncio.run(run_all())

    # --- Step 5: Filter to contradictions above threshold ---
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

    # --- Step 6: Assemble ConflictReport ---
    has_conflicts = len(contradiction_pairs) > 0
    conflict_report = ConflictReport(
        has_conflicts=has_conflicts,
        pairs=contradiction_pairs,
    )

    # Build a human-readable summary for the reasoning trace
    verdict_summary = (
        f"{all_verdicts.count('contradict')} contradict, "
        f"{all_verdicts.count('agree')} agree, "
        f"{all_verdicts.count('unrelated')} unrelated"
    )
    trace_entry = (
        f"[P4] Checked {total_pairs} pairs "
        f"({skipped_count} pre-filtered, {len(filtered_pairs)} sent to LLM). "
        f"Verdicts: {verdict_summary}. "
        f"Confirmed conflicts (>={CONFLICT_CONFIDENCE_THRESHOLD} confidence): "
        f"{len(contradiction_pairs)}. "
        f"has_conflicts={has_conflicts}."
    )

    return {
        **state,
        "conflict_report": conflict_report,
        "reasoning_trace": [trace_entry],
    }


# ---------------------------------------------------------------------------
# Conditional edge router - DO NOT STUB (pipeline.py imports this)
# ---------------------------------------------------------------------------

def route_on_conflict(state: ResearchState) -> str:
    """
    LangGraph conditional edge router.
    Returns the name of the next node based on conflict detection result.
    """
    if state["conflict_report"]["has_conflicts"]:
        return "synthesize_warning"
    return "synthesize_normal"