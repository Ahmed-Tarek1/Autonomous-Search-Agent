"""
agents/p4_conflict.py — Conflict Detector
==========================================
Owner: Person 4
Reads:  state["retrieved_passages"]
Writes: state["conflict_report"], state["reasoning_trace"]

Fix v2: reasoning_trace returned as a plain list for operator.add reducer.

Run in isolation:
    python agents/p4_conflict.py
"""

import json
import os
from itertools import combinations
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from state import (
    ResearchState, Passage, ConflictPair, ConflictReport, mock_state
)

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY", ""),
)

CONFIDENCE_THRESHOLD = 0.75

FEW_SHOT_SYSTEM = """You are a fact-checking agent that classifies whether two passages agree, contradict, or are unrelated.

Return JSON only — no markdown fences, no explanation outside the JSON:
{"verdict": "agree"|"contradict"|"unrelated", "confidence": float, "explanation": str}

Examples:

Passage A: "Coffee improves short-term memory in adults."
Passage B: "Studies confirm caffeine enhances memory consolidation."
{"verdict": "agree", "confidence": 0.92, "explanation": "Both support cognitive benefits of caffeine."}

Passage A: "Intermittent fasting consistently reduces body weight."
Passage B: "Meta-analyses show no significant weight loss advantage of IF over continuous caloric restriction."
{"verdict": "contradict", "confidence": 0.88, "explanation": "A claims IF reduces weight; B disputes the advantage vs alternatives."}

Passage A: "Exercise improves cardiovascular health."
Passage B: "Vitamin D deficiency is common in northern latitudes."
{"verdict": "unrelated", "confidence": 0.97, "explanation": "Topics are entirely different."}"""


def _classify_pair(passage_a: Passage, passage_b: Passage) -> ConflictPair:
    prompt = (
        f"Passage A: {passage_a['text']}\n"
        f"Passage B: {passage_b['text']}"
    )
    response = llm.invoke([
        SystemMessage(content=FEW_SHOT_SYSTEM),
        HumanMessage(content=prompt),
    ])

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        content = response.content.strip().strip("```json").strip("```").strip()
        result = json.loads(content)

    return ConflictPair(
        passage_a=passage_a,
        passage_b=passage_b,
        verdict=result["verdict"],
        confidence=float(result["confidence"]),
        explanation=result["explanation"],
    )


def detect_conflicts(state: ResearchState) -> ResearchState:
    """LangGraph node — P4."""
    passages = state["retrieved_passages"]
    pairs = list(combinations(passages, 2))
    print(f"  [P4] Checking {len(pairs)} passage pairs for conflicts")

    contradiction_pairs: List[ConflictPair] = []

    for pa, pb in pairs:
        pair_result = _classify_pair(pa, pb)
        print(f"    [{pair_result['verdict']} | {pair_result['confidence']:.2f}] "
              f"{pa['source']} vs {pb['source']}")

        if (pair_result["verdict"] == "contradict"
                and pair_result["confidence"] >= CONFIDENCE_THRESHOLD):
            contradiction_pairs.append(pair_result)

    has_conflicts = len(contradiction_pairs) > 0
    conflict_report = ConflictReport(
        has_conflicts=has_conflicts,
        pairs=contradiction_pairs,
    )

    trace_entry = (
        f"[P4] Checked {len(pairs)} pairs — "
        f"found {len(contradiction_pairs)} contradiction(s). "
        f"has_conflicts={has_conflicts}"
    )

    return {
        **state,
        "conflict_report": conflict_report,
        "reasoning_trace": [trace_entry],  # operator.add handles appending
    }


def route_on_conflict(state: ResearchState) -> str:
    """LangGraph conditional edge router."""
    if state["conflict_report"]["has_conflicts"]:
        return "synthesize_warning"
    return "synthesize_normal"


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    state = mock_state()
    state["retrieved_passages"].append({
        "text": "Multiple meta-analyses found no significant weight loss advantage of intermittent fasting compared to continuous caloric restriction.",
        "url": "https://example-journal.org/meta-analysis",
        "title": "IF vs Continuous Restriction: A Meta-Analysis",
        "score": 0.82,
        "source": "example-journal.org",
    })

    result = detect_conflicts(state)
    report = result["conflict_report"]
    print(f"\nhas_conflicts: {report['has_conflicts']}")
    print(f"Contradictions found: {len(report['pairs'])}")
    for pair in report["pairs"]:
        print(f"  - {pair['explanation']}")
    print("Route →", route_on_conflict(result))
