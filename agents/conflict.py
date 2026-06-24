"""
agents/conflict.py — Conflict Detector
==========================================
Reads:  state["retrieved_passages"]
Writes: state["conflict_report"], state["reasoning_trace"]

Contract:
  - conflict_report: ConflictReport {has_conflicts: bool, pairs: List[ConflictPair]}
  - ConflictPair: {passage_a, passage_b, verdict, confidence (0–1), explanation}
  - Only pairs with verdict=="contradict" AND confidence >= 0.75 are included
  - reasoning_trace: plain List[str] with exactly 1 new entry ("[P4] ...")
    LangGraph appends via operator.add.
  - route_on_conflict must remain in this file — pipeline.py imports it.

Run in isolation:
    python agents/conflict.py
"""

from state import ResearchState, ConflictReport, mock_state


def detect_conflicts(state: ResearchState) -> ResearchState:
    """
    TODO (Person 4): Implement pairwise conflict detection using Claude.
    - Generate all combinations(retrieved_passages, 2)
    - For each pair, call LLM with FEW_SHOT_SYSTEM prompt
    - Parse JSON: {verdict, confidence, explanation}
    - Keep pairs where verdict=="contradict" and confidence >= 0.75
    - Set has_conflicts = len(contradictions) > 0
    - Return exactly one reasoning_trace entry
    """
    # Mock output so pipeline runs end-to-end from Day 1
    conflict_report = ConflictReport(has_conflicts=False, pairs=[])
    passages = state["retrieved_passages"]
    n_pairs = len(passages) * (len(passages) - 1) // 2

    return {
        **state,
        "conflict_report": conflict_report,
        "reasoning_trace": [
            f"[P4] STUB — checked {n_pairs} pairs, no conflicts (mock)"
        ],
    }


def route_on_conflict(state: ResearchState) -> str:
    """LangGraph conditional edge router — do NOT stub this."""
    if state["conflict_report"]["has_conflicts"]:
        return "synthesize_warning"
    return "synthesize_normal"


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    state = mock_state()
    result = detect_conflicts(state)
    report = result["conflict_report"]
    print(f"\nhas_conflicts: {report['has_conflicts']}")
    print(f"Contradictions: {len(report['pairs'])}")
    print("Route →", route_on_conflict(result))
    print("Trace:", result["reasoning_trace"])