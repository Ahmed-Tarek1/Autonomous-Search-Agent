"""
agents/decomposer.py — Query Decomposer + Orchestrator
==========================================================
Reads:  state["question"]
Writes: state["sub_questions"], state["reasoning_trace"]

Contract:
  - sub_questions: List[str], 3–5 items, each independently searchable
  - reasoning_trace: plain List[str] with exactly 1 new entry ("[P1] ...")
    LangGraph appends it via the operator.add reducer in state.py.

Run in isolation:
    python agents/decomposer.py
"""

from state import ResearchState, mock_state


def decompose_query(state: ResearchState) -> ResearchState:
    """
    TODO (Person 1): Implement query decomposition using Claude.
    - Call LLM with DECOMPOSE_SYSTEM prompt to generate 3–5 sub-questions
    - Deduplicate (case-insensitive exact match)
    - Run a self-critique LLM pass to sharpen vague sub-questions
    - Return exactly one reasoning_trace entry: "[P1] Decomposed '...' into N sub-questions: ..."
    """
    # Mock output so pipeline runs end-to-end from Day 1
    question = state["question"]
    sub_questions = [
        f"What does research say about {question.lower().rstrip('?')} and health outcomes?",
        f"What are the mechanisms behind {question.lower().rstrip('?')}?",
        f"What are the risks or side effects related to {question.lower().rstrip('?')}?",
        f"What do meta-analyses conclude about {question.lower().rstrip('?')}?",
    ]
    return {
        **state,
        "sub_questions": sub_questions,
        "reasoning_trace": [f"[P1] STUB — decomposed into {len(sub_questions)} sub-questions"],
    }


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    state = mock_state()
    state["question"] = "What are the health effects of intermittent fasting?"
    state["sub_questions"] = []
    state["reasoning_trace"] = []

    result = decompose_query(state)
    print("Sub-questions:")
    for i, q in enumerate(result["sub_questions"], 1):
        print(f"  {i}. {q}")
    print("\nNew trace entry:", result["reasoning_trace"])