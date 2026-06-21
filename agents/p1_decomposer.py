"""
agents/p1_decomposer.py — Query Decomposer + Orchestrator
==========================================================
Owner: Person 1
Reads:  state["question"]
Writes: state["sub_questions"], state["reasoning_trace"]

Fix v2: reasoning_trace returned as a plain list — LangGraph appends it
        via the operator.add reducer defined in state.py.

Run in isolation:
    python agents/p1_decomposer.py
"""

import json
import os
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from state import ResearchState, mock_state

# ---------------------------------------------------------------------------
# LLM — temperature=0 for deterministic decomposition
# ---------------------------------------------------------------------------
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY", ""),
)

DECOMPOSE_SYSTEM = """You are a research query decomposer.
Given a broad research question, break it into 3-5 specific, focused sub-questions.
Each sub-question must be:
- Independently searchable on the web
- Non-overlapping with the others
- Concrete enough to return useful sources

Return a JSON array of strings only. No explanation, no markdown fences."""

CRITIQUE_SYSTEM = """You are a research query reviewer.
Given a list of sub-questions, identify any that are too vague to search for.
Rewrite vague ones to be more specific.
Return the improved list as a JSON array of strings only."""


def decompose_query(state: ResearchState) -> ResearchState:
    """
    LangGraph node — P1.
    Decomposes the research question into focused sub-questions.
    """
    question = state["question"]

    # Step 1: Generate sub-questions
    response = llm.invoke([
        SystemMessage(content=DECOMPOSE_SYSTEM),
        HumanMessage(content=f"Research question: {question}"),
    ])
    sub_questions: List[str] = json.loads(response.content)

    # Step 2: Simple deduplication (exact match, case-insensitive)
    seen = set()
    deduped = []
    for q in sub_questions:
        key = q.lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(q)
    sub_questions = deduped

    # Step 3: Self-critique pass — separate LLM call
    critique_response = llm.invoke([
        SystemMessage(content=CRITIQUE_SYSTEM),
        HumanMessage(content=f"Sub-questions: {json.dumps(sub_questions)}"),
    ])
    sub_questions = json.loads(critique_response.content)

    trace_entry = (
        f"[P1] Decomposed '{question}' into {len(sub_questions)} sub-questions: "
        + ", ".join(f'"{q}"' for q in sub_questions)
    )

    return {
        **state,
        "sub_questions": sub_questions,
        # Return only the new entry — LangGraph appends via operator.add
        "reasoning_trace": [trace_entry],
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
