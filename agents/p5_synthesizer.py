"""
agents/p5_synthesizer.py — Synthesizer + Report Generator
==========================================================
Owner: Person 5
Reads:  state["question"], state["retrieved_passages"], state["conflict_report"]
Writes: state["final_report"], state["citations"], state["unverified_claims"]

Fix v2: Passage compression replaced with simple 500-char truncation —
        saves 5 serial LLM calls (~30s latency reduction).
        reasoning_trace returned as plain list for operator.add reducer.

Run in isolation:
    python agents/p5_synthesizer.py
"""

import json
import os
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from state import ResearchState, Passage, mock_state

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0.3,
    api_key=os.getenv("ANTHROPIC_API_KEY", ""),
)
llm_check = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY", ""),
)

SYNTHESIS_SYSTEM_BASE = """You are a research synthesizer. Write a structured, well-organized report.

Rules (follow strictly):
- Every factual claim MUST be cited inline as [Source N]
- Never assert anything not found in the provided passages
- Use markdown formatting with ## section headers
- End with a ## Sources section listing all cited sources
- Be precise and academic in tone"""

SYNTHESIS_SYSTEM_CONFLICT = SYNTHESIS_SYSTEM_BASE + """

IMPORTANT: The sources contain conflicting evidence. You MUST include a
## Conflicting Evidence section that explicitly presents both sides and
explains the disagreement. Do not take a side — present both views fairly."""

SELF_CHECK_SYSTEM = """You are a fact-checker reviewing a research report.
List every factual claim in the report that does NOT have an inline [Source N] citation.
Return a JSON array of strings. Each string is one uncited claim.
Return [] if all claims are properly cited."""


def _build_passages_block(passages: List[Passage]) -> str:
    """
    Format passages as a numbered block for the synthesis prompt.
    Fix v2: simple 500-char truncation instead of per-passage LLM compression
            — saves 5 serial API calls and ~30s of latency.
    """
    lines = []
    for i, p in enumerate(passages, 1):
        text = p["text"][:500].strip()
        lines.append(f"[Source {i}] {p['title']} ({p['source']})\n{text}")
    return "\n\n".join(lines)


def _build_citations(passages: List[Passage]) -> List[str]:
    return [
        f"[{i}] {p['title']} — {p['url']}"
        for i, p in enumerate(passages, 1)
    ]


def synthesize_report(state: ResearchState) -> ResearchState:
    """LangGraph node — P5 (handles both conflict and non-conflict paths)."""
    passages = state["retrieved_passages"]
    conflict_report = state["conflict_report"]
    question = state["question"]

    passages_block = _build_passages_block(passages)

    if conflict_report["has_conflicts"]:
        system_prompt = SYNTHESIS_SYSTEM_CONFLICT
        conflict_summary = "\n".join(
            f"- {pair['explanation']}" for pair in conflict_report["pairs"]
        )
        user_prompt = (
            f"Research question: {question}\n\n"
            f"Conflicting evidence found:\n{conflict_summary}\n\n"
            f"Sources:\n{passages_block}"
        )
    else:
        system_prompt = SYNTHESIS_SYSTEM_BASE
        user_prompt = (
            f"Research question: {question}\n\n"
            f"Sources:\n{passages_block}"
        )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    final_report = response.content.strip()

    # Self-check pass
    check_response = llm_check.invoke([
        SystemMessage(content=SELF_CHECK_SYSTEM),
        HumanMessage(content=final_report),
    ])
    try:
        unverified_claims: List[str] = json.loads(check_response.content)
    except json.JSONDecodeError:
        unverified_claims = []

    citations = _build_citations(passages)

    trace_entry = (
        f"[P5] Report generated ({len(final_report)} chars). "
        f"Unverified claims: {len(unverified_claims)}. "
        f"Path: {'conflict' if conflict_report['has_conflicts'] else 'normal'}"
    )

    return {
        **state,
        "final_report": final_report,
        "citations": citations,
        "unverified_claims": unverified_claims,
        "reasoning_trace": [trace_entry],  # operator.add handles appending
    }


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    state = mock_state()
    result = synthesize_report(state)

    print("=== FINAL REPORT ===")
    print(result["final_report"])
    print("\n=== CITATIONS ===")
    for c in result["citations"]:
        print(f"  {c}")
    print(f"\n=== UNVERIFIED CLAIMS ({len(result['unverified_claims'])}) ===")
    for claim in result["unverified_claims"]:
        print(f"  - {claim}")
