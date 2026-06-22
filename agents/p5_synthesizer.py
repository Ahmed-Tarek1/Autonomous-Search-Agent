"""
agents/p5_synthesizer.py — Synthesizer + Report Generator
==========================================================
Owner: Person 5
Reads:  state["question"], state["retrieved_passages"], state["conflict_report"]
Writes: state["final_report"], state["citations"], state["unverified_claims"]

Contract:
  - final_report: Markdown string with inline [Source N] citations and ## headers
  - citations: List[str] formatted as "[N] Title — url"
  - unverified_claims: List[str] — claims without a [Source N] tag (from self-check)
  - If conflict_report.has_conflicts is True, report MUST include ## Conflicting Evidence section
  - reasoning_trace: plain List[str] with exactly 1 new entry ("[P5] ...")
    LangGraph appends via operator.add.

Run in isolation:
    python agents/p5_synthesizer.py
"""

from state import ResearchState, mock_state


def synthesize_report(state: ResearchState) -> ResearchState:
    """
    TODO (Person 5): Implement report synthesis using Claude.
    - Build a numbered passages block (truncate each to 500 chars)
    - Choose system prompt: SYNTHESIS_SYSTEM_BASE or SYNTHESIS_SYSTEM_CONFLICT
      based on conflict_report.has_conflicts
    - Call LLM to generate final_report (Markdown with inline [Source N] citations)
    - Run self-check LLM pass: find claims without a citation → unverified_claims
    - Build citations list from retrieved_passages
    - Return exactly one reasoning_trace entry
    """
    # Mock output so pipeline runs end-to-end from Day 1
    passages = state["retrieved_passages"]
    question = state["question"]
    has_conflicts = state["conflict_report"]["has_conflicts"]

    mock_report = (
        f"## Research Summary\n\n"
        f"This is a stub report for: *{question}*\n\n"
        f"{'## Conflicting Evidence\\n\\nSTUB — conflict path active.\\n\\n' if has_conflicts else ''}"
        f"[Source 1] Finding A. [Source 2] Finding B.\n\n"
        f"## Sources\n"
        + "\n".join(f"[{i+1}] {p['title']} — {p['url']}" for i, p in enumerate(passages))
    )

    citations = [
        f"[{i+1}] {p['title']} — {p['url']}"
        for i, p in enumerate(passages)
    ]

    return {
        **state,
        "final_report": mock_report,
        "citations": citations,
        "unverified_claims": [],
        "reasoning_trace": [
            f"[P5] STUB — generated mock report ({len(mock_report)} chars), "
            f"path={'conflict' if has_conflicts else 'normal'}"
        ],
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
    print("Trace:", result["reasoning_trace"])