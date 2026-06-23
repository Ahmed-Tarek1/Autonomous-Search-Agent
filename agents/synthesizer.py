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

# pyrefly: ignore [missing-import]
from state import ResearchState, mock_state
import json
from typing import List, Dict, Any
import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    groq_api_key=os.environ.get("GROQ_API_KEY", "mock-key"),
)

llm_json = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    groq_api_key=os.environ.get("GROQ_API_KEY", "mock-key"),
).bind(response_format={"type": "json_object"})

# ---------------------------------------------------------------------------
# Prompts Definition
# ---------------------------------------------------------------------------
SYNTHESIS_SYSTEM_BASE = """You are an expert research synthesizer. 
Your task is to write a comprehensive, professionally structured Markdown report that directly answers the user's question based ONLY on the provided factual passages.

Rules:
1. Every factual claim you make MUST be immediately followed by an inline citation referencing its source number, formatted exactly as [Source N] (e.g., [Source 1], [Source 2]).
2. NEVER synthesize or assume any facts that are not explicitly stated in the provided passages.
3. Structure your response using clear Markdown headers (##).
4. Do NOT include a final "Sources" or bibliography section inside the report body; the backend will append that automatically from the metadata.
5. CRITICAL: Do NOT write any generic introductory or concluding sentences unless they are strictly derived from and cited with a [Source N]. Every single sentence in the report body must have a citation.
"""

SYNTHESIS_SYSTEM_CONFLICT = (
    SYNTHESIS_SYSTEM_BASE
    + """
6. CRITICAL: A conflict has been detected in the source materials. You MUST include a dedicated section titled "## Conflicting Evidence" early in the report. In this section, explicitly analyze the detected contradictions, explaining which sources disagree and what points they dispute based on the provided conflict details.
"""
)

SELF_CHECK_SYSTEM = """You are a rigorous factual auditor. Your job is to read a generated research report and identify any factual claims that lack an explicit inline citation (e.g., claims missing a [Source N] tag).

You must return your response as a strict JSON object containing a single array of strings under the key "unverified_claims". Each string should be a specific claim from the text that is un-cited. If all claims are properly cited, return an empty array.

Example Output format:
{
  "unverified_claims": ["The study found a 20% increase in baseline metabolic rates.", "Long-term effects include potential vitamin deficiency."]
}
"""

# ---------------------------------------------------------------------------
# Core Logic Node
# ---------------------------------------------------------------------------


def synthesize_report(state: ResearchState) -> ResearchState:
    """
    Person 5 Node: Implements report synthesis using Llama-3.1 via Groq.
    - Builds a numbered passages block (truncates each to 500 chars).
    - Selects system prompt dynamically based on conflict_report.has_conflicts.
    - Runs a self-check LLM pass to fill unverified_claims.
    - Automatically appends bibliography and returns state updates for LangGraph.
    """

    passages = state["retrieved_passages"]
    question = state["question"]
    conflict_report = state["conflict_report"]
    has_conflicts = conflict_report["has_conflicts"]

    passages_block = ""
    for i, p in enumerate(passages):
        truncated_text = p["text"][:500]
        passages_block += f"[Source {i+1}]\nTitle: {p['title']}\nURL: {p['url']}\nContent: {truncated_text}\n\n"

    if has_conflicts:
        system_prompt = SYNTHESIS_SYSTEM_CONFLICT
        conflict_details = "\nDetected Contradictions:\n"
        for pair in conflict_report.get("pairs", []):
            conflict_details += f"- Conflict between '{pair['passage_a']['title']}' and '{pair['passage_b']['title']}': {pair['explanation']}\n"

        user_content = (
            f"Question: {question}\n\n{conflict_details}\n\nPassages:\n{passages_block}"
        )
    else:
        system_prompt = SYNTHESIS_SYSTEM_BASE
        user_content = f"Question: {question}\n\nPassages:\n{passages_block}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)
    final_report = response.content

    # --- Self-check pass for unverified_claims ---
    self_check_user = (
        f"Please audit the following report for un-cited claims:\n\n{final_report}"
    )
    check_messages = [
        SystemMessage(content=SELF_CHECK_SYSTEM),
        HumanMessage(content=self_check_user),
    ]

    check_response = llm_json.invoke(check_messages)
    try:
        unverified_claims_data = json.loads(check_response.content)
        unverified_claims = unverified_claims_data.get("unverified_claims", [])
    except (json.JSONDecodeError, KeyError):
        unverified_claims = []

    # Build citations list
    citations = [f"[{i+1}] {p['title']} — {p['url']}" for i, p in enumerate(passages)]

    final_report += "\n\n## Sources\n" + "\n".join(f"*{c}*" for c in citations)

    path_str = "conflict" if has_conflicts else "normal"
    trace_msg = f"[P5] Successfully synthesized report via {path_str} path using Groq (Llama-3.1). Found {len(unverified_claims)} unverified claims during self-audit."

    return {
        "final_report": final_report,
        "citations": citations,
        "unverified_claims": unverified_claims,
        "reasoning_trace": [trace_msg],
    }


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print("===============================================================")
    print("STARTING UNIT TESTS FOR PERSON 5 NODE (SYNTHESIZER)")
    print("===============================================================")

    if not os.environ.get("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY is not set in environment variables.")
        print("Please run: export GROQ_API_KEY='your_key' or put it in .env")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # TEST CASE 1: Normal Path (No Conflicts)
    # -----------------------------------------------------------------------
    print("\n▶️ [TEST 1/2] Running Normal Path (Happy Path)...")
    try:
        state_normal = mock_state()
        result_normal = synthesize_report(state_normal)

        print("\n✅ TEST 1 SUCCESS!")
        print("=== FINAL REPORT (NORMAL) ===")
        print(result_normal["final_report"])
        print("\n=== CITATIONS ===")
        for c in result_normal["citations"]:
            print(f"  {c}")
        print(
            f"\n=== UNVERIFIED CLAIMS ({len(result_normal['unverified_claims'])}) ==="
        )
        for claim in result_normal["unverified_claims"]:
            print(f"  - {claim}")
        print("\nTrace Appended:", result_normal["reasoning_trace"])

    except Exception as e:
        print(f"❌ TEST 1 FAILED: {e}")

    print("\n" + "=" * 60 + "\n")

    # -----------------------------------------------------------------------
    # TEST CASE 2: Conflict Path (Active Contradictions)
    # -----------------------------------------------------------------------
    print("▶️ [TEST 2/2] Running Conflict Path (Forcing Contradictions)...")
    try:
        state_conflict = mock_state()
        state_conflict["conflict_report"] = {
            "has_conflicts": True,
            "pairs": [
                {
                    "passage_a": state_conflict["retrieved_passages"][0],
                    "passage_b": state_conflict["retrieved_passages"][1],
                    "verdict": "contradict",
                    "confidence": 0.88,
                    "explanation": (
                        "Source 1 claims intermittent fasting consistently leads to a 3-8% weight loss, "
                        "whereas Source 2 strongly disputes this, claiming that metabolic benefits and weight loss "
                        "are merely due to overall continuous caloric restriction, showing no specific advantage for IF."
                    ),
                }
            ],
        }

        result_conflict = synthesize_report(state_conflict)

        print("\n✅ TEST 2 SUCCESS!")
        print("=== FINAL REPORT (WITH CONFLICTS) ===")
        print(result_conflict["final_report"])
        print("\n=== CITATIONS ===")
        for c in result_conflict["citations"]:
            print(f"  {c}")
        print(
            f"\n=== UNVERIFIED CLAIMS ({len(result_conflict['unverified_claims'])}) ==="
        )
        for claim in result_conflict["unverified_claims"]:
            print(f"  - {claim}")
        print("\nTrace Appended:", result_conflict["reasoning_trace"])

    except Exception as e:
        print(f"❌ TEST 2 FAILED: {e}")

    print("\n===============================================================")
    print("UNIT TESTING COMPLETED")
    print("===============================================================")
