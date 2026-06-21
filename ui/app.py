"""
ui/app.py — Streamlit Demo UI
==============================
Owner: Person 5

Run:
    streamlit run ui/app.py

Shows:
  - Research question input
  - Live pipeline progress
  - Final report with inline citations
  - Conflict warning if detected
  - Eval scores
  - Reasoning trace (expandable)
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from pipeline import run_pipeline

st.set_page_config(
    page_title="Autonomous Research Assistant",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.conflict-box {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 12px 0;
}
.unverified-box {
    background: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 12px 0;
}
.score-card {
    background: #e8f5e9;
    border-radius: 6px;
    padding: 10px 14px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🔬 Autonomous Research Assistant")
st.caption(
    "Multi-agent pipeline · LangGraph · Qdrant RAG · Conflict detection · RAGAS eval"
)
st.divider()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
col_input, col_btn = st.columns([5, 1])
with col_input:
    question = st.text_input(
        "Research question",
        placeholder="e.g. What are the health effects of intermittent fasting?",
        label_visibility="collapsed",
    )
with col_btn:
    run_btn = st.button("Research →", type="primary", use_container_width=True)

# Quick-pick examples
st.caption("Try: " + " · ".join([
    "Does coffee improve cognitive performance?",
    "What causes the placebo effect?",
    "Is social media linked to teen depression?",
]))

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
if run_btn and question.strip():
    st.divider()

    # Progress display
    progress_bar = st.progress(0, text="Starting pipeline...")
    step_status = st.empty()
    steps = [
        (0.15, "🔍 Decomposing query into sub-questions..."),
        (0.35, "🌐 Search agent running ReAct loop..."),
        (0.55, "📚 Retrieving and ranking passages..."),
        (0.70, "⚖️ Detecting conflicts between sources..."),
        (0.85, "✍️ Synthesizing report..."),
        (1.00, "📊 Running evaluation..."),
    ]

    # Show progress while pipeline runs in background
    # (In production, wire to the SSE stream endpoint instead)
    for progress, label in steps[:-2]:
        progress_bar.progress(progress, text=label)
        step_status.info(label)
        time.sleep(0.3)

    with st.spinner("Running full pipeline..."):
        start = time.time()
        try:
            result = run_pipeline(question.strip(), run_eval=True)
            latency = time.time() - start
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    progress_bar.progress(1.0, text="✅ Complete!")
    step_status.empty()

    conflict = result.get("conflict_report", {"has_conflicts": False, "pairs": []})

    # ---------------------------------------------------------------------------
    # Results layout
    # ---------------------------------------------------------------------------
    col_report, col_sidebar = st.columns([3, 1])

    with col_report:
        # Conflict warning
        if conflict["has_conflicts"]:
            st.markdown(
                f"""<div class="conflict-box">
                ⚠️ <strong>Conflicting evidence detected</strong> — 
                {len(conflict['pairs'])} contradiction(s) found across sources. 
                See the "Conflicting Evidence" section in the report below.
                </div>""",
                unsafe_allow_html=True,
            )

        # Unverified claims warning
        if result.get("unverified_claims"):
            claims = result["unverified_claims"]
            st.markdown(
                f"""<div class="unverified-box">
                🚨 <strong>{len(claims)} unverified claim(s)</strong> found by self-check 
                (claims without a cited source):
                <ul>{"".join(f"<li>{c}</li>" for c in claims)}</ul>
                </div>""",
                unsafe_allow_html=True,
            )

        # Report
        st.subheader("Research Report")
        st.markdown(result["final_report"])

        # Citations
        if result.get("citations"):
            st.subheader("Sources")
            for c in result["citations"]:
                st.markdown(f"- {c}")

    with col_sidebar:
        # Sub-questions
        st.subheader("Sub-questions")
        for i, q in enumerate(result.get("sub_questions", []), 1):
            st.markdown(f"**{i}.** {q}")

        st.divider()

        # Eval scores
        scores = result.get("eval_scores") or {}
        st.subheader("Eval scores")

        cols = st.columns(2)
        metrics = [
            ("Faithfulness", scores.get("faithfulness")),
            ("Relevancy", scores.get("answer_relevancy")),
            ("Hallucination", scores.get("hallucination_rate")),
            ("Latency (s)", scores.get("latency_seconds")),
        ]
        for i, (label, val) in enumerate(metrics):
            with cols[i % 2]:
                if val is not None:
                    st.metric(label, f"{val:.2f}")
                else:
                    st.metric(label, "—")

        st.divider()

        # Reasoning trace
        with st.expander("Agent reasoning trace", expanded=False):
            for entry in result.get("reasoning_trace", []):
                st.text(entry)

elif run_btn and not question.strip():
    st.warning("Please enter a research question.")
