"""
ui/app.py — Streamlit Demo UI
==============================
Owner: Person 5

Run:
    streamlit run ui/app.py

Shows:
  - Research question input
  - Live pipeline progress (real node events via SSE from FastAPI)
  - Final report with inline citations
  - Conflict warning if detected
  - Eval scores
  - Reasoning trace (expandable)
"""

import sys
import os
import io
import re
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
import markdown as md_lib

import json
import requests

import streamlit as st

# ---------------------------------------------------------------------------
# FastAPI backend URL — override with RESEARCH_API_URL env var if needed
# ---------------------------------------------------------------------------
API_BASE = os.getenv("RESEARCH_API_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def highlight_unverified_claims(report: str, unverified: list[str]) -> str:
    """
    Wrap each unverified claim sentence in a yellow highlight span.
    Matches are case-insensitive and tolerant of minor whitespace differences.
    """
    if not unverified:
        return report

    for claim in unverified:
        pattern = re.escape(claim.strip())
        pattern = re.sub(r"\\ ", r"\\s+", pattern)
        highlighted = (
            f'<mark style="background:#fff176;border-radius:3px;'
            f'padding:1px 3px;" title="Unverified claim">\\g<0></mark>'
        )
        report = re.sub(pattern, highlighted, report, flags=re.IGNORECASE)

    return report


def _safe_text(text: str) -> str:
    """
    Replace characters that ReportLab's built-in fonts (Helvetica/Times/Courier)
    cannot encode — anything outside Latin-1 — with safe ASCII equivalents.
    This prevents the ■ placeholder boxes in the output PDF.
    """
    replacements = {
        "\u2019": "'", "\u2018": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--",
        "\u2026": "...",
        "\u00b7": "-",
        "\u2022": "-",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = text.encode("latin-1", errors="ignore").decode("latin-1")
    return text


def _mark_to_reportlab(text: str, unverified: list[str]) -> str:
    """
    Convert unverified-claim sentences into ReportLab <font backColor> spans
    so they appear highlighted yellow in the PDF.
    """
    if not unverified:
        return text
    for claim in unverified:
        pattern = re.escape(claim.strip())
        pattern = re.sub(r"\\ ", r"\\s+", pattern)
        text = re.sub(
            pattern,
            r'<font backColor="#fff176">\g<0></font>',
            text,
            flags=re.IGNORECASE,
        )
    return text


def build_pdf(question: str, report_md: str, citations: list[str],
              scores: dict, sub_questions: list[str],
              unverified: list[str] | None = None) -> bytes:
    """Render the research report as a styled PDF and return raw bytes."""
    unverified = unverified or []
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    base = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle", parent=base["Title"],
        fontSize=18, spaceAfter=6,
        textColor=colors.HexColor("#1a237e"),
    )
    h2_style = ParagraphStyle(
        "H2", parent=base["Heading2"],
        fontSize=13, spaceBefore=14, spaceAfter=4,
        textColor=colors.HexColor("#283593"),
    )
    body_style = ParagraphStyle(
        "Body", parent=base["Normal"],
        fontSize=10, leading=15, spaceAfter=6,
    )
    caption_style = ParagraphStyle(
        "Caption", parent=base["Normal"],
        fontSize=9, textColor=colors.HexColor("#555555"),
        spaceAfter=10,
    )
    cite_style = ParagraphStyle(
        "Cite", parent=base["Normal"],
        fontSize=9, leading=13, leftIndent=10,
        textColor=colors.HexColor("#444444"),
    )
    score_style = ParagraphStyle(
        "Score", parent=base["Normal"],
        fontSize=9, leading=14,
    )

    story = []

    story.append(Paragraph("Research Report", title_style))
    story.append(Paragraph(f"<i>Q: {_safe_text(question)}</i>", caption_style))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cccccc"), spaceAfter=10))

    def md_to_paragraphs(text: str):
        elems = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                elems.append(Spacer(1, 4))
                continue
            h_match = re.match(r"^(#{1,3})\s+(.*)", line)
            if h_match:
                elems.append(Paragraph(_safe_text(h_match.group(2)), h2_style))
                continue
            if line.startswith(("- ", "* ", "+ ")):
                line = "- " + line[2:]
            line = _mark_to_reportlab(line, unverified)
            line = _safe_text(line)
            line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
            line = re.sub(r"\*(.+?)\*", r"<i>\1</i>", line)
            line = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", line)
            elems.append(Paragraph(line, body_style))
        return elems

    story.extend(md_to_paragraphs(report_md))

    # if citations:
    #     story.append(Spacer(1, 8))
    #     story.append(HRFlowable(width="100%", thickness=0.5,
    #                              color=colors.HexColor("#cccccc"), spaceAfter=6))
    #     story.append(Paragraph("Sources", h2_style))
    #     for c in citations:
    #         story.append(Paragraph(f"- {_safe_text(c)}", cite_style))

    if scores:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Evaluation Scores", h2_style))
        labels = {
            "faithfulness":      "Faithfulness",
            "answer_relevancy":  "Answer Relevancy",
            "hallucination_rate":"Hallucination Rate",
            "latency_seconds":   "Latency (s)",
        }
        for key, label in labels.items():
            val = scores.get(key)
            if val is not None:
                story.append(Paragraph(f"<b>{label}:</b> {val:.2f}", score_style))

    doc.build(story)
    return buf.getvalue()


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

st.caption("Try: " + " · ".join([
    "Does coffee improve cognitive performance?",
    "What causes the placebo effect?",
    "Is social media linked to teen depression?",
]))

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "result_question" not in st.session_state:
    st.session_state.result_question = ""
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

# ---------------------------------------------------------------------------
# Pipeline execution — streams from FastAPI /research/stream via SSE
# ---------------------------------------------------------------------------
if run_btn and question.strip():
    st.divider()

    progress_bar = st.progress(0, text="Connecting to research pipeline...")
    step_status  = st.empty()

    # Placeholders created inside the run_btn block only — if they were
    # created unconditionally they would blank themselves on every Streamlit
    # rerun (e.g. download button click), wiping the live sidebar content.
    # The static results block below re-renders everything from session_state.
    sub_q_placeholder = st.sidebar.empty()
    trace_placeholder = st.sidebar.empty()

    # Node names must match exactly what pipeline.py passes to graph.add_node()
    NODE_PROGRESS = {
        "decompose":          (0.15, "🔍 Decomposing query into sub-questions..."),
        "search":             (0.35, "🌐 Search agent running ReAct loop..."),
        "retrieve":           (0.55, "📚 Retrieving and ranking passages..."),
        "detect":             (0.70, "⚖️ Detecting conflicts between sources..."),
        "synthesize_normal":  (0.90, "✍️ Synthesizing report..."),
        "synthesize_warning": (0.90, "✍️ Synthesizing report (with conflict warnings)..."),
    }

    nodes_seen:         list[str] = []
    live_sub_questions: list[str] = []
    live_trace:         list[str] = []
    result = None

    try:
        with requests.post(
            f"{API_BASE}/research/stream",
            json={"question": question.strip(), "run_eval": True},
            stream=True,
            timeout=300,
        ) as resp:
            if resp.status_code != 200:
                st.error(f"Backend error {resp.status_code}: {resp.text}")
                st.stop()

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue

                try:
                    event = json.loads(line[len("data:"):].strip())
                except json.JSONDecodeError as e:
                    print(f"[DEBUG] Bad SSE line: {repr(line)} | error: {e}")
                    continue

                etype = event.get("type")
                data  = event.get("data", {})

                if etype == "start":
                    step_status.info("Pipeline started — waiting for first node...")

                elif etype == "node_done":
                    node = data.get("node", "")
                    nodes_seen.append(node)

                    if node in NODE_PROGRESS:
                        frac, label = NODE_PROGRESS[node]
                    else:
                        frac  = min(0.88, 0.1 + 0.8 * len(nodes_seen) / max(len(NODE_PROGRESS), 1))
                        label = f"⚙️ Running node: {node}..."

                    progress_bar.progress(frac, text=label)
                    step_status.info(label)

                    if data.get("sub_questions"):
                        live_sub_questions = data["sub_questions"]
                    if data.get("trace"):
                        live_trace.extend(data["trace"])

                    if live_sub_questions:
                        with sub_q_placeholder.container():
                            st.subheader("Sub-questions")
                            for i, q in enumerate(live_sub_questions, 1):
                                st.markdown(f"**{i}.** {q}")

                    if live_trace:
                        with trace_placeholder.container():
                            with st.expander("Agent reasoning trace (live)", expanded=True):
                                for entry in live_trace[-10:]:
                                    st.text(entry)

                    if data.get("conflict_detected"):
                        step_status.warning("⚠️ Conflicting evidence detected — continuing...")

                elif etype == "complete":
                    # eval runs on the backend after astream finishes;
                    # scores arrive here in the complete event payload.
                    progress_bar.progress(0.95, text="📊 Running evaluation...")
                    step_status.info("📊 Running evaluation...")

                    result = {
                        "question":          question.strip(),
                        "sub_questions":     live_sub_questions,
                        "final_report":      data.get("final_report", ""),
                        "citations":         data.get("citations", []),
                        "unverified_claims": data.get("unverified_claims", []),
                        "conflict_report": {
                            "has_conflicts": data.get("conflict_detected", False),
                            "pairs":         [],
                        },
                        # eval_scores is populated by evaluate_state() in main.py
                        # after the astream loop; it arrives here in the complete event.
                        "eval_scores":       data.get("eval_scores"),
                        "reasoning_trace":   data.get("reasoning_trace", live_trace),
                    }
                    progress_bar.progress(1.0, text="✅ Complete!")
                    step_status.empty()

                elif etype == "error":
                    st.error(f"Pipeline error: {data.get('message', 'unknown error')}")
                    st.stop()

    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to the FastAPI backend at **{API_BASE}**. "
            "Make sure it is running with: `uvicorn main:app --reload`"
        )
        st.stop()
    except requests.exceptions.Timeout:
        st.error("Request timed out after 5 minutes. The pipeline may still be running.")
        st.stop()

    if result is None:
        st.error("Stream ended without a 'complete' event. Check the backend logs.")
        st.stop()

    unverified = result.get("unverified_claims") or []
    pdf_bytes = build_pdf(
        question=question,
        report_md=result["final_report"],
        citations=result.get("citations", []),
        scores=result.get("eval_scores") or {},
        sub_questions=result.get("sub_questions", []),
        unverified=unverified,
    )

    st.session_state.result          = result
    st.session_state.result_question = question
    st.session_state.pdf_bytes       = pdf_bytes

    # Do NOT clear the live placeholders here — clearing then re-rendering
    # in the same Streamlit pass causes a blank flash.  The static results
    # block below renders the same content from session_state on every rerun,
    # so the sidebar stays populated after the download button is clicked.

elif run_btn and not question.strip():
    st.warning("Please enter a research question.")

# ---------------------------------------------------------------------------
# Results layout — rendered from session_state so it survives reruns
# ---------------------------------------------------------------------------
if st.session_state.result is not None:
    result    = st.session_state.result
    pdf_bytes = st.session_state.pdf_bytes

    st.divider()
    conflict = result.get("conflict_report", {"has_conflicts": False, "pairs": []})

    col_report, col_sidebar = st.columns([3, 1])

    with col_report:
        if conflict["has_conflicts"]:
            st.markdown(
                f"""<div class="conflict-box">
                ⚠️ <strong>Conflicting evidence detected</strong> — 
                {len(conflict['pairs'])} contradiction(s) found across sources. 
                See the "Conflicting Evidence" section in the report below.
                </div>""",
                unsafe_allow_html=True,
            )

        unverified = result.get("unverified_claims") or []
        if unverified:
            st.markdown(
                f"""<div class="unverified-box">
                🚨 <strong>{len(unverified)} unverified claim(s)</strong> detected by 
                self-check — highlighted <mark style="background:#fff176;
                border-radius:3px;padding:1px 4px;">in yellow</mark> in the report below.
                </div>""",
                unsafe_allow_html=True,
            )

        st.subheader("Research Report")
        report_html = highlight_unverified_claims(result["final_report"], unverified)
        st.markdown(report_html, unsafe_allow_html=True)

        if result.get("citations"):
            st.subheader("Sources")
            for c in result["citations"]:
                st.markdown(f"- {c}")

        st.divider()
        st.download_button(
            label="⬇️ Export as PDF",
            data=pdf_bytes,
            file_name="research_report.pdf",
            mime="application/pdf",
            use_container_width=False,
        )

    with col_sidebar:
        st.subheader("Sub-questions")
        for i, q in enumerate(result.get("sub_questions", []), 1):
            st.markdown(f"**{i}.** {q}")

        st.divider()

        scores = result.get("eval_scores") or {}
        st.subheader("Eval scores")

        if not scores:
            st.caption("Scores unavailable — eval may have been skipped or failed.")
        else:
            cols = st.columns(2)
            metrics = [
                ("Faithfulness",  scores.get("faithfulness")),
                ("Relevancy",     scores.get("answer_relevancy")),
                ("Hallucination", scores.get("hallucination_rate")),
                ("Latency (s)",   scores.get("latency_seconds")),
            ]
            for i, (label, val) in enumerate(metrics):
                with cols[i % 2]:
                    st.metric(label, f"{val:.2f}" if val is not None else "—")

        st.divider()

        with st.expander("Agent reasoning trace", expanded=False):
            for entry in result.get("reasoning_trace", []):
                st.text(entry)