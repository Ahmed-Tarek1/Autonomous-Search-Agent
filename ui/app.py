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

import streamlit as st
from pipeline import run_pipeline


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
        # Escape the claim for use in a regex, then allow flexible whitespace
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
        # Common Unicode punctuation -> ASCII equivalents
        "\u2019": "'", "\u2018": "'",   # curly single quotes
        "\u201c": '"', "\u201d": '"',   # curly double quotes
        "\u2013": "-", "\u2014": "--",  # en-dash, em-dash
        "\u2026": "...",                # ellipsis
        "\u00b7": "-",                  # middle dot
        "\u2022": "-",                  # bullet
        # Strip emojis and other non-Latin-1 symbols outright
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Drop anything still outside Latin-1 (e.g. emoji)
    text = text.encode("latin-1", errors="ignore").decode("latin-1")
    return text


def _mark_to_reportlab(text: str, unverified: list[str]) -> str:
    """
    Convert unverified-claim sentences into ReportLab <font backColor> spans
    so they appear highlighted yellow in the PDF (instead of HTML <mark> tags,
    which ReportLab ignores).
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
        "ReportTitle",
        parent=base["Title"],
        fontSize=18,
        spaceAfter=6,
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
        alignment=TA_LEFT,
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

    # Title — use plain ASCII so no emoji ■ box appears
    story.append(Paragraph("Research Report", title_style))
    story.append(Paragraph(f"<i>Q: {_safe_text(question)}</i>", caption_style))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cccccc"), spaceAfter=10))

    def md_to_paragraphs(text: str):
        """
        Convert markdown text to ReportLab Paragraph elements.
        Applies:
          - Heading detection (# / ## / ###)
          - Bullet conversion (-, *, +)
          - Bold / italic / code inline markup
          - Unverified-claim yellow highlighting via <font backColor>
          - Safe Latin-1 encoding to prevent ■ boxes
        """
        elems = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                elems.append(Spacer(1, 4))
                continue
            # headings
            h_match = re.match(r"^(#{1,3})\s+(.*)", line)
            if h_match:
                heading_text = _safe_text(h_match.group(2))
                elems.append(Paragraph(heading_text, h2_style))
                continue
            # bullets — use ASCII hyphen instead of Unicode bullet
            if line.startswith(("- ", "* ", "+ ")):
                line = "- " + line[2:]
            # Apply highlight BEFORE safe_text so we can match original wording,
            # then sanitise the result (backColor tag will survive safe_text
            # because it uses only ASCII characters itself).
            line = _mark_to_reportlab(line, unverified)
            line = _safe_text(line)
            # bold / italic / code (after safe_text so tags aren't mangled)
            line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
            line = re.sub(r"\*(.+?)\*", r"<i>\1</i>", line)
            line = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", line)
            elems.append(Paragraph(line, body_style))
        return elems

    story.extend(md_to_paragraphs(report_md))

    # Citations
    if citations:
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                 color=colors.HexColor("#cccccc"), spaceAfter=6))
        story.append(Paragraph("Sources", h2_style))
        for c in citations:
            story.append(Paragraph(f"- {_safe_text(c)}", cite_style))

    # Eval scores
    if scores:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Evaluation Scores", h2_style))
        labels = {
            "faithfulness": "Faithfulness",
            "answer_relevancy": "Answer Relevancy",
            "hallucination_rate": "Hallucination Rate",
            "latency_seconds": "Latency (s)",
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

# Quick-pick examples
st.caption("Try: " + " · ".join([
    "Does coffee improve cognitive performance?",
    "What causes the placebo effect?",
    "Is social media linked to teen depression?",
]))

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
# Persist pipeline results across reruns (e.g. when the download button is
# clicked Streamlit reruns the whole script — without session_state the
# results would disappear from the screen).
if "result" not in st.session_state:
    st.session_state.result = None
if "result_question" not in st.session_state:
    st.session_state.result_question = ""
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

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
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    progress_bar.progress(1.0, text="✅ Complete!")
    step_status.empty()

    # Build PDF once and cache it so the download button never triggers a
    # re-run that loses the displayed results.
    unverified = result.get("unverified_claims") or []
    pdf_bytes = build_pdf(
        question=question,
        report_md=result["final_report"],
        citations=result.get("citations", []),
        scores=result.get("eval_scores") or {},
        sub_questions=result.get("sub_questions", []),
        unverified=unverified,
    )

    # Persist everything in session state
    st.session_state.result = result
    st.session_state.result_question = question
    st.session_state.pdf_bytes = pdf_bytes

elif run_btn and not question.strip():
    st.warning("Please enter a research question.")

# ---------------------------------------------------------------------------
# Results layout — rendered from session_state so it survives the download
# button rerun as well as the initial pipeline run rerun.
# ---------------------------------------------------------------------------
if st.session_state.result is not None:
    result = st.session_state.result
    question_display = st.session_state.result_question
    pdf_bytes = st.session_state.pdf_bytes

    st.divider()
    conflict = result.get("conflict_report", {"has_conflicts": False, "pairs": []})

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

        # Unverified claims — inline highlight notice (no separate list)
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

        # Report — with unverified claims highlighted inline
        st.subheader("Research Report")
        report_html = highlight_unverified_claims(result["final_report"], unverified)
        # Render as HTML so <mark> tags work
        st.markdown(report_html, unsafe_allow_html=True)

        # Citations
        if result.get("citations"):
            st.subheader("Sources")
            for c in result["citations"]:
                st.markdown(f"- {c}")

        # ── PDF export ──────────────────────────────────────────────────────
        st.divider()
        # pdf_bytes was built once after pipeline ran and stored in session_state.
        # Clicking this button will trigger a Streamlit rerun, but since we
        # render from session_state the page content is unchanged.
        st.download_button(
            label="⬇️ Export as PDF",
            data=pdf_bytes,
            file_name="research_report.pdf",
            mime="application/pdf",
            use_container_width=False,
        )

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