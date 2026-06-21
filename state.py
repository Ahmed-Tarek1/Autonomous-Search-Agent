"""
state.py — Shared ResearchState
================================
This is the contract all 6 modules code against.
NEVER change a field name or type after Day 1 without syncing the whole team.

Fix v2: reasoning_trace uses Annotated[List, operator.add] so LangGraph
        appends entries across nodes instead of last-write-wins overwriting.
"""

import operator
from typing import Annotated, TypedDict, List, Optional

from dotenv import load_dotenv
load_dotenv()  # must run before any LangChain import


class SearchResult(TypedDict):
    url: str
    title: str
    snippet: str
    source: str  # domain, e.g. "pubmed.ncbi.nlm.nih.gov"


class Passage(TypedDict):
    text: str
    url: str
    title: str
    score: float  # cosine similarity from Qdrant
    source: str   # domain name


class ConflictPair(TypedDict):
    passage_a: Passage
    passage_b: Passage
    verdict: str        # "agree" | "contradict" | "unrelated"
    confidence: float   # 0.0 – 1.0
    explanation: str


class ConflictReport(TypedDict):
    has_conflicts: bool
    pairs: List[ConflictPair]


class ResearchState(TypedDict):
    # Set by user — never overwritten
    question: str

    # Written by P1 — Query Decomposer
    sub_questions: List[str]

    # Annotated with operator.add so LangGraph appends rather than overwrites.
    # Every module does: return {**state, "reasoning_trace": ["[PX] ..."]}
    # LangGraph merges by concatenating, not replacing.
    reasoning_trace: Annotated[List[str], operator.add]

    # Written by P2 — Search Agent
    search_results: List[SearchResult]

    # Written by P3 — Retriever
    retrieved_passages: List[Passage]

    # Written by P4 — Conflict Detector
    conflict_report: ConflictReport

    # Written by P5 — Synthesizer
    final_report: str            # Markdown with inline [Source N] citations
    citations: List[str]         # ["[1] Title — url", ...]
    unverified_claims: List[str] # self-check: claims without a source

    # Written by P6 — Evaluator (outside graph)
    eval_scores: Optional[dict]


# ---------------------------------------------------------------------------
# Mock state helpers — every module uses these to test in isolation from Day 1
# ---------------------------------------------------------------------------

def mock_state() -> ResearchState:
    """A fully populated mock state for unit testing any module."""
    return ResearchState(
        question="What are the health effects of intermittent fasting?",
        sub_questions=[
            "What does research say about intermittent fasting and weight loss?",
            "How does intermittent fasting affect blood glucose and insulin sensitivity?",
            "What are the cognitive effects of intermittent fasting?",
            "Are there risks or side effects of long-term intermittent fasting?",
        ],
        reasoning_trace=["[P1] Decomposed into 4 sub-questions."],
        search_results=[
            SearchResult(
                url="https://pubmed.ncbi.nlm.nih.gov/example1",
                title="Intermittent fasting and weight loss: a systematic review",
                snippet="Studies show intermittent fasting leads to 3-8% weight loss over 3-24 weeks.",
                source="pubmed.ncbi.nlm.nih.gov",
            ),
            SearchResult(
                url="https://www.nejm.org/example2",
                title="Metabolic Effects of Intermittent Fasting",
                snippet="IF improves insulin sensitivity and reduces fasting glucose in overweight adults.",
                source="nejm.org",
            ),
        ],
        retrieved_passages=[
            Passage(
                text="Studies show intermittent fasting leads to 3-8% weight loss over 3-24 weeks.",
                url="https://pubmed.ncbi.nlm.nih.gov/example1",
                title="Intermittent fasting and weight loss: a systematic review",
                score=0.91,
                source="pubmed.ncbi.nlm.nih.gov",
            ),
            Passage(
                text="IF improves insulin sensitivity and reduces fasting glucose in overweight adults.",
                url="https://www.nejm.org/example2",
                title="Metabolic Effects of Intermittent Fasting",
                score=0.87,
                source="nejm.org",
            ),
        ],
        conflict_report=ConflictReport(has_conflicts=False, pairs=[]),
        final_report="",
        citations=[],
        unverified_claims=[],
        eval_scores=None,
    )
