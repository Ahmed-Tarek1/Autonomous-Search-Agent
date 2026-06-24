"""
agents/retriever.py — Semantic Retriever
============================================
Owner: Person 3
Reads:  state["search_results"], state["question"], state["sub_questions"]
Writes: state["retrieved_passages"], state["reasoning_trace"]

Contract:
  - retrieved_passages: List[Passage], max 5 items, max 2 per domain
  - Each Passage: {text, url, title, score (float), source (domain)}
  - reasoning_trace: plain List[str] with exactly 1 new entry ("[P3] ...")
    LangGraph appends via operator.add.

Run in isolation:
    python agents/retriever.py
"""

from state import ResearchState, Passage, mock_state


def retrieve_passages(state: ResearchState) -> ResearchState:
    """
    TODO (Person 3): Implement semantic retrieval with Qdrant + cross-encoder.
    - Chunk each search result snippet (300 chars, 50 overlap)
    - Embed with SentenceTransformer("all-MiniLM-L6-v2") and upsert to Qdrant
    - Query Qdrant with question + each sub-question (top-20 candidates)
    - Re-rank with CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    - Apply diversity filter: max 2 passages per domain, keep top 5
    - Delete Qdrant collection after retrieval (clean-up for next run)
    - Return exactly one reasoning_trace entry
    """
    # Mock output so pipeline runs end-to-end from Day 1
    passages: list[Passage] = [
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
    ]
    return {
        **state,
        "retrieved_passages": passages,
        "reasoning_trace": [
            f"[P3] STUB — returning {len(passages)} mock passages"
        ],
    }


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    state = mock_state()
    result = retrieve_passages(state)
    print(f"\nRetrieved {len(result['retrieved_passages'])} passages:")
    for p in result["retrieved_passages"]:
        print(f"  [{p['score']:.3f}] [{p['source']}] {p['title'][:60]}")
    print("Trace:", result["reasoning_trace"])