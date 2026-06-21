"""
agents/p3_retriever.py — Semantic Retriever
============================================
Owner: Person 3
Reads:  state["search_results"], state["question"], state["sub_questions"]
Writes: state["retrieved_passages"], state["reasoning_trace"]

Fully deterministic — no LLM calls.

Fix v2: Qdrant client wrapped in try/except to prevent import-time crash.
        Passage compression replaced with simple truncation (saves 5 LLM calls).

Run in isolation:
    python agents/p3_retriever.py
"""

import os
import uuid
from typing import List

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from state import ResearchState, Passage, SearchResult, mock_state

# ---------------------------------------------------------------------------
# Models — loaded once at import time
# ---------------------------------------------------------------------------
print("[P3] Loading embedding model...")
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

print("[P3] Loading cross-encoder...")
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

COLLECTION_NAME = "research_passages"
TOP_K_RETRIEVE = 20
TOP_K_FINAL = 5
MAX_PER_DOMAIN = 2
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# ---------------------------------------------------------------------------
# Qdrant client — lazy with safe fallback
# Fix v2: wraps construction in try/except so import never raises.
# ---------------------------------------------------------------------------
def _build_qdrant() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")
    if url:
        try:
            return QdrantClient(url=url, api_key=key)
        except Exception as e:
            print(f"[P3] Qdrant cloud connection failed ({e}) — using in-memory")
    print("[P3] No QDRANT_URL set — using in-memory Qdrant")
    return QdrantClient(":memory:")

qdrant = _build_qdrant()


def _chunk_text(text: str) -> List[str]:
    """Split text into overlapping character chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _ensure_collection(dim: int = 384):
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def _ingest(search_results: List[SearchResult]) -> int:
    """Chunk, embed, and upsert all search results into Qdrant."""
    _ensure_collection()
    points = []

    for result in search_results:
        text = result["snippet"].strip()
        if not text:
            continue
        # Fix v2: simple truncation instead of LLM compression — saves API calls
        text = text[:1500]
        chunks = _chunk_text(text)
        embeddings = EMBEDDER.encode(chunks, show_progress_bar=False)

        for chunk, vec in zip(chunks, embeddings):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist(),
                payload={
                    "text": chunk,
                    "url": result["url"],
                    "title": result["title"],
                    "source": result["source"],
                },
            ))

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def retrieve_passages(state: ResearchState) -> ResearchState:
    """
    LangGraph node — P3.
    Embeds search results → Qdrant → cross-encoder re-rank → diversity filter.
    """
    n_chunks = _ingest(state["search_results"])
    print(f"  [P3] Ingested {n_chunks} chunks into Qdrant")

    queries = [state["question"]] + state.get("sub_questions", [])
    candidate_map = {}

    for query in queries:
        q_vec = EMBEDDER.encode(query, show_progress_bar=False).tolist()
        hits = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=q_vec,
            limit=TOP_K_RETRIEVE,
        )
        for hit in hits:
            pid = str(hit.id)
            if pid not in candidate_map or hit.score > candidate_map[pid][1]:
                candidate_map[pid] = (hit.payload, hit.score)

    candidates = list(candidate_map.values())
    print(f"  [P3] {len(candidates)} candidates before re-ranking")

    if candidates:
        pairs = [(state["question"], c[0]["text"]) for c in candidates]
        rerank_scores = RERANKER.predict(pairs)
        ranked = sorted(
            zip(candidates, rerank_scores),
            key=lambda x: -x[1],
        )
    else:
        ranked = []

    domain_count: dict = {}
    passages: List[Passage] = []

    for (payload, _), rerank_score in ranked:
        domain = payload["source"]
        if domain_count.get(domain, 0) >= MAX_PER_DOMAIN:
            continue
        passages.append(Passage(
            text=payload["text"],
            url=payload["url"],
            title=payload["title"],
            score=float(rerank_score),
            source=domain,
        ))
        domain_count[domain] = domain_count.get(domain, 0) + 1
        if len(passages) == TOP_K_FINAL:
            break

    domains_kept = list({p["source"] for p in passages})
    trace_entry = (
        f"[P3] Retrieved {len(passages)} passages from {len(domains_kept)} domains: "
        + str(domains_kept)
    )

    # Clean up collection for next run
    try:
        qdrant.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    return {
        **state,
        "retrieved_passages": passages,
        "reasoning_trace": [trace_entry],  # LangGraph appends via operator.add
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
