"""
agents/retriever.py — Semantic Retriever
Reads:  state["search_results"], state["question"], state["sub_questions"]
Writes: state["retrieved_passages"], state["reasoning_trace"]

Contract:
  - retrieved_passages: list[Passage], max 5 items
  - Each Passage: {text, url, title, score (float), source (domain)}
  - reasoning_trace: plain list[str] with exactly 1 new entry ("[P3] ...")
    LangGraph appends via operator.add.

Run in isolation:
    python agents/retriever.py
"""

import os
import re
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer, CrossEncoder

from state import ResearchState, Passage, SearchResult, mock_state
import yaml
from dotenv import load_dotenv

def load_variables():
    file = open("./configs.yaml")
    configs = yaml.safe_load(file)
    file.close()
    load_dotenv()
    
    return configs

configs = load_variables()

# ---------------------------------------------------------------------------
# Tuning knobs — all configurable from .env, no code change needed
# ---------------------------------------------------------------------------
_CHUNK_SIZE    = configs.get("CHUNK_SIZE", 512)      # max chars per chunk
_CHUNK_OVERLAP = configs.get("CHUNK_OVERLAP", 64)    # overlap between adjacent chunks
_TOP_K         = configs.get("TOP_K", 20)            # candidates retrieved per query per retriever
_CE_TOP_N      = configs.get("CE_TOP_N", 40)         # max candidates passed to cross-encoder
_FINAL_TOP_N   = configs.get("TOP_N", 5)             # final passages returned
_MMR_LAMBDA    = configs.get("MMR_LAMBDA", 0.7)      # 1.0 = pure relevance, 0.0 = pure diversity
_RRF_K         = configs.get("RRF_K", 60)                                           # standard RRF constant (not tuned)





# BGE models are purpose-built for retrieval — consistently outperform MiniLM on MTEB benchmarks.
# BGE requires a query-side prefix for asymmetric retrieval; passages are encoded as-is.
_EMBED_MODEL    = configs["EMBEDDING_MODEL"]
_RERANK_MODEL   = configs["RERANKING_MODEL"]
_QUERY_PREFIX   = configs["QUERY_PREFIX"]
_COLLECTION     = configs["COLLECTION_NAME"]

# ---------------------------------------------------------------------------
# Splitter — tries paragraph → sentence → space → char in order.
# Never cuts a sentence mid-way unless it alone exceeds _CHUNK_SIZE.
# ---------------------------------------------------------------------------
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE,
    chunk_overlap=_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
    length_function=len,
)

# ---------------------------------------------------------------------------
# Module-level singletons — loaded once at import, reused on every call.
# These three names are monkeypatched by the test suite to avoid real I/O.
# ---------------------------------------------------------------------------
EMBEDDER = SentenceTransformer(_EMBED_MODEL)   # bi-encoder: text → dense vector
RERANKER = CrossEncoder(_RERANK_MODEL)         # cross-encoder: scores (query, passage) pairs
qdrant   = QdrantClient(":memory:")            # vector DB; swap URL+key for cloud


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer for BM25 — punctuation stripped."""
    return re.findall(r'\w+', text.lower())


def _mmr(
    ranked_ids: list[int],
    ce_scores: list[float],
    vectors: np.ndarray,
    n: int,
    lam: float,
) -> list[int]:
    """
    Maximal Marginal Relevance selection.

    Picks n passages that balance relevance (cross-encoder score) and diversity
    (dissimilarity to already-selected passages).
    lam=1.0 → pure relevance ranking; lam=0.0 → pure diversity.
    """
    if not ranked_ids:
        return []

    # Normalize cross-encoder scores to [0, 1] so they're on the same scale
    # as cosine similarity when computing the MMR trade-off
    scores = np.array(ce_scores, dtype=float)
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores = (scores - s_min) / (s_max - s_min)

    selected: list[int] = []
    remaining = list(zip(ranked_ids, scores.tolist()))

    while len(selected) < n and remaining:
        if not selected:
            # First pick: simply take the most relevant passage
            best_idx = int(np.argmax([s for _, s in remaining]))
        else:
            sel_vecs = vectors[selected]          # (num_selected, dim)
            best_score, best_idx = -np.inf, 0
            for j, (rid, rel) in enumerate(remaining):
                v = vectors[rid]
                # Cosine similarity between candidate and every selected passage
                sims = (sel_vecs @ v) / (
                    np.linalg.norm(sel_vecs, axis=1) * np.linalg.norm(v) + 1e-9
                )
                # MMR: maximise relevance, penalise similarity to what's already picked
                mmr_score = lam * rel - (1 - lam) * float(sims.max())
                if mmr_score > best_score:
                    best_score, best_idx = mmr_score, j

        selected.append(remaining.pop(best_idx)[0])

    return selected


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def retrieve_passages(state: ResearchState) -> ResearchState:
    search_results: list[SearchResult] = state["search_results"]
    question: str = state["question"]
    sub_questions: list[str] = state.get("sub_questions", [])

    if not search_results:
        return {
            "retrieved_passages": [],
            "reasoning_trace": ["[P3] No search results to retrieve from."],
        }

    # ------------------------------------------------------------------
    # Step 1 — Chunk
    # RecursiveCharacterTextSplitter splits on natural boundaries first
    # (paragraphs → sentences → spaces) so each chunk holds one coherent idea.
    # records carries (text, url, title, source) so metadata travels with chunks.
    # ------------------------------------------------------------------
    records: list[tuple] = []
    for sr in search_results:
        for chunk in _splitter.split_text(sr["snippet"]):
            records.append((chunk, sr["url"], sr["title"], sr["source"]))

    if not records:
        return {
            "retrieved_passages": [],
            "reasoning_trace": ["[P3] Chunking produced no records."],
        }

    # ------------------------------------------------------------------
    # Steps 2 & 3 — Embed passages + build BM25 index (parallel)
    # Both only depend on records; neither depends on the other.
    # BGE releases the GIL during tensor computation, so BM25 (pure Python)
    # runs concurrently on the CPU while the model handles inference.
    # ------------------------------------------------------------------
    texts = [r[0] for r in records]
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_vecs = pool.submit(EMBEDDER.encode, texts, show_progress_bar=False)
        future_bm25 = pool.submit(BM25Okapi, [_tokenize(t) for t in texts])
        vectors = future_vecs.result()
        bm25    = future_bm25.result()

    # ------------------------------------------------------------------
    # Step 4 — Store vectors in Qdrant
    # Metadata (url, title, source) stored per-point so no separate mapping needed.
    # ------------------------------------------------------------------
    existing = {c.name for c in qdrant.get_collections().collections}
    if _COLLECTION in existing:
        qdrant.delete_collection(_COLLECTION)

    qdrant.create_collection(
        collection_name=_COLLECTION,
        vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
    )
    qdrant.upsert(
        collection_name=_COLLECTION,
        points=[
            PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload={
                    "text":   records[i][0],
                    "url":    records[i][1],
                    "title":  records[i][2],
                    "source": records[i][3],
                },
            )
            for i in range(len(records))
        ],
    )

    try:
        # ------------------------------------------------------------------
        # Step 5 — Multi-query BM25 + dense retrieval → RRF fusion
        #
        # For each query (main question + sub-questions):
        #   - BM25 gives a keyword-match ranking
        #   - Qdrant gives a semantic similarity ranking
        # RRF merges both ranked lists into a single score using position:
        #   score += 1 / (60 + rank)   for every list the candidate appears in
        # This avoids the problem of BM25 and cosine scores being on different scales.
        # ------------------------------------------------------------------
        queries = [question] + list(sub_questions)
        # BGE query prefix enables asymmetric retrieval (query vs passage optimisation)
        query_vecs = EMBEDDER.encode(
            [_QUERY_PREFIX + q for q in queries],
            show_progress_bar=False,
        )

        rrf_scores: dict[int, float] = {}

        for query, qvec in zip(queries, query_vecs):
            # BM25 ranking for this query
            bm25_ranked = sorted(
                range(len(records)),
                key=lambda j: bm25.get_scores(_tokenize(query))[j],
                reverse=True,
            )[:_TOP_K]
            for rank, record_id in enumerate(bm25_ranked):
                rrf_scores[record_id] = rrf_scores.get(record_id, 0.0) + 1.0 / (_RRF_K + rank)

            # Dense ranking for this query
            for rank, hit in enumerate(qdrant.query_points(
                collection_name=_COLLECTION,
                query=qvec.tolist(),
                limit=_TOP_K,
            ).points):
                # Integer IDs index directly into records; resolve string IDs (mocks) via payload
                if isinstance(hit.id, int) and hit.id < len(records):
                    record_id = hit.id
                else:
                    try:
                        record_id = next(
                            i for i, r in enumerate(records) if r[0] == hit.payload["text"]
                        )
                    except StopIteration:
                        continue
                rrf_scores[record_id] = rrf_scores.get(record_id, 0.0) + 1.0 / (_RRF_K + rank)

        # Candidates sorted by fused RRF score descending
        rrf_ranked = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        # ------------------------------------------------------------------
        # Step 6 — Cross-encoder re-rank
        # Takes (question, chunk) as a pair — full attention between both texts
        # gives precise relevance scoring that the bi-encoder approximates.
        # Limited to _CE_TOP_N candidates (never the full corpus).
        # ------------------------------------------------------------------
        ce_candidates = rrf_ranked[:_CE_TOP_N]
        ce_scores = (
            RERANKER.predict([(question, records[i][0]) for i in ce_candidates])
            if ce_candidates else []
        )
        ce_ranked       = sorted(zip(ce_candidates, ce_scores), key=lambda x: x[1], reverse=True)
        ce_ids          = [idx for idx, _ in ce_ranked]
        ce_scores_list  = [s for _, s in ce_ranked]
        ce_score_map    = dict(zip(ce_ids, ce_scores_list))

        # ------------------------------------------------------------------
        # Step 7 — MMR diversity selection
        # Picks passages that are relevant to the query but dissimilar from
        # each other — content diversity, not just source-domain diversity.
        # ------------------------------------------------------------------
        selected_ids = _mmr(ce_ids, ce_scores_list, vectors, _FINAL_TOP_N, _MMR_LAMBDA)

        # ------------------------------------------------------------------
        # Step 8 — Build Passage objects
        # Score field carries the cross-encoder score (most meaningful relevance signal).
        # ------------------------------------------------------------------
        passages: list[Passage] = [
            Passage(
                text=records[rid][0],
                url=records[rid][1],
                title=records[rid][2],
                score=float(ce_score_map.get(rid, 0.0)),
                source=records[rid][3],
            )
            for rid in selected_ids
        ]

        return {
            "retrieved_passages": passages,
            "reasoning_trace": [
                f"[P3] Retrieved {len(passages)} passages from {len(search_results)} results "
                f"(chunked into {len(records)}, BM25+dense RRF fusion, "
                f"{len(ce_candidates)} cross-encoder re-ranked, MMR diversity selected)."
            ],
        }

    finally:
        # Always clean up — even if an exception is raised mid-retrieval.
        # Guard 1 (delete-before-write) handles the case where this crashes
        # before reaching here, but try/finally ensures it on the happy path too.
        qdrant.delete_collection(_COLLECTION)


# ---------------------------------------------------------------------------
# Local test — run with: python agents/retriever.py
# Uses mock_state() from state.py — no API keys needed.
# Runs multiple back-to-back calls to verify isolation and show per-call timing.
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     import time

#     test_cases = [
#         {
#             "label": "Q1 — intermittent fasting (default mock)",
#             "state": mock_state(),
#         },
#         {
#             "label": "Q2 — same question, second call (isolation check)",
#             "state": mock_state(),
#         },
#         {
#             "label": "Q3 — different question + search results",
#             "state": {
#                 **mock_state(),
#                 "question": "What are the effects of sleep deprivation on cognitive performance?",
#                 "sub_questions": [
#                     "How does sleep deprivation affect memory consolidation?",
#                     "What cognitive tasks are most impaired by lack of sleep?",
#                 ],
#                 "search_results": [
#                     {
#                         "url": "https://pubmed.ncbi.nlm.nih.gov/sleep1",
#                         "title": "Sleep deprivation and working memory",
#                         "snippet": "Even one night of sleep deprivation significantly impairs working memory capacity and attention in healthy adults.",
#                         "source": "pubmed.ncbi.nlm.nih.gov",
#                     },
#                     {
#                         "url": "https://www.nature.com/sleep2",
#                         "title": "REM sleep and memory consolidation",
#                         "snippet": "REM sleep plays a critical role in consolidating declarative and procedural memories acquired during waking hours.",
#                         "source": "nature.com",
#                     },
#                     {
#                         "url": "https://www.nejm.org/sleep3",
#                         "title": "Cognitive effects of chronic partial sleep loss",
#                         "snippet": "Chronic restriction to 6 hours of sleep per night produces cognitive deficits equivalent to two nights of total sleep deprivation.",
#                         "source": "nejm.org",
#                     },
#                 ],
#             },
#         },
#     ]

#     total_start = time.time()

#     for i, tc in enumerate(test_cases, 1):
#         print(f"\n{'='*60}")
#         print(f"{tc['label']}")
#         print(f"{'='*60}")

#         t0 = time.time()
#         result = retrieve_passages(tc["state"])
#         elapsed = time.time() - t0

#         passages = result["retrieved_passages"]
#         print(f"Passages returned: {len(passages)}  |  took {elapsed:.2f}s\n")
#         for j, p in enumerate(passages, 1):
#             print(f"  #{j}  score={p['score']:.3f}  [{p['source']}]")
#             print(f"       Title : {p['title']}")
#             print(f"       Text  : {p['text']}")
#             print()
#         print(f"  Trace: {result['reasoning_trace'][0]}")

#     print(f"\n{'='*60}")
#     print(f"Total wall-clock: {time.time() - total_start:.2f}s  ({len(test_cases)} calls)")
