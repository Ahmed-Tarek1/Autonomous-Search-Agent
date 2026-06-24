# 🔬 Autonomous Research Assistant

> **ITI Agentic AI Course Project**  
> Multi-agent research pipeline built with LangGraph, Qdrant, and RAGAS.

---

## What it does

Takes a research question, autonomously searches the web, retrieves and ranks passages, detects conflicting evidence between sources, and synthesizes a grounded, cited report — all via a 6-node LangGraph pipeline.

**Example:** *"What are the health effects of intermittent fasting?"*  
→ 4 sub-questions → 12 web sources → 5 ranked passages → conflict check → cited Markdown report → RAGAS scores

---

## Architecture

```
User question
     │
     ▼
Query decomposer      → 3–5 focused sub-questions
     │
     ▼
Search agent          → ReAct loop (Tavily), deduplicated results
     │
     ▼
Semantic retriever    → Qdrant + MiniLM + cross-encoder re-ranking
     │
     ▼
Conflict detector     → few-shot LLM classification of source pairs
     │
     ├── no conflict  ──► synthesize_normal
     └── conflict     ──► ynthesize_with_warning
                               │
                               ▼
                          Evaluator  (RAGAS + hallucination rate)
```

**Stack:** LangGraph · Claude (Anthropic) · Tavily · Qdrant · sentence-transformers · RAGAS · FastAPI · Streamlit

---

## Repo structure

```
research_assistant/
├── state.py                  # Shared ResearchState — the team contract
├── pipeline.py               # LangGraph graph (owned by P1)
├── main.py                   # FastAPI backend (owned by P5)
│
├── agents/
│   ├── __init__.py
│   ├── decomposer.py      # Query decomposer
│   ├── search.py          # Search agent (ReAct)
│   ├── retriever.py       # Semantic retriever
│   ├── conflict.py        # Conflict detector + router
│   ├── synthesizer.py     # Report synthesizer
│   └── evaluator.py       # RAGAS evaluation suite
│
├── ui/
│   └── app.py                # Streamlit demo UI
│
├── tests/
│   ├── conftest.py
│   └── test_all_modules.py   # Unit tests for all 6 modules
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/YOUR_ORG/research-assistant.git
cd research-assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
```

Open `.env` and fill in:

### 3. Load environment variables

```bash
# macOS / Linux
export $(cat .env | xargs)

# Or use python-dotenv — already imported in each agent file
```

---

## Running

### Run the full pipeline (quickest test)

```bash
python pipeline.py
```

### Run a specific module in isolation

```bash
# Each agent file has a __main__ block for isolated testing
python agents/decomposer.py
python agents/search.py
python agents/retriever.py
python agents/conflict.py
python agents/synthesizer.py
python agents/evaluator.py
```

### Run the FastAPI backend

```bash
uvicorn main:app --reload
# API docs at http://localhost:8000/docs
```

### Run the Streamlit UI

```bash
streamlit run ui/app.py
```

### Run tests

```bash
pytest tests/ -v
```

> All tests use mock state — no API keys needed for the test suite.

---

## Shared state contract

```python
class ResearchState(TypedDict):
    question:            str          # Input — never overwritten
    sub_questions:       List[str]    # P1 writes
    reasoning_trace:     List[str]    # All modules append
    search_results:      List[SearchResult]    # P2 writes
    retrieved_passages:  List[Passage]         # P3 writes
    conflict_report:     ConflictReport        # P4 writes
    final_report:        str                   # P5 writes
    citations:           List[str]             # P5 writes
    unverified_claims:   List[str]             # P5 writes
    eval_scores:         Optional[dict]        # P6 writes
```

---

## Evaluation metrics

Person 6 runs the benchmark suite against 5 research questions, 3 runs each:

| Metric | Target | How computed |
|---|---|---|
| Faithfulness | > 0.85 | RAGAS — claims grounded in retrieved passages |
| Answer relevancy | > 0.80 | RAGAS — report answers the question |
| Hallucination rate | < 0.10 | `unverified_claims / total_claims` |
| Latency | < 90s | Wall-clock time for full pipeline |

Compare against **baseline**: same 5 questions through a single bare LLM call with no pipeline.

---

## Demo script (Day 7)

1. **P1** — opens demo, shows pipeline diagram (2 min)
2. **P2** — submits question 1 (clean), narrates live ReAct loop in the UI
3. **P3** — explains why 5 passages were selected from 40 candidates
4. **P4** — submits question 2 (conflicting sources), narrates the conflict detection moment
5. **P5** — shows final report: inline citations, "Conflicting Evidence" section, PDF download
6. **P6** — shows benchmark results slide: pipeline vs baseline scores

**Best demo question for conflict detection:**  
*"Is intermittent fasting more effective than continuous caloric restriction for weight loss?"*  
— real scientific literature genuinely disagrees on this.

---

## Troubleshooting

**Qdrant connection error**  
→ Check `QDRANT_URL` and `QDRANT_API_KEY` in `.env`. Use `:memory:` mode for local testing (set automatically if `QDRANT_URL` is not set).

**Tavily rate limit**  
→ Pre-cache demo question results by running `python pipeline.py` before the demo. Results are deterministic within a session.

**Token limit exceeded in synthesizer**  
→ P3's compression step reduces each passage to 2–3 sentences before feeding P5. If still hitting limits, reduce `TOP_K_FINAL` in `p3_retriever.py` from 5 to 3.

**RAGAS import error**  
→ `pip install ragas datasets`. If still failing, P6 can fall back to the custom hallucination rate metric alone.

---

## Contributing

Each person works on their own branch:

```bash
git checkout -b feat/p2-search-agent
# ... make changes ...
git push origin feat/p2-search-agent
# open a PR to main
```

Branch naming: `feat/p{N}-{module-name}` (e.g. `feat/p4-conflict-detector`)
