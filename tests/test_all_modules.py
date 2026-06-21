"""
tests/test_all_modules.py — Unit tests for all 6 modules
==========================================================
Every test uses mock_state() so no API keys are needed.

Fix v2:
  - P1 tests monkeypatch both LLM calls (decompose + critique) separately
  - All reasoning_trace assertions updated for operator.add pattern
    (nodes now return ["entry"] not state["trace"] + ["entry"])

Run:
    pytest tests/ -v
"""

import pytest
from state import (
    ResearchState, ConflictReport, mock_state,
    SearchResult, Passage, ConflictPair,
)


# ---------------------------------------------------------------------------
# P1 — Query Decomposer
# ---------------------------------------------------------------------------

class TestDecomposer:
    def _make_responses(self, *contents):
        """Return an iterator of fake LLM responses."""
        from itertools import cycle
        responses = iter([
            type('R', (), {'content': c})() for c in contents
        ])
        return lambda msgs: next(responses)

    def test_output_keys_present(self, monkeypatch):
        """sub_questions and reasoning_trace must be in output."""
        from agents.p1_decomposer import decompose_query
        import agents.p1_decomposer as mod

        # Two calls: decompose + critique
        monkeypatch.setattr(mod.llm, "invoke", self._make_responses(
            '["Sub-question 1?", "Sub-question 2?", "Sub-question 3?"]',
            '["Sub-question 1?", "Sub-question 2?", "Sub-question 3?"]',
        ))

        state = mock_state()
        state["sub_questions"] = []
        result = decompose_query(state)

        assert "sub_questions" in result
        assert "reasoning_trace" in result
        assert len(result["sub_questions"]) >= 1

    def test_no_duplicates(self, monkeypatch):
        """Duplicate sub-questions should be removed before critique."""
        from agents.p1_decomposer import decompose_query
        import agents.p1_decomposer as mod

        # First call returns duplicates; critique call returns deduped version
        monkeypatch.setattr(mod.llm, "invoke", self._make_responses(
            '["Same question?", "Same question?", "Different question?"]',
            '["Same question?", "Different question?"]',
        ))

        state = mock_state()
        state["sub_questions"] = []
        result = decompose_query(state)

        assert len(result["sub_questions"]) == len(set(result["sub_questions"]))

    def test_trace_is_list_of_one_entry(self, monkeypatch):
        """
        Fix v2: nodes return ["entry"] (plain list) for operator.add.
        reasoning_trace in node output should be exactly 1 new entry.
        """
        from agents.p1_decomposer import decompose_query
        import agents.p1_decomposer as mod

        monkeypatch.setattr(mod.llm, "invoke", self._make_responses(
            '["Q1?", "Q2?"]',
            '["Q1?", "Q2?"]',
        ))

        state = mock_state()
        state["sub_questions"] = []
        state["reasoning_trace"] = []
        result = decompose_query(state)

        # The node returns a plain list with one new entry
        assert isinstance(result["reasoning_trace"], list)
        assert len(result["reasoning_trace"]) == 1
        assert "[P1]" in result["reasoning_trace"][0]


# ---------------------------------------------------------------------------
# P2 — Search Agent
# ---------------------------------------------------------------------------

class TestSearchAgent:
    def test_output_keys_present(self, monkeypatch):
        from agents.p2_search import search_agent
        import agents.p2_search as mod

        monkeypatch.setattr(mod.web_search, "invoke", lambda inputs: [
            {"url": f"https://example{i}.com/", "title": f"Result {i}",
             "snippet": "Some content.", "source": f"example{i}.com"}
            for i in range(3)
        ])
        monkeypatch.setattr(mod.llm, "invoke",
                            lambda msgs: type('R', (), {'content': 'STOP'})())

        state = mock_state()
        state["search_results"] = []
        result = search_agent(state)

        assert "search_results" in result
        assert isinstance(result["search_results"], list)

    def test_deduplication(self, monkeypatch):
        """Duplicate URLs must not appear in search_results."""
        from agents.p2_search import search_agent
        import agents.p2_search as mod

        monkeypatch.setattr(mod.web_search, "invoke", lambda inputs: [
            {"url": "https://same.com/page", "title": "Same",
             "snippet": "x", "source": "same.com"},
            {"url": "https://same.com/page", "title": "Same",
             "snippet": "x", "source": "same.com"},
        ])
        monkeypatch.setattr(mod.llm, "invoke",
                            lambda msgs: type('R', (), {'content': 'STOP'})())

        state = mock_state()
        state["search_results"] = []
        result = search_agent(state)

        urls = [r["url"] for r in result["search_results"]]
        assert len(urls) == len(set(urls)), "Duplicate URLs found"

    def test_trace_is_list(self, monkeypatch):
        """reasoning_trace returned by node must be a plain list."""
        from agents.p2_search import search_agent
        import agents.p2_search as mod

        monkeypatch.setattr(mod.web_search, "invoke", lambda inputs: [])
        monkeypatch.setattr(mod.llm, "invoke",
                            lambda msgs: type('R', (), {'content': 'STOP'})())

        state = mock_state()
        result = search_agent(state)
        assert isinstance(result["reasoning_trace"], list)


# ---------------------------------------------------------------------------
# P3 — Retriever
# ---------------------------------------------------------------------------

class TestRetriever:
    def _mock_p3(self, monkeypatch):
        import numpy as np

        class FakeEmbedder:
            def encode(self, texts, show_progress_bar=False):
                if isinstance(texts, str):
                    return np.zeros(384)
                return np.zeros((len(texts), 384))

        class FakeHit:
            def __init__(self, i):
                self.id = str(i)
                self.score = 0.9 - i * 0.01
                self.payload = {
                    "text": f"Passage {i}", "url": f"https://src{i}.com/",
                    "title": f"Title {i}", "source": f"src{i}.com",
                }

        class FakeQdrant:
            def get_collections(self):
                class C: collections = []
                return C()
            def create_collection(self, **kw): pass
            def upsert(self, **kw): pass
            def search(self, **kw): return [FakeHit(i) for i in range(8)]
            def delete_collection(self, *a): pass

        class FakeReranker:
            def predict(self, pairs): return [0.9 - i * 0.01 for i in range(len(pairs))]

        import agents.p3_retriever as mod
        monkeypatch.setattr(mod, "EMBEDDER", FakeEmbedder())
        monkeypatch.setattr(mod, "RERANKER", FakeReranker())
        monkeypatch.setattr(mod, "qdrant", FakeQdrant())

    def test_max_5_passages(self, monkeypatch):
        from agents.p3_retriever import retrieve_passages
        self._mock_p3(monkeypatch)

        state = mock_state()
        result = retrieve_passages(state)
        assert len(result["retrieved_passages"]) <= 5

    def test_passage_schema(self, monkeypatch):
        from agents.p3_retriever import retrieve_passages
        self._mock_p3(monkeypatch)

        state = mock_state()
        result = retrieve_passages(state)
        for p in result["retrieved_passages"]:
            for key in ("text", "url", "score", "source", "title"):
                assert key in p, f"Missing key '{key}' in passage"

    def test_trace_is_list(self, monkeypatch):
        from agents.p3_retriever import retrieve_passages
        self._mock_p3(monkeypatch)

        state = mock_state()
        result = retrieve_passages(state)
        assert isinstance(result["reasoning_trace"], list)
        assert len(result["reasoning_trace"]) == 1


# ---------------------------------------------------------------------------
# P4 — Conflict Detector
# ---------------------------------------------------------------------------

class TestConflictDetector:
    def test_no_conflict(self, monkeypatch):
        from agents.p4_conflict import detect_conflicts
        import agents.p4_conflict as mod

        monkeypatch.setattr(mod.llm, "invoke", lambda msgs:
            type('R', (), {'content': '{"verdict":"agree","confidence":0.95,"explanation":"Both agree."}'})())

        result = detect_conflicts(mock_state())
        assert result["conflict_report"]["has_conflicts"] is False

    def test_conflict_detected(self, monkeypatch):
        from agents.p4_conflict import detect_conflicts
        import agents.p4_conflict as mod

        monkeypatch.setattr(mod.llm, "invoke", lambda msgs:
            type('R', (), {'content': '{"verdict":"contradict","confidence":0.90,"explanation":"They disagree."}'})())

        result = detect_conflicts(mock_state())
        assert result["conflict_report"]["has_conflicts"] is True
        assert len(result["conflict_report"]["pairs"]) >= 1

    def test_low_confidence_ignored(self, monkeypatch):
        from agents.p4_conflict import detect_conflicts
        import agents.p4_conflict as mod

        monkeypatch.setattr(mod.llm, "invoke", lambda msgs:
            type('R', (), {'content': '{"verdict":"contradict","confidence":0.50,"explanation":"Maybe."}'})())

        result = detect_conflicts(mock_state())
        assert result["conflict_report"]["has_conflicts"] is False

    def test_trace_is_list(self, monkeypatch):
        from agents.p4_conflict import detect_conflicts
        import agents.p4_conflict as mod

        monkeypatch.setattr(mod.llm, "invoke", lambda msgs:
            type('R', (), {'content': '{"verdict":"agree","confidence":0.9,"explanation":"OK."}'})())

        result = detect_conflicts(mock_state())
        assert isinstance(result["reasoning_trace"], list)
        assert len(result["reasoning_trace"]) == 1


# ---------------------------------------------------------------------------
# P4 — Router
# ---------------------------------------------------------------------------

class TestRouter:
    def test_routes_normal(self):
        from agents.p4_conflict import route_on_conflict
        state = mock_state()
        state["conflict_report"] = ConflictReport(has_conflicts=False, pairs=[])
        assert route_on_conflict(state) == "synthesize_normal"

    def test_routes_warning(self):
        from agents.p4_conflict import route_on_conflict
        state = mock_state()
        state["conflict_report"] = ConflictReport(has_conflicts=True, pairs=[])
        assert route_on_conflict(state) == "synthesize_warning"


# ---------------------------------------------------------------------------
# P5 — Synthesizer
# ---------------------------------------------------------------------------

class TestSynthesizer:
    def _mock_p5(self, monkeypatch, report="## Report\n\nFinding [Source 1]."):
        import agents.p5_synthesizer as mod

        call_count = {"n": 0}
        def fake_invoke(messages):
            call_count["n"] += 1
            return type('R', (), {'content': report})()

        def fake_check(messages):
            return type('R', (), {'content': '[]'})()

        monkeypatch.setattr(mod.llm, "invoke", fake_invoke)
        monkeypatch.setattr(mod.llm_check, "invoke", fake_check)

    def test_output_keys_present(self, monkeypatch):
        from agents.p5_synthesizer import synthesize_report
        self._mock_p5(monkeypatch)

        result = synthesize_report(mock_state())
        assert "final_report" in result
        assert "citations" in result
        assert "unverified_claims" in result
        assert len(result["final_report"]) > 0

    def test_citations_match_passages(self, monkeypatch):
        from agents.p5_synthesizer import synthesize_report
        self._mock_p5(monkeypatch)

        state = mock_state()
        result = synthesize_report(state)
        assert len(result["citations"]) == len(state["retrieved_passages"])

    def test_trace_is_list(self, monkeypatch):
        from agents.p5_synthesizer import synthesize_report
        self._mock_p5(monkeypatch)

        result = synthesize_report(mock_state())
        assert isinstance(result["reasoning_trace"], list)
        assert len(result["reasoning_trace"]) == 1


# ---------------------------------------------------------------------------
# P6 — Evaluator
# ---------------------------------------------------------------------------

class TestEvaluator:
    def test_scores_dict_returned(self, monkeypatch):
        from agents.p6_evaluator import evaluate_state
        import agents.p6_evaluator as mod

        monkeypatch.setattr(mod.llm, "invoke",
                            lambda msgs: type('R', (), {'content': '5'})())
        monkeypatch.setattr(mod, "compute_ragas_scores",
                            lambda s: {"faithfulness": 0.9, "answer_relevancy": 0.85})

        state = mock_state()
        state["final_report"] = "Some report [Source 1]."
        state["unverified_claims"] = []

        scores = evaluate_state(state, latency_seconds=10.0)
        assert "faithfulness" in scores
        assert "hallucination_rate" in scores
        assert "latency_seconds" in scores

    def test_zero_hallucination_for_clean_report(self, monkeypatch):
        from agents.p6_evaluator import evaluate_state
        import agents.p6_evaluator as mod

        monkeypatch.setattr(mod.llm, "invoke",
                            lambda msgs: type('R', (), {'content': '3'})())
        monkeypatch.setattr(mod, "compute_ragas_scores",
                            lambda s: {"faithfulness": None, "answer_relevancy": None})

        state = mock_state()
        state["final_report"] = "A report [Source 1]."
        state["unverified_claims"] = []

        scores = evaluate_state(state)
        assert scores["hallucination_rate"] == 0.0
