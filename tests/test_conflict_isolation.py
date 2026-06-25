"""
tests/test_conflict_isolation.py
==================================
Isolation test for Conflict Detector.
Tests both the no-conflict path and the conflict-detected path.
No API key required - the LangChain chain is monkeypatched.

Run from repo root:
    pytest tests/test_conflict_isolation.py -v
"""
import pytest
from state import Passage, ConflictReport, mock_state
import os

# Prevent crash when loading agents if GROQ_API_KEY is not in .env
os.environ["GROQ_API_KEY"] = "mock_key_for_testing_so_chatgroq_doesnt_crash"

# ---------------------------------------------------------------------------
# Helper: build a fake conflict_chain.invoke() response
# ---------------------------------------------------------------------------

def _fake_chain(verdict: str, confidence: float, explanation: str):
    """Returns a fake chain object with an invoke() method."""
    class FakeChain:
        def invoke(self, *args, **kwargs):
            return {
                "verdict": verdict,
                "confidence": confidence,
                "explanation": explanation,
            }
    return FakeChain()


# ---------------------------------------------------------------------------
# Test: fewer than 2 passages — skip immediately
# ---------------------------------------------------------------------------

class TestConflictEdgeCases:
    def test_single_passage_no_check(self):
        """With only 1 passage there are no pairs - should return no conflicts."""
        from agents.conflict import detect_conflicts
        state = mock_state()
        state["retrieved_passages"] = state["retrieved_passages"][:1]
        result = detect_conflicts(state)
        assert result["conflict_report"]["has_conflicts"] is False
        assert "[Conflict Detector]" in result["reasoning_trace"][-1]

    def test_empty_passages_no_check(self):
        """With 0 passages should return no conflicts gracefully."""
        from agents.conflict import detect_conflicts
        state = mock_state()
        state["retrieved_passages"] = []
        result = detect_conflicts(state)
        assert result["conflict_report"]["has_conflicts"] is False


# ---------------------------------------------------------------------------
# Test: normal path — passages agree
# ---------------------------------------------------------------------------

class TestNoConflict:
    def test_agree_verdict_produces_no_conflict(self, monkeypatch):
        """LLM returns 'agree' -> has_conflicts must be False."""
        import agents.conflict as mod
        monkeypatch.setattr(mod, "conflict_chain", _fake_chain("agree", 0.90, "Both agree."))

        result = mod.detect_conflicts(mock_state())
        assert result["conflict_report"]["has_conflicts"] is False
        assert len(result["conflict_report"]["pairs"]) == 0

    def test_low_confidence_contradict_ignored(self, monkeypatch):
        """contradict with confidence < threshold must NOT count as conflict."""
        import agents.conflict as mod
        monkeypatch.setattr(mod, "conflict_chain", _fake_chain("contradict", 0.50, "Maybe conflict."))

        result = mod.detect_conflicts(mock_state())
        assert result["conflict_report"]["has_conflicts"] is False

    def test_unrelated_verdict_produces_no_conflict(self, monkeypatch):
        """LLM returns 'unrelated' -> has_conflicts must be False."""
        import agents.conflict as mod
        monkeypatch.setattr(mod, "conflict_chain", _fake_chain("unrelated", 0.99, "Different topics."))

        result = mod.detect_conflicts(mock_state())
        assert result["conflict_report"]["has_conflicts"] is False


# ---------------------------------------------------------------------------
# Test: conflict path — high-confidence contradiction detected
# ---------------------------------------------------------------------------

class TestConflictDetected:
    def _conflicting_state(self):
        """State with 3 passages where pair (1,2) should contradict."""
        state = mock_state()
        state["retrieved_passages"] = [
            Passage(
                text="Intermittent fasting significantly improves insulin sensitivity after 12 weeks.",
                url="https://pubmed.ncbi.nlm.nih.gov/study1",
                title="IF and Insulin: Positive RCT",
                score=0.91,
                source="pubmed.ncbi.nlm.nih.gov",
            ),
            Passage(
                text="A meta-analysis of 14 RCTs found no significant improvement in insulin sensitivity from intermittent fasting.",
                url="https://www.nejm.org/study2",
                title="IF Meta-Analysis: Null Result",
                score=0.88,
                source="nejm.org",
            ),
            Passage(
                text="Participants showed 3-8% body weight reduction over 3-24 weeks across systematic reviews.",
                url="https://www.thelancet.com/study3",
                title="IF and Weight Loss: Systematic Review",
                score=0.85,
                source="thelancet.com",
            ),
        ]
        return state

    def test_high_confidence_contradict_flagged(self, monkeypatch):
        """High-confidence contradict -> has_conflicts=True, pair stored."""
        import agents.conflict as mod
        monkeypatch.setattr(mod, "conflict_chain", _fake_chain("contradict", 0.91, "Direct contradiction."))

        result = mod.detect_conflicts(self._conflicting_state())
        assert result["conflict_report"]["has_conflicts"] is True
        assert len(result["conflict_report"]["pairs"]) >= 1

    def test_conflict_pair_schema(self, monkeypatch):
        """Each ConflictPair must have all required fields."""
        import agents.conflict as mod
        monkeypatch.setattr(mod, "conflict_chain", _fake_chain("contradict", 0.91, "Contradicting."))

        result = mod.detect_conflicts(self._conflicting_state())
        for pair in result["conflict_report"]["pairs"]:
            assert "passage_a" in pair
            assert "passage_b" in pair
            assert "verdict" in pair
            assert "confidence" in pair
            assert "explanation" in pair
            assert pair["verdict"] == "contradict"
            assert pair["confidence"] >= 0.75


# ---------------------------------------------------------------------------
# Test: reasoning trace contract
# ---------------------------------------------------------------------------

class TestReasoningTrace:
    def test_trace_is_list_of_one_entry(self, monkeypatch):
        """Node must return exactly 1 new reasoning_trace entry (LangGraph operator.add)."""
        import agents.conflict as mod
        monkeypatch.setattr(mod, "conflict_chain", _fake_chain("agree", 0.9, "Fine."))

        state = mock_state()
        result = mod.detect_conflicts(state)
        assert isinstance(result["reasoning_trace"], list)
        assert len(result["reasoning_trace"]) == 1
        assert result["reasoning_trace"][0].startswith("[Conflict Detector]")


# ---------------------------------------------------------------------------
# Test: router
# ---------------------------------------------------------------------------

class TestRouter:
    def test_routes_to_normal(self):
        from agents.conflict import route_on_conflict
        state = mock_state()
        state["conflict_report"] = ConflictReport(has_conflicts=False, pairs=[])
        assert route_on_conflict(state) == "synthesize_normal"

    def test_routes_to_warning(self):
        from agents.conflict import route_on_conflict
        state = mock_state()
        state["conflict_report"] = ConflictReport(has_conflicts=True, pairs=[])
        assert route_on_conflict(state) == "synthesize_warning"
