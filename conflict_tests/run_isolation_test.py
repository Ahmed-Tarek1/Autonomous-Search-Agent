import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from state import Passage, mock_state
from agents.conflict import detect_conflicts, route_on_conflict

print("=" * 60)
print("P4 CONFLICT DETECTOR - Full Isolation Test")
print("=" * 60)

# ---------------------------------------------------------------------------
# Test 1: Standard mock state (no conflicts expected)
# The two default passages are from different topics so should not conflict.
# ---------------------------------------------------------------------------
print("\n[Test 1] Standard mock state (agreeing passages, no conflicts expected)...")
state = mock_state()
result = detect_conflicts(state)
report = result["conflict_report"]
print(f"  has_conflicts:       {report['has_conflicts']}")
print(f"  Contradiction pairs: {len(report['pairs'])}")
print(f"  Route ->             {route_on_conflict(result)}")
print(f"  Trace:               {result['reasoning_trace'][-1]}")

expected_route_1 = "synthesize_normal"
if route_on_conflict(result) == expected_route_1:
    print(f"  [PASS] Correctly routed to '{expected_route_1}'")
else:
    print(f"  [WARN] Expected '{expected_route_1}', got '{route_on_conflict(result)}'")

# ---------------------------------------------------------------------------
# Test 2: Injected conflicting passages
# Passage 1 and Passage 2 directly contradict on insulin sensitivity.
# Passage 3 is on a different aspect (weight loss) - should agree or be unrelated.
# Expected: has_conflicts=True, route -> synthesize_warning
# ---------------------------------------------------------------------------
print("\n[Test 2] Injecting known-conflicting passages (insulin sensitivity debate)...")

conflicting_state = mock_state()
conflicting_state["retrieved_passages"] = [
    Passage(
        text="Intermittent fasting significantly improves insulin sensitivity in overweight adults after 12 weeks of treatment.",
        url="https://pubmed.ncbi.nlm.nih.gov/study1",
        title="IF and Insulin Sensitivity: A Positive RCT",
        score=0.91,
        source="pubmed.ncbi.nlm.nih.gov",
    ),
    Passage(
        text="A 2023 meta-analysis of 14 randomized controlled trials found no statistically significant improvement in insulin sensitivity from intermittent fasting regimens compared to continuous caloric restriction controls.",
        url="https://www.nejm.org/study2",
        title="IF Meta-Analysis: Null Result on Insulin",
        score=0.88,
        source="nejm.org",
    ),
    Passage(
        text="Participants in intermittent fasting groups showed 3-8% body weight reduction over 3-24 weeks across multiple systematic reviews.",
        url="https://www.thelancet.com/study3",
        title="IF and Weight Loss: Systematic Review",
        score=0.85,
        source="thelancet.com",
    ),
]

result2 = detect_conflicts(conflicting_state)
report2 = result2["conflict_report"]
print(f"  has_conflicts:       {report2['has_conflicts']}")
print(f"  Contradiction pairs: {len(report2['pairs'])}")

if report2["pairs"]:
    for i, pair in enumerate(report2["pairs"], 1):
        print(f"\n  --- Conflict Pair {i} ---")
        print(f"    Source A:    {pair['passage_a']['source']}")
        print(f"    Source B:    {pair['passage_b']['source']}")
        print(f"    Verdict:     {pair['verdict']}")
        print(f"    Confidence:  {pair['confidence']:.2f}")
        print(f"    Explanation: {pair['explanation']}")

print(f"\n  Route -> {route_on_conflict(result2)}")
print(f"  Trace:   {result2['reasoning_trace'][-1]}")

expected_route_2 = "synthesize_warning"
if route_on_conflict(result2) == expected_route_2:
    print(f"  [PASS] Correctly routed to '{expected_route_2}'")
else:
    print(f"  [NOTE] Expected '{expected_route_2}' but got '{route_on_conflict(result2)}'")
    print(f"         This may happen if Gemini classified the pair as 'agree' or confidence < 0.75.")

print("\n" + "=" * 60)
print("Isolation test complete.")
print("=" * 60)
