import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import json
import re

api_key = os.environ.get("GEMINI_API_KEY", "")
print(f"[DEBUG] GEMINI_API_KEY present: {bool(api_key)}")
print(f"[DEBUG] Key prefix: {api_key[:12]}..." if api_key else "[DEBUG] NO KEY FOUND in .env")

if not api_key:
    print("\n[ACTION REQUIRED]")
    print("  1. Go to: https://aistudio.google.com/app/apikey")
    print("  2. Create an API key")
    print("  3. Create a .env file in the repo root with:")
    print("     GEMINI_API_KEY=your-key-here")
    sys.exit(1)

try:
    from google import genai
    from agents.conflict import FEW_SHOT_SYSTEM, GEMINI_MODEL

    client = genai.Client(api_key=api_key)

    # The exact pair that SHOULD produce "contradict"
    passage_a = "Intermittent fasting significantly improves insulin sensitivity in overweight adults after 12 weeks of treatment."
    passage_b = "A 2023 meta-analysis of 14 randomized controlled trials found no statistically significant improvement in insulin sensitivity from intermittent fasting regimens compared to continuous caloric restriction controls."

    full_prompt = (
        FEW_SHOT_SYSTEM
        + "\n\n"
        + f'Passage A (source: pubmed.ncbi.nlm.nih.gov):\n"{passage_a}"\n\n'
        + f'Passage B (source: nejm.org):\n"{passage_b}"'
    )

    print(f"\n[DEBUG] Sending to Gemini ({GEMINI_MODEL})...")
    print(f"  Passage A: {passage_a[:70]}...")
    print(f"  Passage B: {passage_b[:70]}...")

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=full_prompt,
    )

    raw = response.text.strip()
    print(f"\n[DEBUG] Raw Gemini response:\n{raw}\n")

    json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if json_match:
        parsed = json.loads(json_match.group())
        print(f"[DEBUG] Parsed result:")
        print(f"  Verdict:     {parsed.get('verdict')}")
        print(f"  Confidence:  {parsed.get('confidence')}")
        print(f"  Explanation: {parsed.get('explanation')}")
        print("\n[EXPECTED] verdict=contradict, confidence >= 0.75")
        if parsed.get('verdict') == 'contradict' and float(parsed.get('confidence', 0)) >= 0.75:
            print("[PASS] Conflict correctly detected!")
        else:
            print("[NOTE] Verdict was not a high-confidence contradict - check the passages or prompt.")
    else:
        print("[WARN] No JSON found in response - model may have returned unexpected format")

except ImportError:
    print("\n[ERROR] google-genai package not installed.")
    print("  Run: pip install google-genai")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
