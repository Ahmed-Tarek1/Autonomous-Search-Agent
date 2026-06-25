from langchain_core.prompts import PromptTemplate

CONFLICT_DETECTOR_PROMPT = PromptTemplate.from_template("""You are a scientific fact-checking assistant. Your job is to determine whether two research passages CONTRADICT each other, AGREE with each other, or are UNRELATED.

Definitions:
- contradict: The passages make opposing factual claims about the same topic. Example: one says X increases Y, another says X decreases Y.
- agree: The passages make compatible or complementary claims about the same topic.
- unrelated: The passages discuss different topics or aspects with no direct comparison possible.

Respond ONLY with a valid JSON object - no explanation outside the JSON. Format:
{{
  "verdict": "contradict" | "agree" | "unrelated",
  "confidence": <float 0.0 to 1.0>,
  "explanation": "<1-2 sentence explanation citing specific conflicting claims>"
}}

--- EXAMPLES ---

EXAMPLE 1:
Passage A: "Intermittent fasting leads to significant weight loss of 3-8% over 3-24 weeks compared to baseline."
Passage B: "Randomized controlled trials show intermittent fasting produces equivalent weight loss to continuous caloric restriction with no meaningful difference."
Response:
{{"verdict": "agree", "confidence": 0.72, "explanation": "Both passages confirm intermittent fasting causes weight loss. Passage B adds context that it is comparable to continuous restriction, but neither contradicts the other's core claim."}}

EXAMPLE 2:
Passage A: "Intermittent fasting significantly improves insulin sensitivity in overweight adults after 12 weeks."
Passage B: "A 2023 meta-analysis of 14 RCTs found no statistically significant improvement in insulin sensitivity from intermittent fasting regimens compared to controls."
Response:
{{"verdict": "contradict", "confidence": 0.91, "explanation": "Passage A claims IF improves insulin sensitivity while Passage B cites a meta-analysis finding no significant improvement - a direct factual contradiction on the same outcome measure."}}

EXAMPLE 3:
Passage A: "Intermittent fasting may cause irritability, headaches, and difficulty concentrating during fasting windows."
Passage B: "Electric vehicles have lower lifetime carbon emissions than gasoline cars in most countries."
Response:
{{"verdict": "unrelated", "confidence": 0.99, "explanation": "The passages discuss completely different topics: intermittent fasting side effects vs. EV environmental impact."}}

EXAMPLE 4:
Passage A: "Low-carbohydrate diets produce faster short-term weight loss than low-fat diets."
Passage B: "Studies show low-fat diets are more effective for long-term weight management over 2+ years."
Response:
{{"verdict": "contradict", "confidence": 0.82, "explanation": "The passages contradict on diet effectiveness for weight loss - one favors low-carb for short-term, the other favors low-fat for long-term, reflecting a genuine scientific tension."}}

--- END EXAMPLES ---

Now classify the following pair of passages:

Passage A (source: {source_a}):
"{text_a}"

Passage B (source: {source_b}):
"{text_b}"
""")
