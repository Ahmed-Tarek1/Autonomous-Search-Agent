"""
agents/decomposer.py — Query Decomposer + Orchestrator
==========================================================
Reads:  state["question"]
Writes: state["sub_questions"], state["reasoning_trace"]

Contract:
  - sub_questions: List[str], 3–5 items, each independently searchable
  - reasoning_trace: plain List[str] with exactly 1 new entry ("[P1] ...")
    LangGraph appends it via the operator.add reducer in state.py.

Run in isolation:
    python agents/decomposer.py
"""

from state import ResearchState, mock_state
from agents.prompts.decomposer_prompt import DECOMPOSER_PROMPT
from helpers.llm_caller import LLMCaller
from dotenv import load_dotenv
import os
import yaml
import json

def load_variables():
    file = open("./configs.yaml")
    configs = yaml.safe_load(file)
    file.close()
    load_dotenv()
    
    return configs

configs = load_variables()

decomposer: LLMCaller = LLMCaller(api_key=os.getenv("GROQ_API_KEY"), model=configs["MAIN_MODEL"],  system_prompt=DECOMPOSER_PROMPT, identifier="Decomposer", verbose=False)


def decompose_query(state: ResearchState) -> ResearchState:
    
    """
    TODO (Person 1): Implement query decomposition using Claude.
    - Call LLM with DECOMPOSE_SYSTEM prompt to generate 3–5 sub-questions
    - Deduplicate (case-insensitive exact match)
    - Run a self-critique LLM pass to sharpen vague sub-questions
    - Return exactly one reasoning_trace entry: "[P1] Decomposed '...' into N sub-questions: ..."
    """
    global decomposer
    question: str = state["question"]
    MIN_QUESTIONS = configs["MIN_QUESTIONS"]
    MAX_QUESTIONS = configs["MAX_QUESTIONS"]
    decomposed_query: list[str] = decomposer.call(question=question, maximum=MAX_QUESTIONS, minimum=MIN_QUESTIONS)
    

    sub_questions = json.loads(decomposed_query)

    # Mock output so pipeline runs end-to-end from Day 1
    # sub_questions = [
    #     f"What does research say about {question.lower().rstrip('?')} and health outcomes?",
    #     f"What are the mechanisms behind {question.lower().rstrip('?')}?",
    #     f"What are the risks or side effects related to {question.lower().rstrip('?')}?",
    #     f"What do meta-analyses conclude about {question.lower().rstrip('?')}?",
    # ]
    return {
    "sub_questions": sub_questions,
    "reasoning_trace": [
        f"[P1] Decomposed '{question}' into {len(sub_questions)} sub-questions"
    ],
}


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    state = mock_state()
    state["question"] = "What are the health effects of intermittent fasting?"
    state["sub_questions"] = []
    state["reasoning_trace"] = []

    result = decompose_query(state)
    print("Sub-questions:")
    for i, q in enumerate(result["sub_questions"], 1):
        print(f"  {i}. {q}")
    print("\nNew trace entry:", result["reasoning_trace"])