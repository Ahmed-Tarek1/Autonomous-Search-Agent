"""
agents package
==============
Imports are explicit so teammates can import individual agents
without loading all LLM clients. Only import what you need.
"""
# from agents.decomposer import decompose_query
# from agents.search import search_agent
# from agents.retriever import retrieve_passages
# from agents.conflict import detect_conflicts, route_on_conflict
# from agents.synthesizer import synthesize_report
# from agents.evaluator import evaluate_state

__all__ = [
    "decompose_query",
    "search_agent",
    "retrieve_passages",
    "detect_conflicts",
    "route_on_conflict",
    "synthesize_report",
    "evaluate_state",
]
