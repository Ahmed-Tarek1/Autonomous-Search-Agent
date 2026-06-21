"""
agents package
==============
Fix v2: Imports are explicit so teammates can import individual agents
without loading all LLM clients. Only import what you need.
"""
from agents.p1_decomposer import decompose_query
from agents.p2_search import search_agent
from agents.p3_retriever import retrieve_passages
from agents.p4_conflict import detect_conflicts, route_on_conflict
from agents.p5_synthesizer import synthesize_report
from agents.p6_evaluator import evaluate_state

__all__ = [
    "decompose_query",
    "search_agent",
    "retrieve_passages",
    "detect_conflicts",
    "route_on_conflict",
    "synthesize_report",
    "evaluate_state",
]
