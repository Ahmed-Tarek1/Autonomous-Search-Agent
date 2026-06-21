"""
main.py — FastAPI Backend
==========================
Owner: Person 5

Fix v2: /research/stream now uses LangGraph's real astream() for live
        per-node events instead of fake sleep-based progress.

Run:
    uvicorn main:app --reload

Endpoints:
    POST /research          → run full pipeline, return final state
    POST /research/stream   → real SSE stream via LangGraph astream()
    GET  /health            → health check
"""

from dotenv import load_dotenv
load_dotenv()

import json
import time
import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pipeline import run_pipeline, get_pipeline
from state import ResearchState

app = FastAPI(
    title="Autonomous Research Assistant",
    description="Multi-agent research pipeline with RAG and conflict detection",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    question: str
    run_eval: bool = True


class ResearchResponse(BaseModel):
    question: str
    sub_questions: list
    final_report: str
    citations: list
    unverified_claims: list
    conflict_detected: bool
    conflict_count: int
    eval_scores: dict | None
    reasoning_trace: list
    latency_seconds: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "research-assistant", "version": "2.0.0"}


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    """Run the full pipeline synchronously. Returns complete report + scores."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start = time.time()
    try:
        state: ResearchState = run_pipeline(req.question, run_eval=req.run_eval)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.time() - start
    conflict_report = state.get("conflict_report", {"has_conflicts": False, "pairs": []})

    return ResearchResponse(
        question=state["question"],
        sub_questions=state["sub_questions"],
        final_report=state["final_report"],
        citations=state["citations"],
        unverified_claims=state["unverified_claims"],
        conflict_detected=conflict_report["has_conflicts"],
        conflict_count=len(conflict_report.get("pairs", [])),
        eval_scores=state.get("eval_scores"),
        reasoning_trace=state["reasoning_trace"],
        latency_seconds=round(latency, 2),
    )


@app.post("/research/stream")
async def research_stream(req: ResearchRequest):
    """
    Stream real pipeline progress as Server-Sent Events via LangGraph astream().

    Fix v2: Uses get_pipeline().astream() so each SSE event corresponds to
            a real node completing — no fake sleep-based progress.

    Event types:
        node_start    — a LangGraph node just started
        node_done     — a node completed; partial state included
        complete      — final report ready
        error         — something went wrong
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    async def event_stream() -> AsyncGenerator[str, None]:
        def emit(event_type: str, data: dict) -> str:
            return f"data: {json.dumps({'type': event_type, 'data': data})}\n\n"

        initial_state = ResearchState(
            question=req.question,
            sub_questions=[],
            reasoning_trace=[],
            search_results=[],
            retrieved_passages=[],
            conflict_report={"has_conflicts": False, "pairs": []},
            final_report="",
            citations=[],
            unverified_claims=[],
            eval_scores=None,
        )

        yield emit("start", {"question": req.question})

        final_state = None
        try:
            # LangGraph astream() yields (node_name, output_state) per node
            async for node_name, output in get_pipeline().astream(
                initial_state, stream_mode="updates"
            ):
                final_state = output

                # Emit a progress event for each completed node
                yield emit("node_done", {
                    "node": node_name,
                    "trace": output.get("reasoning_trace", []),
                    "sub_questions": output.get("sub_questions", []),
                    "search_count": len(output.get("search_results", [])),
                    "passage_count": len(output.get("retrieved_passages", [])),
                    "conflict_detected": output.get(
                        "conflict_report", {}
                    ).get("has_conflicts", False),
                })
                await asyncio.sleep(0)  # yield control to event loop

        except Exception as e:
            yield emit("error", {"message": str(e)})
            return

        if final_state:
            conflict = final_state.get(
                "conflict_report", {"has_conflicts": False, "pairs": []}
            )
            yield emit("complete", {
                "final_report": final_state.get("final_report", ""),
                "citations": final_state.get("citations", []),
                "unverified_claims": final_state.get("unverified_claims", []),
                "conflict_detected": conflict["has_conflicts"],
                "reasoning_trace": final_state.get("reasoning_trace", []),
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
