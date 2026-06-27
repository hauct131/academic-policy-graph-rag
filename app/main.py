import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables if .env file exists
load_dotenv()

from app.policy_qa_service import PolicyQAService, log_request_if_enabled

qa_service = PolicyQAService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load resources at startup
    await asyncio.to_thread(qa_service.load_resources)
    yield

app = FastAPI(
    title="Academic Policy Graph RAG API",
    description="A Graph RAG-based Academic Policy Assistant for student guidance.",
    version="1.0.0",
    lifespan=lifespan
)

import time
from collections import defaultdict
from fastapi import Request
from fastapi.responses import JSONResponse

RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))
_rate_limit_window_seconds = 60
_request_log: dict[str, list[float]] = defaultdict(list)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/policy/ask":
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        timestamps = _request_log[client_ip]
        # Xoá các timestamp cũ hơn window
        cutoff = now - _rate_limit_window_seconds
        timestamps[:] = [t for t in timestamps if t > cutoff]
        if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."},
                headers={"Retry-After": str(_rate_limit_window_seconds)}
            )
        timestamps.append(now)
    return await call_next(request)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Student policy question (max 2000 characters)"
    )
    top_k: int = Field(5, ge=1, le=10, description="Top K chunks to retrieve")
    show_evidence_text: bool = Field(False, description="Whether to append raw evidence text")


@app.get("/api/v1/graph-rag/health")
async def health_check():
    return {"status": "ok"}


@app.post("/policy/ask")
async def ask_question_endpoint(request: AskRequest):
    # Validate empty question
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=422,
            detail="question must not be empty or whitespace only"
        )

    # Check/reload service if not initialized
    if not qa_service.initialized:
        await asyncio.to_thread(qa_service.load_resources)
        
    if not qa_service.initialized:
        details = ", ".join(qa_service.missing_resources)
        log_request_if_enabled(
            question_len=len(request.question),
            status="503",
            top_k=request.top_k,
            warnings=qa_service.missing_resources
        )
        raise HTTPException(
            status_code=503,
            detail=f"Service Unavailable: {details}"
        )

    try:
        answer, metadata, warnings = await asyncio.to_thread(
            qa_service.get_qa_response,
            question=request.question,
            top_k=request.top_k,
            show_evidence_text=request.show_evidence_text
        )
        
        log_request_if_enabled(
            question_len=len(request.question),
            status="ok",
            top_k=request.top_k,
            warnings=warnings
        )
        
        return {
            "answer": answer,
            "question": request.question,
            "top_k": request.top_k,
            "status": "ok",
            "warnings": warnings,
            "metadata": metadata
        }
    except Exception as e:
        import logging
        logging.error(f"Internal QA API Error: {str(e)}", exc_info=True)
        log_request_if_enabled(
            question_len=len(request.question),
            status="500",
            top_k=request.top_k,
            warnings=[str(e)]
        )
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred while processing the request."
        )
