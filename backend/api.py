"""
api.py
------
FastAPI application exposing the Healthcare AI Chatbot backend.

Endpoints:
- GET  /health          -> liveness check
- POST /chat            -> full (non-streaming) chat response
- POST /chat/stream     -> Server-Sent-Events style streaming chat response
- POST /session/clear   -> clear conversation memory for a session
- POST /documents/upload -> upload a new PDF and rebuild the vector index
- GET  /documents        -> list indexed documents

The pipeline for every chat request follows the architecture:
User -> Guardrails -> Conversation Memory -> Retriever (FAISS) ->
Prompt Builder -> Gemini -> Response

CHANGE: citation handling now uses `rag.group_citations()` /
`rag.format_citations()` so every response exposes a deduplicated,
per-document citation list (source + merged page numbers) instead of a
flat, possibly-duplicated list of (source, page, score) tuples. Relevance
scores are never included in any API response — see `rag.py` for where
they're logged at DEBUG level instead.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.guardrails import GuardrailEngine
from backend.llm import get_llm_client
from backend.memory import conversation_memory
from backend.prompts import EMERGENCY_RESPONSE_TEMPLATE, MEDICAL_DISCLAIMER, build_user_prompt
from backend.rag import RetrievedChunk, format_citations, get_vector_store, group_citations
from backend.utils import get_logger

logger = get_logger(__name__)
settings = get_settings()
guardrails = GuardrailEngine()

# Tracks the most recent retrieval metadata per session so the streaming
# endpoint's plain-text stream can be paired with structured citations
# afterward, without re-invoking the LLM a second time.
_last_turn_metadata: dict = {}

app = FastAPI(
    title="Healthcare AI Chatbot API",
    description="Educational healthcare Q&A backend with RAG, guardrails, and memory.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Payload for a chat request."""

    session_id: str = Field(..., description="Unique chat/session identifier.")
    message: str = Field(..., min_length=1, description="The user's message.")


# CHANGE: SourceCitation (source/page/score, one entry per chunk, could
# contain duplicates) replaced with CitationGroup (source + all its merged,
# deduplicated pages, one entry per document). No score field — relevance
# scores are never exposed via the API.
class CitationGroup(BaseModel):
    """A single document's citation entry: source filename + merged pages."""

    source: str
    pages: List[int]


class ChatResponse(BaseModel):
    """Full (non-streaming) chat response payload."""

    answer: str
    is_emergency: bool
    citations: List[CitationGroup]
    references_text: str  # Pre-formatted "📚 References" block, ready to render
    disclaimer: str


class ClearSessionRequest(BaseModel):
    session_id: str


def _build_citation_payload(chunks: List[RetrievedChunk]) -> dict:
    """Build the citations/references_text pair shared by all chat endpoints.

    Centralizing this avoids duplicating the grouping/formatting logic
    between `/chat` and `/chat/stream`.

    Args:
        chunks: Retrieved chunks for the turn (may contain duplicate
            source/page pairs across chunks).

    Returns:
        A dict with "citations" (list of {source, pages}) and
        "references_text" (the pre-formatted "📚 References" block).
    """
    groups = group_citations(chunks)
    return {
        "citations": [{"source": g.source, "pages": sorted(set(g.pages))} for g in groups],
        "references_text": format_citations(chunks),
    }


@app.on_event("startup")
def on_startup() -> None:
    """Warm up the vector store on startup so first request isn't slow."""
    logger.info("Starting Healthcare AI Chatbot API...")
    try:
        get_vector_store()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to initialize vector store on startup.")


@app.get("/health")
def health() -> dict:
    """Simple liveness/readiness check."""
    store = get_vector_store()
    return {
        "status": "ok",
        "vector_store_ready": store.is_loaded,
        "indexed_chunks": store.size,
    }


def _handle_emergency(session_id: str, message: str) -> ChatResponse:
    conversation_memory.add_turn(session_id, "user", message)
    conversation_memory.add_turn(session_id, "assistant", EMERGENCY_RESPONSE_TEMPLATE)
    return ChatResponse(
        answer=EMERGENCY_RESPONSE_TEMPLATE,
        is_emergency=True,
        citations=[],
        references_text="",
        disclaimer=MEDICAL_DISCLAIMER,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Handle a full (non-streaming) chat turn.

    Pipeline: Guardrails -> Memory -> Retriever -> Prompt Builder -> LLM.
    """
    try:
        guard_result = guardrails.check_message(request.message)

        if guard_result.is_emergency:
            return _handle_emergency(request.session_id, request.message)

        history = conversation_memory.get_history(request.session_id)
        store = get_vector_store()
        chunks = store.search(request.message)

        prompt = build_user_prompt(request.message, chunks, history)
        llm = get_llm_client()
        raw_answer = llm.generate(prompt)
        answer = guardrails.sanitize_response(raw_answer)

        conversation_memory.add_turn(request.session_id, "user", request.message)
        conversation_memory.add_turn(request.session_id, "assistant", answer)

        citation_payload = _build_citation_payload(chunks)
        return ChatResponse(
            answer=answer,
            is_emergency=False,
            citations=citation_payload["citations"],
            references_text=citation_payload["references_text"],
            disclaimer=MEDICAL_DISCLAIMER,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error handling /chat request")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Handle a streaming chat turn, yielding plain-text chunks as they arrive."""
    guard_result = guardrails.check_message(request.message)

    if guard_result.is_emergency:
        response_obj = _handle_emergency(request.session_id, request.message)
        _last_turn_metadata[request.session_id] = {
            "is_emergency": True,
            "citations": [],
            "references_text": "",
            "disclaimer": MEDICAL_DISCLAIMER,
        }

        def emergency_gen():
            yield response_obj.answer

        return StreamingResponse(emergency_gen(), media_type="text/plain")

    history = conversation_memory.get_history(request.session_id)
    store = get_vector_store()
    chunks = store.search(request.message)
    prompt = build_user_prompt(request.message, chunks, history)
    llm = get_llm_client()

    conversation_memory.add_turn(request.session_id, "user", request.message)

    citation_payload = _build_citation_payload(chunks)
    _last_turn_metadata[request.session_id] = {
        "is_emergency": False,
        "citations": citation_payload["citations"],
        "references_text": citation_payload["references_text"],
        "disclaimer": MEDICAL_DISCLAIMER,
    }

    def token_generator():
        collected = []
        try:
            for piece in llm.stream(prompt):
                collected.append(piece)
                yield piece
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error during streaming generation")
            error_msg = f"\n\n[Error generating response: {exc}]"
            collected.append(error_msg)
            yield error_msg
        finally:
            full_answer = guardrails.sanitize_response("".join(collected))
            conversation_memory.add_turn(request.session_id, "assistant", full_answer)

    return StreamingResponse(token_generator(), media_type="text/plain")


@app.get("/session/{session_id}/last_sources")
def last_sources(session_id: str) -> dict:
    """Return citation metadata for the most recent streamed turn in a session.

    Used by the Streamlit frontend after consuming `/chat/stream`, so the UI
    can display citations and the emergency flag without a second LLM call.
    """
    return _last_turn_metadata.get(
        session_id,
        {"is_emergency": False, "citations": [], "references_text": "", "disclaimer": MEDICAL_DISCLAIMER},
    )


@app.post("/session/clear")
def clear_session(request: ClearSessionRequest) -> dict:
    """Clear conversation memory for a given session (used by 'New Chat')."""
    conversation_memory.clear(request.session_id)
    return {"status": "cleared", "session_id": request.session_id}


@app.get("/session/new")
def new_session() -> dict:
    """Generate a new unique session id for the frontend to use."""
    return {"session_id": str(uuid.uuid4())}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Upload a new PDF into the knowledge base and rebuild the FAISS index.

    Args:
        file: The uploaded PDF file.

    Returns:
        A status dict including the new number of indexed chunks.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    dest_path: Path = settings.docs_dir / file.filename
    try:
        with open(dest_path, "wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}") from exc
    finally:
        file.file.close()

    logger.info("Saved uploaded PDF to %s, rebuilding index...", dest_path)
    store = get_vector_store()
    store.build_index()

    return {
        "status": "indexed",
        "filename": file.filename,
        "total_chunks": store.size,
    }


@app.get("/documents")
def list_documents() -> dict:
    """List currently indexed source PDF filenames."""
    docs = sorted(p.name for p in settings.docs_dir.glob("*.pdf"))
    store = get_vector_store()
    return {"documents": docs, "total_chunks": store.size}
