# Architecture

## Overview

The Healthcare AI Chatbot is split into two independent processes that
communicate over HTTP:

1. **Frontend** — a Streamlit application (`frontend/streamlit_app.py`)
   responsible only for presentation: rendering chat bubbles, streaming
   text as it arrives, showing source citations, and handling PDF uploads.
2. **Backend** — a FastAPI application (`backend/api.py`) responsible for
   all business logic: safety checks, memory, retrieval, prompt
   construction, and calling the Gemini LLM.

This separation means the backend can be reused by any client (a mobile
app, a different frontend, or another service) without changes.

## Component Diagram

```
┌────────────┐      HTTP (JSON / stream)      ┌──────────────────┐
│  Streamlit │ ─────────────────────────────▶ │     FastAPI       │
│  Frontend  │ ◀───────────────────────────── │     (api.py)       │
└────────────┘                                 └─────────┬────────┘
                                                          │
                     ┌────────────────────────────────────┼─────────────────────────┐
                     ▼                                    ▼                         ▼
              ┌─────────────┐                    ┌────────────────┐        ┌────────────────┐
              │ Guardrails   │                    │ Conversation    │        │  Vector Store   │
              │ (guardrails  │                    │ Memory          │        │  (rag.py +      │
              │  .py)        │                    │ (memory.py)     │        │  embeddings.py) │
              └─────────────┘                    └────────────────┘        └────────┬────────┘
                                                                                       │
                                                                                       ▼
                                                                              ┌────────────────┐
                                                                              │  FAISS Index    │
                                                                              │ (data/vectorstore)│
                                                                              └────────────────┘
                     ┌───────────────────────────────────────────────────────────────┘
                     ▼
              ┌─────────────┐        ┌──────────────┐
              │  Prompt      │──────▶ │  Gemini 2.5   │
              │  Builder     │        │  Flash        │
              │ (prompts.py) │        │ (llm.py)      │
              └─────────────┘        └──────────────┘
```

## Backend Modules

| Module | Responsibility |
|---|---|
| `config.py` | Loads and validates all environment-driven settings via Pydantic. |
| `utils.py` | Shared logging setup and text-cleaning helpers. |
| `guardrails.py` | Regex-based emergency and diagnosis/prescription-request detection; post-response safety reinforcement. |
| `memory.py` | Thread-safe rolling window of recent conversation turns, keyed by session id. |
| `embeddings.py` | Loads `BAAI/bge-small-en-v1.5` via `sentence-transformers` and exposes `embed_documents` / `embed_query`. |
| `rag.py` | Loads PDFs with PyMuPDF, splits them into overlapping chunks, builds/persists a FAISS `IndexFlatIP` index, and performs top-k semantic search. |
| `prompts.py` | Defines the system prompt (safety contract) and assembles the final user-turn prompt from retrieved context + memory + question. |
| `llm.py` | Wraps `google-generativeai`'s Gemini client for both full and streaming generation. |
| `api.py` | FastAPI routes wiring all of the above together, plus PDF upload/re-indexing endpoints. |

## Data Flow (Happy Path)

1. User sends a message from the Streamlit UI.
2. Frontend POSTs to `/chat/stream` with `{session_id, message}`.
3. Backend runs `GuardrailEngine.check_message()`.
   - If an emergency pattern is detected, the pipeline short-circuits and
     returns the emergency guidance template immediately (no LLM call).
4. Otherwise, the backend fetches conversation history from
   `ConversationMemory`.
5. The backend embeds the query and searches the FAISS index for the top-k
   most relevant chunks (`VectorStore.search`).
6. `prompts.build_user_prompt()` assembles a single prompt containing the
   system-level rules (via Gemini's `system_instruction`), the conversation
   history, the retrieved context (with source/page labels), and the
   question.
7. `GeminiClient.stream()` streams the response back token-by-token; the
   Streamlit UI renders it progressively.
8. After the stream completes, the backend stores the full answer in
   memory and caches source citations for the frontend to fetch via
   `/session/{id}/last_sources` (avoiding a second LLM call).
9. The frontend displays the answer, expandable source citations, and the
   medical disclaimer.

## Persistence

- **Vector store**: FAISS index + chunk metadata are persisted to
  `data/vectorstore/index.faiss` and `metadata.pkl`. On startup, the
  backend loads this index if present; otherwise it builds one from
  `data/docs/*.pdf`.
- **Conversation memory**: kept in-process (in-memory dict), scoped to a
  session id generated per browser session. This is sufficient for a
  single-instance deployment; a production system would back this with
  Redis or a database (see README's Future Improvements).
