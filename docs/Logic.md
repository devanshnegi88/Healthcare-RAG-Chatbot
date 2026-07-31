# Logic

This document explains the reasoning behind each stage of the pipeline in
more detail than the architecture overview.

## 1. Query Flow

```
User message
  → Guardrail check (emergency? diagnosis/prescription request?)
  → [if emergency] return emergency template, skip LLM entirely
  → Retrieve conversation history (memory.py)
  → Embed query → FAISS top-k search (rag.py)
  → Build prompt (prompts.py): system rules + history + retrieved context + question
  → Call Gemini 2.5 Flash (llm.py), streaming tokens back
  → Post-process response with guardrails.sanitize_response()
  → Store turn in memory
  → Return answer + sources + disclaimer
```

Every request is stateless at the HTTP layer — the `session_id` sent by the
client is the only correlator used to fetch the right conversation history,
which keeps the backend horizontally scalable (aside from the in-memory
store, which is documented as a known limitation).

## 2. RAG Pipeline

**Indexing (offline / on startup / on upload):**
1. `load_pdfs()` opens every PDF in `data/docs/` with PyMuPDF and extracts
   text page-by-page, cleaning whitespace via `utils.clean_text`.

2. `split_documents()` performs a sliding-window character split
   (`CHUNK_SIZE` characters, `CHUNK_OVERLAP` overlap) per page, preserving
   the originating filename and page number as metadata on every chunk.

3. `EmbeddingModel.embed_documents()` encodes all chunks in batches using
   `BAAI/bge-small-en-v1.5`, L2-normalizing the output so that inner-product
   search is equivalent to cosine similarity.

4. A FAISS `IndexFlatIP` index is built from the normalized embeddings and
   persisted to disk (`index.faiss` + `metadata.pkl`), so subsequent
   process restarts don't need to re-embed everything.

**Retrieval (per query):**
1. `EmbeddingModel.embed_query()` encodes the user's question, prefixing it
   with the BGE-recommended query instruction string to align it with how
   documents were embedded.

2. `VectorStore.search()` runs FAISS's nearest-neighbor search, returning
   the top-k (`TOP_K`, default 5) chunks with similarity scores.

3. Chunks are wrapped as `RetrievedChunk` objects carrying `source`and `page`
   which the prompt builder and the API's citation payload
   both consume.

**Why this design:** a simple `IndexFlatIP` + character-based splitter is
easy to reason about, requires no external services, and is fast enough for
a knowledge base of the size expected in this assignment. It can be swapped
for `IndexIVFFlat`/`IndexHNSWFlat` or a recursive/semantic splitter without
touching any other module, since `VectorStore` is the sole integration
point used by `api.py`.

## 3. Prompt Engineering

The **system prompt** (`prompts.SYSTEM_PROMPT`) is passed to Gemini as a
`system_instruction`, which keeps it authoritative across the entire
conversation regardless of what appears in the user turn. It encodes five
non-negotiable rules:

1. Never diagnose.
2. Never prescribe medication or dosages.
3. Never replace professional care — always point back to a clinician.
4. Never hallucinate — ground answers in retrieved context, and say
   explicitly when information isn't available.
5. Defer to emergency guidance for emergency-flagged conversations.

The **user-turn prompt** (`prompts.build_user_prompt`) is assembled fresh
for every request from three blocks:

- `CONVERSATION HISTORY` — the rolling memory window, so the model has
  short-term context (e.g., a follow-up question like "what about
  children?").
- `RETRIEVED CONTEXT FROM KNOWLEDGE BASE` — the top-k chunks, each labeled
  `[Source N: filename (page X)]` so the model can cite them naturally in
  its answer.
- `CURRENT USER QUESTION` — the raw question, plus an explicit instruction
  reminding the model to cite sources and fall back to general knowledge
  (with a disclosure) when context is insufficient.

Keeping the system prompt and user prompt in separate, clearly delimited
blocks (rather than one large blob) reduces the chance of the model
conflating retrieved document text with instructions, and makes it easy to
unit test `build_user_prompt` independently of the LLM call.

## 4. Conversation Memory

`ConversationMemory` keeps a `deque` (bounded to `MAX_MEMORY_TURNS * 2`
entries — user + assistant per turn) per `session_id`, guarded by a lock for
thread safety under FastAPI's threaded request handling. It intentionally
does **not** persist to disk: session ids are ephemeral (generated per
browser session by the Streamlit frontend), and the "New Chat" button calls
`ConversationMemory.clear()` explicitly.

## 5. Guardrails

Guardrails run in two places:

- **Pre-LLM (`check_message`)**: regex pattern matching against a curated
  list of emergency phrases (chest pain, stroke symptoms, severe bleeding,
  poisoning, etc.). A match causes the pipeline to skip the LLM entirely
  and return a fixed `EMERGENCY_RESPONSE_TEMPLATE`, guaranteeing consistent,
  fast, and safe behavior for the highest-stakes case rather than trusting
  the LLM to always catch it.
- **Post-LLM (`sanitize_response`)**: a lightweight defense-in-depth check
  that scans the model's own output for risky diagnostic/prescriptive
  phrasing (e.g., "you have", "I prescribe") and appends a clarifying note
  if found, without attempting to silently rewrite the model's answer.

This two-layer approach follows the principle that safety-critical
detection (emergencies) should not depend solely on the LLM's judgment,
while still allowing the LLM's own system-prompt-driven behavior to handle
the broad space of "don't diagnose / don't prescribe" phrasing.

## 6. LLM Workflow

`GeminiClient` wraps `google-generativeai`'s `GenerativeModel`, configured
once with the system prompt at construction time. Two methods are exposed:

- `generate()` — used by the non-streaming `/chat` endpoint (useful for
  programmatic/API consumers).
- `stream()` — used by `/chat/stream`; yields text deltas as they arrive
  from Gemini, which the FastAPI endpoint re-yields as a
  `StreamingResponse`, and which Streamlit consumes chunk-by-chunk to
  render a live "typing" effect.

Both methods apply the same `temperature` and `max_output_tokens` from
configuration, so behavior is consistent between streaming and
non-streaming calls.
