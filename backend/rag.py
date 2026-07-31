"""
rag.py
------
Retrieval-Augmented Generation pipeline for the Healthcare AI Chatbot.

Uses LangChain end-to-end for the RAG plumbing, with FAISS as the vector
store:

- `langchain_community.document_loaders.PyMuPDFLoader` for PDF loading
  (PyMuPDF-backed, matching the assignment's PyMuPDF requirement).
- `langchain.text_splitter.RecursiveCharacterTextSplitter` for
  sentence/paragraph-aware chunking.
- `langchain_community.vectorstores.FAISS` as the vector store, fed by a
  small custom `Embeddings` adapter around our BGE `sentence-transformers`
  model so LangChain can call it internally.

Responsibilities:
- Load PDFs from the `data/docs` folder.
- Split extracted text into overlapping chunks, tagging each chunk with
  full provenance metadata (source, page, chunk id).
- Embed chunks and store them in a local FAISS index (persisted to disk
  under `data/vectorstore`).
- Retrieve the top-k most relevant chunks for a given user query, carrying
  their metadata through untouched.
- Format retrieved chunks into a clean, deduplicated, per-document citation
  list suitable for display in the UI (no relevance scores exposed).

This module is the sole integration point for the vector database, so
swapping the backing store or splitter only requires changes here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import get_settings
from backend.embeddings import get_embedding_model
from backend.utils import clean_text, get_logger

logger = get_logger(__name__)

INDEX_DIRNAME = "faiss_index"


@dataclass
class RetrievedChunk:
    """A chunk retrieved for a query, including its full provenance metadata.

    `score` is intentionally kept on this internal dataclass (used only for
    DEBUG logging and prompt-building) but is never surfaced to the API
    response or UI — see `format_citations()` / `to_citation_groups()`
    below, which strip it out entirely.
    """

    text: str
    source: str
    page: int
    chunk_id: int  # CHANGE: chunk id is now always populated (was -1 before splitting)
    score: float


# --------------------------------------------------------------------------- #
# CHANGE: New dataclass representing a single grouped citation entry, i.e.
# one PDF document with all of its cited page numbers merged and
# deduplicated. This is the shape the API/UI should consume instead of a
# flat, possibly-duplicated list of (source, page, score) tuples.
# --------------------------------------------------------------------------- #
@dataclass
class CitationGroup:
    """A single document's citation entry: source filename + merged pages."""

    source: str
    pages: List[int] = field(default_factory=list)

    def formatted_pages(self) -> str:
        """Return pages as a human-readable, sorted, comma-separated string."""
        unique_sorted_pages = sorted(set(self.pages))
        return ", ".join(str(p) for p in unique_sorted_pages)


def load_pdfs(docs_dir: Path) -> List[Document]:
    """Load every PDF in `docs_dir` into LangChain `Document` objects.

    Each Document corresponds to one page, with `source` and `page` set in
    its metadata, using LangChain's `PyMuPDFLoader` (PyMuPDF-backed).

    Args:
        docs_dir: Directory containing .pdf files.

    Returns:
        A list of LangChain `Document` objects, one per non-empty page.
    """
    documents: List[Document] = []
    pdf_paths = sorted(docs_dir.glob("*.pdf"))

    if not pdf_paths:
        logger.warning("No PDF files found in %s", docs_dir)
        return documents

    for pdf_path in pdf_paths:
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            pages = loader.load()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load PDF %s: %s", pdf_path, exc)
            continue

        page_count = 0
        for page_doc in pages:
            text = clean_text(page_doc.page_content)
            if not text:
                continue
            page_number = int(page_doc.metadata.get("page", 0)) + 1
            # CHANGE: metadata now explicitly includes "source" and "page"
            # only at this stage; "chunk_id" is added later in
            # split_documents() once the final chunk boundaries are known,
            # since one page can produce multiple chunks.
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": pdf_path.name, "page": page_number},
                )
            )
            page_count += 1

        logger.info("Loaded %s (%d pages with text)", pdf_path.name, page_count)

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """Split page-level LangChain Documents into overlapping chunks.

    Uses LangChain's `RecursiveCharacterTextSplitter`, which tries to break
    on paragraph/sentence/word boundaries before falling back to a hard
    character cut, giving cleaner chunk boundaries than a fixed-width slice.

    CHANGE: every resulting chunk is now stamped with a globally unique
    `chunk_id` in its metadata (in addition to the `source`/`page` metadata
    inherited from its parent page Document). This guarantees full
    provenance metadata — source, page, and chunk id — is attached at the
    moment of chunk creation and therefore flows automatically through
    FAISS indexing and retrieval without any extra plumbing.

    Args:
        documents: Page-granularity Documents from `load_pdfs`.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap in characters between consecutive chunks.

    Returns:
        A list of chunked Documents, each carrying `source`, `page`, and
        `chunk_id` metadata.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # CHANGE: assign a stable, unique chunk_id to every chunk's metadata.
    # This makes chunk_id available end-to-end (index -> retrieval ->
    # RetrievedChunk) for debugging/traceability, per requirement #1.
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    logger.info("Split %d page documents into %d chunks", len(documents), len(chunks))
    return chunks


class VectorStore:
    """LangChain `FAISS`-backed vector store with disk persistence."""

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._embeddings = BGEEmbeddingsAdapter()
        self._store: Optional[FAISS] = None
        self._index_dir: Path = settings.vectorstore_dir / INDEX_DIRNAME

    @property
    def is_loaded(self) -> bool:
        return self._store is not None

    @property
    def size(self) -> int:
        if self._store is None:
            return 0
        return self._store.index.ntotal

    def build_index(self, docs_dir: Optional[Path] = None) -> None:
        """Build a fresh FAISS index from all PDFs in the docs directory.

        Args:
            docs_dir: Optional override for the documents directory.
        """
        docs_dir = docs_dir or self._settings.docs_dir

        page_documents = load_pdfs(docs_dir)
        chunks = split_documents(page_documents)

        if not chunks:
            logger.warning("No chunks produced; index will be empty.")
            self._store = None
            return

        # NOTE: metadata (source/page/chunk_id) set in split_documents() is
        # stored by LangChain's FAISS.from_documents() alongside each
        # vector automatically — no separate metadata plumbing needed.
        store = FAISS.from_documents(chunks, embedding=self._embeddings)
        self._store = store
        self._persist()
        logger.info("Built FAISS index with %d vectors", store.index.ntotal)

    def _persist(self) -> None:
        """Save the FAISS index to disk via LangChain's built-in helper."""
        if self._store is None:
            return
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self._index_dir))
        logger.info("Persisted FAISS index to %s", self._index_dir)

    def load(self) -> bool:
        """Load a previously persisted FAISS index from disk, if it exists.

        Returns:
            True if a valid index was loaded, False otherwise.
        """
        if not (self._index_dir / "index.faiss").exists():
            logger.info("No persisted FAISS index found at %s", self._index_dir)
            return False

        try:
            self._store = FAISS.load_local(
                str(self._index_dir),
                embeddings=self._embeddings,
                # Safe here: the index is written by this same application,
                # never loaded from an untrusted source.
                allow_dangerous_deserialization=True,
            )
            logger.info("Loaded FAISS index with %d vectors", self._store.index.ntotal)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load FAISS index: %s", exc)
            return False

    def ensure_ready(self) -> None:
        """Load the index from disk, or build it if none exists yet."""
        if not self.load():
            logger.info("Building index for the first time...")
            self.build_index()

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Retrieve the top-k most relevant chunks for a query.

        CHANGE: now reads `chunk_id` out of each result's metadata (in
        addition to the existing `source`/`page`) so full provenance
        metadata is preserved end-to-end into `RetrievedChunk`. Relevance
        scores are still attached to `RetrievedChunk` (needed for DEBUG
        logging and internal ranking) but, per requirement #6, are never
        forwarded beyond this layer into the API response or UI.

        Args:
            query: The user's natural-language query.
            top_k: Number of chunks to retrieve (defaults to config value).

        Returns:
            A list of RetrievedChunk sorted by descending relevance score.
        """
        top_k = top_k or self._settings.top_k

        if self._store is None or self._store.index.ntotal == 0:
            logger.warning("Vector store is empty; returning no context.")
            return []

        k = min(top_k, self._store.index.ntotal)
        results_with_scores = self._store.similarity_search_with_relevance_scores(query, k=k)

        results: List[RetrievedChunk] = []
        for doc, score in results_with_scores:
            chunk = RetrievedChunk(
                text=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                page=int(doc.metadata.get("page", 0)),
                chunk_id=int(doc.metadata.get("chunk_id", -1)),
                score=float(score),
            )
            results.append(chunk)
            # CHANGE: relevance scores are logged at DEBUG level only,
            # per requirement #6 — never printed/returned anywhere else.
            logger.debug(
                "Retrieved chunk_id=%s source=%s page=%s score=%.4f",
                chunk.chunk_id,
                chunk.source,
                chunk.page,
                chunk.score,
            )
        return results


class BGEEmbeddingsAdapter(Embeddings):
    """Adapts our `EmbeddingModel` (sentence-transformers/BGE) to LangChain's
    `Embeddings` interface, so LangChain's `FAISS` vectorstore can call it
    for both indexing and querying.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document chunks.

        Args:
            texts: List of raw chunk texts.

        Returns:
            A list of embedding vectors (as plain lists of floats).
        """
        model = get_embedding_model()
        return model.embed_documents(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string.

        Args:
            text: The raw user query.

        Returns:
            An embedding vector (as a plain list of floats).
        """
        model = get_embedding_model()
        return model.embed_query(text).tolist()


# --------------------------------------------------------------------------- #
# CHANGE: New citation formatting layer (requirements #3-#5, #8).
#
# `group_citations()` deduplicates and groups retrieved chunks by source
# document, merging every page number cited from that document into one
# sorted, de-duplicated list. `format_citations()` renders that grouped
# structure into the exact display format requested:
#
#   📚 References
#   • Healthy diet.pdf (Pages 1, 3, 4, 6)
#   • WHO Nutrition.pdf (Pages 2, 5)
#
# Neither function touches similarity scores, retrieval strategy, prompts,
# or business logic — they operate purely on the metadata already carried
# by `RetrievedChunk`.
# --------------------------------------------------------------------------- #
def group_citations(chunks: List[RetrievedChunk]) -> List[CitationGroup]:
    """Group retrieved chunks by source document, merging duplicate pages.

    Args:
        chunks: Retrieved chunks (each carrying source/page metadata).

    Returns:
        A list of `CitationGroup`, one per unique source document, in the
        order each source was first encountered. Pages within a group are
        deduplicated and sorted ascending.
    """
    grouped: Dict[str, CitationGroup] = {}
    for chunk in chunks:
        if chunk.source not in grouped:
            grouped[chunk.source] = CitationGroup(source=chunk.source, pages=[])
        grouped[chunk.source].pages.append(chunk.page)
    return list(grouped.values())


def format_citations(chunks: List[RetrievedChunk]) -> str:
    """Render retrieved chunks into the final, user-facing citation block.

    Produces output of the exact form:

        📚 References
        • Healthy diet.pdf (Pages 1, 3, 4, 6)
        • WHO Nutrition.pdf (Pages 2, 5)

    No relevance/similarity scores are included, no duplicate sources are
    listed, and duplicate pages from the same document are merged into one
    entry (requirements #3-#5).

    Args:
        chunks: Retrieved chunks to cite.

    Returns:
        A formatted citation string, or an empty string if there are no
        chunks to cite (callers should skip rendering the block entirely
        in that case).
    """
    if not chunks:
        return ""

    groups = group_citations(chunks)
    lines = ["📚 References"]
    for group in groups:
        page_word = "Page" if len(set(group.pages)) == 1 else "Pages"
        lines.append(f"• {group.source} ({page_word} {group.formatted_pages()})")
    return "\n".join(lines)


_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return a process-wide singleton VectorStore, ready for querying."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        _vector_store.ensure_ready()
    return _vector_store