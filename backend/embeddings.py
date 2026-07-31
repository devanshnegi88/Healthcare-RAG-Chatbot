"""
embeddings.py
-------------
Wraps the BAAI/bge-small-en-v1.5 sentence-embedding model behind a small
reusable interface used by the RAG pipeline for both indexing and querying.

Uses `sentence-transformers` (a lightweight, well supported way to load the
BGE model family) rather than re-implementing pooling logic by hand.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

from backend.config import get_settings
from backend.utils import get_logger

logger = get_logger(__name__)

# BGE models recommend prefixing queries (not documents) with an
# instruction string to improve retrieval quality.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingModel:
    """Thin wrapper around a SentenceTransformer embedding model."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # local import: heavy dep

        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        logger.info("Loading embedding model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of document chunks (no query instruction prefix).

        Args:
            texts: List of raw chunk texts.

        Returns:
            A (n, dim) float32 numpy array of L2-normalized embeddings.
        """
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single user query, using the BGE query instruction prefix.

        Args:
            text: The raw user query.

        Returns:
            A (dim,) float32 numpy array, L2-normalized.
        """
        prefixed = BGE_QUERY_INSTRUCTION + text
        embedding = self._model.encode(
            [prefixed],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return embedding.astype("float32")


@lru_cache()
def get_embedding_model() -> EmbeddingModel:
    """Return a process-wide cached EmbeddingModel instance."""
    return EmbeddingModel()
