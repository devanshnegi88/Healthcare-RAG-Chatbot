"""
config.py
---------
Centralized configuration for the Healthcare AI Chatbot backend.

All environment-dependent values (API keys, model names, paths, and
tunable RAG parameters) are loaded here from environment variables
via python-dotenv so the rest of the codebase never touches os.environ
directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field,ConfigDict

# Load .env file from project root (if present) before reading env vars.
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


class Settings(BaseModel):
    """Application-wide configuration, validated with Pydantic."""

    # --- LLM ---
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = Field(default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    llm_temperature: float = Field(default=float(os.getenv("LLM_TEMPERATURE", "0.3")))
    llm_max_output_tokens: int = Field(default=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024")))

    # --- Embeddings ---
    embedding_model_name: str = Field(
        default=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    )

    # --- Paths ---
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    docs_dir: Path = BASE_DIR / "data" / "docs"
    vectorstore_dir: Path = BASE_DIR / "data" / "vectorstore"

    # --- RAG tuning ---
    chunk_size: int = Field(default=int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = Field(default=int(os.getenv("CHUNK_OVERLAP", "120")))
    top_k: int = Field(default=int(os.getenv("TOP_K", "5")))

    # --- Memory ---
    max_memory_turns: int = Field(default=int(os.getenv("MAX_MEMORY_TURNS", "6")))

    # --- API ---
    api_host: str = Field(default=os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = Field(default=int(os.getenv("API_PORT", "8000")))
    frontend_port: int = Field(default=int(os.getenv("FRONTEND_PORT", "8501")))

    # --- Logging ---
    log_level: str = Field(default=os.getenv("LOG_LEVEL", "INFO"))

model_config = ConfigDict(
    arbitrary_types_allowed=True
)


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    settings = Settings()
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    return settings
