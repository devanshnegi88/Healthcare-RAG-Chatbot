"""
memory.py
---------
Simple in-process conversation memory manager.

Stores a rolling window of recent turns per session (chat) id, so the
prompt builder can include short-term context without unbounded growth.
For a production deployment this would be backed by Redis or a database;
here it is an in-memory store suitable for a single backend process, which
matches the scope of this assignment.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Deque, Dict, List

from backend.config import get_settings
from backend.utils import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    """Thread-safe rolling conversation memory keyed by session id."""

    def __init__(self, max_turns: int | None = None) -> None:
        settings = get_settings()
        self._max_turns = max_turns or settings.max_memory_turns
        self._store: Dict[str, Deque[dict]] = defaultdict(
            lambda: deque(maxlen=self._max_turns * 2)
        )
        self._lock = threading.Lock()

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Add a single turn (user or assistant message) to memory.

        Args:
            session_id: Unique identifier for the chat session.
            role: Either "user" or "assistant".
            content: The message text.
        """
        with self._lock:
            self._store[session_id].append({"role": role, "content": content})
        logger.debug("Memory updated for session=%s role=%s", session_id, role)

    def get_history(self, session_id: str) -> List[dict]:
        """Return the current conversation history for a session.

        Args:
            session_id: Unique identifier for the chat session.

        Returns:
            A list of {"role", "content"} dicts, oldest first.
        """
        with self._lock:
            return list(self._store.get(session_id, []))

    def clear(self, session_id: str) -> None:
        """Clear stored history for a session (used by 'New Chat')."""
        with self._lock:
            if session_id in self._store:
                self._store[session_id].clear()
        logger.info("Cleared memory for session=%s", session_id)


# Module-level singleton so the API layer and RAG layer share one memory store.
conversation_memory = ConversationMemory()
