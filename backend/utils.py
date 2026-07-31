"""
utils.py
--------
Shared utility functions: logging setup, text cleaning helpers, and
small reusable helpers used across the backend modules.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Optional

from backend.config import get_settings

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance, cached by name.

    Args:
        name: Logical name for the logger (usually __name__ of caller).

    Returns:
        A configured `logging.Logger` instance with a stream handler.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    settings = get_settings()
    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level.upper())

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    _LOGGERS[name] = logger
    return logger


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters from raw PDF text.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text with collapsed whitespace.
    """
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate(text: str, max_chars: int = 300) -> str:
    """Truncate text to a maximum number of characters, appending an ellipsis.

    Args:
        text: Text to truncate.
        max_chars: Maximum number of characters to keep.

    Returns:
        Possibly truncated text.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def safe_str(value: Optional[str], default: str = "") -> str:
    """Return `value` if it is a non-empty string, else `default`."""
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == "":
        return default
    return value
