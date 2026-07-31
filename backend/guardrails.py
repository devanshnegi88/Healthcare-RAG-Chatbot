"""
guardrails.py
-------------
Safety layer for the Healthcare AI Chatbot.

Responsibilities:
- Detect potential medical emergencies from user input using keyword and
  pattern matching, and short-circuit the pipeline with emergency guidance.
- Detect attempts to request diagnosis or prescriptions and soften the
  question routing (the LLM system prompt also enforces this, this is a
  defense-in-depth layer).
- Provide a single `GuardrailEngine` class used by the API layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from backend.utils import get_logger

logger = get_logger(__name__)


# Keyword/pattern groups that strongly indicate a medical emergency.
EMERGENCY_PATTERNS: List[str] = [
    r"\bchest pain\b",
    r"\bheart attack\b",
    r"\bcan'?t breathe\b",
    r"\bcannot breathe\b",
    r"\bdifficulty breathing\b",
    r"\bshortness of breath\b",
    r"\bstroke\b",
    r"\bface drooping\b",
    r"\bslurred speech\b",
    r"\bnumbness on one side\b",
    r"\bsevere bleeding\b",
    r"\bheavy bleeding\b",
    r"\bwon'?t stop bleeding\b",
    r"\bpoison(ing)?\b",
    r"\boverdose\b",
    r"\bunconscious\b",
    r"\bnot breathing\b",
    r"\bsevere allergic reaction\b",
    r"\banaphylaxis\b",
    r"\bsuicidal\b",
    r"\bwant to die\b",
    r"\bkill myself\b",
    r"\bseizure\b",
    r"\bchoking\b",
    r"\bsevere burn\b",
    r"\bcoughing (up )?blood\b",
    r"\bblue lips\b",
    r"\bcrushing pain\b",
]

_EMERGENCY_REGEX = re.compile("|".join(EMERGENCY_PATTERNS), re.IGNORECASE)

# Patterns suggesting the user wants a direct diagnosis or prescription.
DIAGNOSIS_REQUEST_PATTERNS = [
    r"\bwhat disease do i have\b",
    r"\bdo i have (cancer|diabetes|covid|hiv)\b",
    r"\bdiagnose me\b",
    r"\bwhat'?s wrong with me\b",
]
PRESCRIPTION_REQUEST_PATTERNS = [
    r"\bwhat (medicine|medication|drug|dose|dosage) should i take\b",
    r"\bprescribe\b",
    r"\bhow many (mg|milligrams|pills)\b",
]

_DIAGNOSIS_REGEX = re.compile("|".join(DIAGNOSIS_REQUEST_PATTERNS), re.IGNORECASE)
_PRESCRIPTION_REGEX = re.compile("|".join(PRESCRIPTION_REQUEST_PATTERNS), re.IGNORECASE)


@dataclass
class GuardrailResult:
    """Outcome of running guardrail checks on a user message."""

    is_emergency: bool
    is_diagnosis_request: bool
    is_prescription_request: bool
    matched_terms: List[str]


class GuardrailEngine:
    """Encapsulates all safety checks applied before/after LLM calls."""

    def check_message(self, message: str) -> GuardrailResult:
        """Run all guardrail checks against a raw user message.

        Args:
            message: The user's raw input text.

        Returns:
            A `GuardrailResult` describing what was detected.
        """
        matched_terms = [m.group(0) for m in _EMERGENCY_REGEX.finditer(message)]
        is_emergency = len(matched_terms) > 0
        is_diagnosis_request = bool(_DIAGNOSIS_REGEX.search(message))
        is_prescription_request = bool(_PRESCRIPTION_REGEX.search(message))

        if is_emergency:
            logger.warning("Emergency pattern detected in user message: %s", matched_terms)

        return GuardrailResult(
            is_emergency=is_emergency,
            is_diagnosis_request=is_diagnosis_request,
            is_prescription_request=is_prescription_request,
            matched_terms=matched_terms,
        )

    def sanitize_response(self, response: str) -> str:
        """Apply light post-processing safety checks to the LLM's response.

        This is a defense-in-depth check: if the model still slips in
        diagnostic or prescriptive language despite the system prompt, we
        append a reinforcement note. We do not attempt to rewrite the model's
        text automatically to avoid corrupting valid educational content.

        Args:
            response: The raw text returned by the LLM.

        Returns:
            The response, possibly with an appended safety reminder.
        """
        risky_phrases = [
            "you have",
            "you are diagnosed with",
            "i prescribe",
            "take this medication",
            "your dosage should be",
        ]
        lowered = response.lower()
        if any(phrase in lowered for phrase in risky_phrases):
            logger.info("Post-response guardrail reinforcement triggered.")
            response += (
                "\n\n_Note: The above is general educational information, "
                "not a diagnosis or prescription. Please confirm with a "
                "licensed healthcare professional._"
            )
        return response
