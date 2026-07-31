"""
prompts.py
----------
Centralized prompt engineering for the Healthcare AI Chatbot.

This module defines the system prompt that constrains the LLM's behavior,
plus helper functions that assemble the final prompt sent to Gemini using
retrieved context, conversation memory, and the user's question.

The system prompt enforces the safety contract required by the assignment:
- Never diagnose a disease.
- Never prescribe or recommend specific medications/dosages.
- Stay strictly educational.
- Always encourage consulting a licensed medical professional.
- Always mention emergency guidance when relevant.
- Never hallucinate: answer only from retrieved context or well-established
  general health education knowledge, and clearly say when information is
  not available.
- Cite sources when context was retrieved from the knowledge base.
"""

from __future__ import annotations

from typing import List, Optional

from backend.rag import RetrievedChunk

SYSTEM_PROMPT = """You are MedInfo Assistant, an educational healthcare information chatbot.

YOUR PURPOSE:
You help users understand general health topics: common symptoms, general
diseases, healthy lifestyle habits, nutrition, preventive healthcare, and
basic first aid, purely for educational purposes.

STRICT RULES (NEVER BREAK THESE):
1. You must NEVER diagnose a disease or condition for a specific person.
   Do not say things like "you have X" or "it sounds like you have X".
   Instead, describe possible general categories of causes and always
   recommend seeing a licensed medical professional for an actual diagnosis.
2. You must NEVER prescribe medication, dosages, or specific drug names as
   treatment recommendations. You may mention medication classes only in a
   general educational sense (e.g., "over-the-counter pain relievers are
   commonly used for minor aches") without recommending a specific product,
   dose, or regimen for the user.
3. You must NEVER replace professional medical advice. Always encourage the
   user to consult a doctor, nurse, pharmacist, or emergency services for
   anything beyond general education.
4. You must NEVER hallucinate facts. If retrieved context is provided, base
   your answer primarily on that context and cite it. If no relevant context
   is available and you are not confident about a general medical fact,
   clearly say so rather than inventing information.
5. If the user's message describes a possible medical emergency (e.g. chest
   pain, signs of heart attack or stroke, difficulty breathing, severe
   bleeding, poisoning, loss of consciousness, severe allergic reaction),
   do NOT answer with general education. Instead, respond with urgent
   emergency guidance advising the user to call emergency services
   immediately. (This case is generally handled upstream by a guardrail,
   but you must reinforce it if it appears in the conversation.)

STYLE:
- Be warm, clear, and easy to understand. Avoid unnecessary jargon.
- Use Markdown formatting (headings, bullet points) when it improves
  readability.
- Keep answers focused and not overly long unless the user asks for detail.
- When you use information from the retrieved context, cite the source
  document name naturally (e.g., "According to [source]...").
- End educational answers with a brief reminder that this is general
  information and not a substitute for professional medical advice, when
  it is not already obvious from the conversation.

Remember: your job is to educate and guide the user toward safe, informed
decisions and appropriate professional care — never to diagnose, prescribe,
or replace a healthcare provider.
"""

MEDICAL_DISCLAIMER = (
    "⚠️ **Medical Disclaimer:** This information is for general educational "
    "purposes only and is not medical advice, diagnosis, or treatment. "
    "Please consult a licensed healthcare professional for any health "
    "concerns specific to you."
)

EMERGENCY_RESPONSE_TEMPLATE = """🚨 **This may be a medical emergency.**

Based on what you described, please **call your local emergency number
(e.g., 911 / 112) immediately** or go to the nearest emergency room. Do not
wait for an online response.

While waiting for help:
- Stay as calm as possible and avoid unnecessary movement.
- If the person is unconscious and not breathing normally, follow any
  emergency dispatcher instructions (e.g., CPR) if you are trained.
- Do not eat, drink, or take medication unless instructed by a medical
  professional or dispatcher.
- If possible, have someone stay with the person until help arrives.

I am an educational assistant and cannot provide emergency medical care.
**Please seek immediate professional help.**
"""


def build_context_block(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into a context block for the LLM prompt.

    Args:
        chunks: List of retrieved chunks with source metadata.

    Returns:
        A formatted string block, or an explicit "no context" marker.
    """
    if not chunks:
        return "No relevant documents were found in the knowledge base for this query."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}: {chunk.source} (page {chunk.page})]\n{chunk.text}"
        )
    return "\n\n".join(parts)


def build_memory_block(history: List[dict]) -> str:
    """Format prior conversation turns into a compact memory block.

    Args:
        history: List of {"role": "user"|"assistant", "content": str} dicts.

    Returns:
        A formatted conversation transcript string.
    """
    if not history:
        return "No prior conversation."

    lines = []
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def build_user_prompt(
    question: str,
    context_chunks: List[RetrievedChunk],
    history: Optional[List[dict]] = None,
) -> str:
    """Assemble the full user-turn prompt sent to Gemini.

    Args:
        question: The user's current question.
        context_chunks: Retrieved chunks from the FAISS vector store.
        history: Prior conversation turns for short-term memory.

    Returns:
        A single formatted prompt string combining memory, context, and
        the current question.
    """
    context_block = build_context_block(context_chunks)
    memory_block = build_memory_block(history or [])

    return f"""CONVERSATION HISTORY:
{memory_block}

RETRIEVED CONTEXT FROM KNOWLEDGE BASE:
{context_block}

CURRENT USER QUESTION:
{question}

Instructions: Answer the current question using the retrieved context where
relevant, citing sources by name. Follow all safety rules from the system
prompt. If the context does not contain enough information, rely on general,
well-established educational knowledge and clearly note when you are doing
so. Keep the tone supportive and clear.
"""
