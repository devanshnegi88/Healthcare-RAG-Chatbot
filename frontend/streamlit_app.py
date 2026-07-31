"""
streamlit_app.py
----------------
Professional Streamlit frontend for the Healthcare AI Chatbot.

Talks to the FastAPI backend over HTTP, streaming the assistant's response
token-by-token, showing source citations, a medical disclaimer, suggested
questions, and an optional PDF upload panel for extending the knowledge base.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

SUGGESTED_QUESTIONS = [
    "What are common symptoms of the flu?",
    "How can I improve my sleep hygiene?",
    "What foods are good for heart health?",
    "What basic first aid steps should I take for a minor burn?",
    "How can I reduce my risk of type 2 diabetes?",
]

st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources"}


def call_new_session() -> None:
    """Reset the chat: get a fresh session id and clear local history."""
    try:
        requests.post(
            f"{BACKEND_URL}/session/clear",
            json={"session_id": st.session_state.session_id},
            timeout=10,
        )
    except requests.RequestException:
        pass
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []


def stream_chat_response(message: str):
    """Stream a response from the backend, yielding text chunks.

    Args:
        message: The user's message text.

    Yields:
        Successive text chunks from the backend streaming endpoint.
    """
    with requests.post(
        f"{BACKEND_URL}/chat/stream",
        json={"session_id": st.session_state.session_id, "message": message},
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk


def fetch_last_turn_metadata() -> dict:
    """Fetch citation metadata for the most recent streamed turn.

    This avoids a second LLM call: the backend caches sources/emergency
    flag computed during `/chat/stream` and exposes them here.
    """
    response = requests.get(
        f"{BACKEND_URL}/session/{st.session_state.session_id}/last_sources",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🩺 Healthcare AI Assistant")
    st.caption("Educational healthcare information, powered by RAG + Gemini 2.5 Flash")

    st.divider()
    st.markdown("### About")
    st.write(
        "This assistant answers general questions about symptoms, diseases, "
        "nutrition, healthy lifestyle, preventive care, and first aid. "
        "It does **not** diagnose conditions or prescribe medication."
    )

    st.divider()
    if st.button("🆕 New Chat", use_container_width=True):
        call_new_session()
        st.rerun()

    st.divider()
    st.markdown("### 📄 Upload Knowledge Base PDFs")
    uploaded_file = st.file_uploader("Upload a PDF to extend the knowledge base", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Index this PDF", use_container_width=True):
            with st.spinner("Uploading and rebuilding the index..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    resp = requests.post(f"{BACKEND_URL}/documents/upload", files=files, timeout=300)
                    resp.raise_for_status()
                    data = resp.json()
                    st.success(
                        f"Indexed '{data['filename']}'. Total chunks: {data['total_chunks']}"
                    )
                except requests.RequestException as exc:
                    st.error(f"Failed to index PDF: {exc}")

    st.divider()
    st.markdown(
        "⚠️ **Disclaimer:** For general education only. Not a substitute "
        "for professional medical advice, diagnosis, or treatment."
    )

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("Healthcare AI Assistant")
st.caption(
    "Ask about symptoms, nutrition, healthy habits, preventive care, or first aid."
)

# Suggested questions (only show before the first message)
if not st.session_state.messages:
    st.markdown("#### 💡 Try asking:")
    cols = st.columns(len(SUGGESTED_QUESTIONS))
    clicked_question = None
    for col, question in zip(cols, SUGGESTED_QUESTIONS):
        with col:
            if st.button(question, use_container_width=True):
                clicked_question = question
else:
    clicked_question = None

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🩺" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg.get("is_emergency"):
            st.error("This was flagged as a potential emergency.")
        # CHANGE: render the backend's pre-formatted "📚 References" block
        # (grouped by document, deduplicated pages, no relevance scores)
        # instead of a raw per-chunk (source, page, score) list.
        if msg.get("references_text"):
            with st.expander("📚 References"):
                st.markdown(msg["references_text"])
        if msg["role"] == "assistant":
            st.caption(msg.get("disclaimer", ""))

# Chat input (also accepts a suggested question click)
user_input = st.chat_input("Type your health question here...")
final_input = clicked_question or user_input

if final_input:
    st.session_state.messages.append({"role": "user", "content": final_input})
    with st.chat_message("user"):
        st.markdown(final_input)

    with st.chat_message("assistant", avatar="🩺"):
        placeholder = st.empty()
        full_text = ""
        try:
            with st.spinner("Thinking..."):
                for chunk in stream_chat_response(final_input):
                    full_text += chunk
                    placeholder.markdown(full_text + "▌")
            placeholder.markdown(full_text)
        except requests.RequestException as exc:
            full_text = f"⚠️ Sorry, I couldn't reach the backend service: {exc}"
            placeholder.markdown(full_text)

        # Fetch structured metadata (citations / emergency flag / disclaimer)
        # CHANGE: "sources" (flat, scored, per-chunk) replaced with
        # "references_text" — the backend's pre-formatted, grouped,
        # deduplicated "📚 References" block. No scores are ever received
        # or displayed here.
        references_text = ""
        is_emergency = False
        disclaimer = (
            "⚠️ **Medical Disclaimer:** This information is for general "
            "educational purposes only and is not medical advice."
        )
        try:
            meta = fetch_last_turn_metadata()
            references_text = meta.get("references_text", "")
            is_emergency = meta.get("is_emergency", False)
            disclaimer = meta.get("disclaimer", disclaimer)
        except requests.RequestException:
            pass

        if is_emergency:
            st.error("This was flagged as a potential emergency.")
        if references_text:
            references_text = references_text.replace("📚 References", "").strip()

        with st.expander("📚 References"):
            st.markdown(
                references_text.replace("\n", "  \n")
            )
            
        st.caption(disclaimer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_text,
            "references_text": references_text,
            "is_emergency": is_emergency,
            "disclaimer": disclaimer,
        }
    )
