"""
app.py
======
Streamlit entry-point for the Hybrid RAG Document Assistant.

Purpose:
    - Provide the full web UI: document upload, chat interface,
      retrieved chunks viewer, score display, and evaluation dashboard.
    - Wire the RAGPipeline to the UI.

Workflow:
    1. User uploads document(s) → pipeline.ingest()
    2. User types a question    → pipeline.query()
    3. Answer streams to screen, sources and chunks shown below

Run:
    streamlit run app.py

Best Practices:
    - Use st.session_state to preserve pipeline and chat history across reruns
    - Cache the pipeline with @st.cache_resource to avoid re-loading models
    - Stream LLM tokens for perceived speed (st.write_stream)
    - Show a loading spinner during ingestion and retrieval
"""

from __future__ import annotations

# Fix Windows OMP duplicate-library warning (PyTorch + numpy on the same process)
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
from pathlib import Path
from typing import List

import streamlit as st

# Initialise logging before any other local import
from utils.logger import setup_logging
setup_logging(level="INFO")

from config.settings import settings
from pipeline import RAGPipeline
from ui.components import (
    render_chunk_card,
    render_evaluation_panel,
    render_latency_metrics,
    render_settings_sidebar,
    render_sources_panel,
)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=settings.app_title,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Global styles
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .stChatMessage { border-radius: 12px; }
    .upload-zone { border: 2px dashed #aaa; border-radius: 10px; padding: 1rem; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Pipeline — cached so models load once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="⚙️ Loading AI models…")
def get_pipeline() -> RAGPipeline:
    """
    Create and cache the RAGPipeline.
    Expensive model loading (SentenceTransformer, CrossEncoder) happens here.
    """
    return RAGPipeline()


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "messages": [],              # chat history [{role, content, meta}, …]
        "indexed_files": [],         # list of ingested filenames
        "last_result": None,         # last RAG result dict
        "pipeline_ready": False,     # True after at least one doc indexed
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(pipeline: RAGPipeline) -> None:
    """Render the sidebar: upload panel + settings + status."""
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#764ba2;'>📁 Document Manager</h2>",
            unsafe_allow_html=True,
        )

        # --- File uploader ---
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or TXT",
            type=["pdf", "docx", "doc", "txt", "md"],
            accept_multiple_files=True,
            help="Max file size: 50 MB per file",
        )

        if uploaded_files:
            new_files = [
                f for f in uploaded_files
                if f.name not in st.session_state.indexed_files
            ]
            if new_files:
                if st.button(
                    f"📥 Index {len(new_files)} new file(s)",
                    type="primary",
                    use_container_width=True,
                ):
                    for uploaded_file in new_files:
                        with st.spinner(f"Indexing '{uploaded_file.name}'…"):
                            try:
                                result = pipeline.ingest(
                                    file_bytes=uploaded_file.getvalue(),
                                    filename=uploaded_file.name,
                                )
                                st.session_state.indexed_files.append(uploaded_file.name)
                                st.session_state.pipeline_ready = True
                                st.success(
                                    f"✅ **{uploaded_file.name}** — "
                                    f"{result['num_chunks']} chunks indexed "
                                    f"in {result['elapsed_ms']}ms"
                                )
                            except Exception as exc:
                                st.error(f"❌ Failed to index '{uploaded_file.name}': {exc}")
            else:
                st.info("All uploaded files are already indexed.")

        # --- Indexed files ---
        if st.session_state.indexed_files:
            st.markdown("---")
            st.markdown("**📚 Indexed Documents**")
            for fname in st.session_state.indexed_files:
                st.markdown(f"✅ {fname}")

        # --- Clear memory ---
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            pipeline.clear_memory()
            st.rerun()

        # --- Status ---
        st.markdown("---")
        st.markdown("**📊 System Status**")
        status = "🟢 Ready" if st.session_state.pipeline_ready else "🔴 No documents indexed"
        st.markdown(status)

        # GPU status panel
        from utils.helpers import get_gpu_info
        gpu = get_gpu_info()
        st.markdown("**🖥️ Hardware**")
        if gpu["available"]:
            used_pct = (gpu["reserved_mb"] / gpu["total_mb"] * 100) if gpu["total_mb"] else 0
            st.markdown(
                f"<small>🟢 {gpu['name']}<br>"
                f"VRAM: {gpu['reserved_mb']:,} / {gpu['total_mb']:,} MB "
                f"({used_pct:.0f}%)</small>",
                unsafe_allow_html=True,
            )
            st.progress(min(used_pct / 100, 1.0))
        else:
            st.markdown("<small>🟡 CPU only (no CUDA)</small>", unsafe_allow_html=True)

        st.caption(f"v{settings.app_version}")


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------

def render_chat(pipeline: RAGPipeline) -> None:
    """Render the main chat interface."""
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show sources for assistant messages that have them
            if msg["role"] == "assistant" and msg.get("sources"):
                st.caption("📚 " + " · ".join(msg["sources"]))

    # Chat input
    if question := st.chat_input(
        placeholder="Ask a question about your documents…",
        disabled=not st.session_state.pipeline_ready,
    ):
        # User message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Assistant response
        with st.chat_message("assistant"):
            if not pipeline.is_ready():
                st.warning("Please upload and index at least one document first.")
                return

            # Stream the answer
            answer_placeholder = st.empty()
            full_answer = ""

            with st.spinner("🔍 Retrieving…"):
                # Use non-streaming path to get full result dict
                result = pipeline.query(question, stream=False)

            answer = result.get("answer", "")
            sources = result.get("sources", [])
            chunks = result.get("chunks", [])
            meta = result.get("retrieval_meta", {})

            # Simulate streaming display for UX
            with answer_placeholder:
                words = answer.split(" ")
                displayed = ""
                for word in words:
                    displayed += word + " "
                    answer_placeholder.markdown(displayed + "▌")
                    time.sleep(0.015)
                answer_placeholder.markdown(answer)

            # Sources
            if sources:
                st.caption("📚 " + " · ".join(sources))

            # Latency metrics
            render_latency_metrics(meta)

            # Save to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "meta": meta,
                "chunks": chunks,
            })
            st.session_state.last_result = result


# ---------------------------------------------------------------------------
# Retrieved Chunks tab
# ---------------------------------------------------------------------------

def render_chunks_tab() -> None:
    """Display the retrieved and re-ranked chunks from the last query."""
    result = st.session_state.get("last_result")
    if result is None:
        st.info("Submit a query first to see retrieved chunks here.")
        return

    chunks = result.get("chunks", [])
    if not chunks:
        st.warning("No chunks retrieved for the last query.")
        return

    st.markdown(f"### 🔎 {len(chunks)} Re-ranked Chunks")
    for i, (doc, score) in enumerate(chunks, 1):
        render_chunk_card(idx=i, doc=doc, score=score, expanded=(i == 1))


# ---------------------------------------------------------------------------
# Evaluation Dashboard tab
# ---------------------------------------------------------------------------

def render_eval_tab() -> None:
    """Display full evaluation metrics for the last query."""
    result = st.session_state.get("last_result")
    if result is None:
        st.info("Submit a query to see evaluation metrics.")
        return

    meta = result.get("retrieval_meta", {})
    chunks = result.get("chunks", [])

    # Latency breakdown
    st.markdown("### ⏱️ Latency Breakdown")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", f"{meta.get('total_ms', 0)} ms")
    col2.metric("Retrieval", f"{meta.get('retrieval_ms', 0)} ms")
    col3.metric("Re-rank", f"{meta.get('rerank_ms', 0)} ms")
    col4.metric("LLM", f"{meta.get('llm_ms', 0)} ms")

    st.markdown("---")

    # GPU memory snapshot
    from utils.helpers import get_gpu_info
    gpu = get_gpu_info()
    if gpu["available"]:
        st.markdown("### 🖥️ GPU Memory Snapshot")
        g1, g2, g3 = st.columns(3)
        g1.metric("Total VRAM", f"{gpu['total_mb']:,} MB")
        g2.metric("In Use", f"{gpu['reserved_mb']:,} MB")
        g3.metric("Free", f"{gpu['free_mb']:,} MB")
        st.markdown("---")

    # Score table
    st.markdown("### 📊 Per-Chunk Score Breakdown")
    if chunks:
        import pandas as pd

        rows = []
        for i, (doc, score) in enumerate(chunks, 1):
            m = doc.metadata
            rows.append({
                "Rank": i,
                "Source": m.get("source", "?"),
                "Page": m.get("page", "?"),
                "Semantic": m.get("semantic_score", "—"),
                "BM25": m.get("bm25_score", "—"),
                "Fused": m.get("fused_score", "—"),
                "Re-rank": m.get("rerank_score", "—"),
                "Chunk (preview)": doc.page_content[:80] + "…",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    init_session_state()

    # Header
    st.markdown(
        f"""
        <div class="main-header">
            <h1>🔍 {settings.app_title}</h1>
            <p>Hybrid RAG · Semantic + BM25 · CrossEncoder Re-ranking · Local LLM</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pipeline = get_pipeline()
    render_sidebar(pipeline)

    # Main tabs
    tab_chat, tab_chunks, tab_eval = st.tabs(
        ["💬 Chat", "📄 Retrieved Chunks", "🔬 Evaluation"]
    )

    with tab_chat:
        render_chat(pipeline)

    with tab_chunks:
        render_chunks_tab()

    with tab_eval:
        render_eval_tab()


if __name__ == "__main__":
    main()
