"""
ui/components.py
================
Reusable Streamlit UI components for the Hybrid RAG application.

Purpose:
    Extract complex rendering logic out of app.py into small,
    testable, single-purpose functions.

Components:
    - render_chunk_card()        : Display a single retrieved chunk
    - render_score_bar()         : Visual score bar (semantic / BM25 / rerank)
    - render_evaluation_panel()  : Full evaluation dashboard
    - render_sources_panel()     : Source citation panel
    - render_latency_metrics()   : Latency breakdown display
    - render_settings_sidebar()  : Interactive settings panel

Best Practices:
    - Keep each component stateless (accept data, render, return nothing)
    - Use st.expander for details to keep the UI clean
    - Colour-code scores: green (high), yellow (medium), red (low)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import streamlit as st
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    """Return a hex colour based on a [0,1] score."""
    if score >= 0.7:
        return "#28a745"   # green
    elif score >= 0.4:
        return "#ffc107"   # amber
    else:
        return "#dc3545"   # red


def _score_badge(label: str, score: float) -> str:
    """Render a coloured HTML badge for a score."""
    color = _score_color(score)
    return (
        f'<span style="background:{color};color:white;'
        f'padding:2px 8px;border-radius:12px;font-size:0.8em;">'
        f'{label}: {score:.4f}</span>'
    )


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def render_chunk_card(
    idx: int,
    doc: Document,
    score: float,
    expanded: bool = False,
) -> None:
    """
    Render a single retrieved chunk as a styled card with metadata.

    Args:
        idx:      1-based chunk index.
        doc:      The Document object.
        score:    Final score (rerank or fused).
        expanded: Whether the expander starts open.
    """
    meta = doc.metadata
    source = meta.get("source", "Unknown")
    page = meta.get("page", "?")
    semantic = meta.get("semantic_score", None)
    bm25 = meta.get("bm25_score", None)
    rerank = meta.get("rerank_score", None)

    title = f"📄 [{idx}] {source} — Page {page}   |   Score: {score:.4f}"

    with st.expander(title, expanded=expanded):
        # Score badges
        badges = []
        if semantic is not None:
            badges.append(_score_badge("Semantic", semantic))
        if bm25 is not None:
            badges.append(_score_badge("BM25", bm25))
        if rerank is not None:
            badges.append(_score_badge("Re-rank", rerank))

        if badges:
            st.markdown(" &nbsp; ".join(badges), unsafe_allow_html=True)
            st.markdown("")

        # Chunk text
        st.text_area(
            label="Content",
            value=doc.page_content,
            height=150,
            disabled=True,
            label_visibility="collapsed",
            key=f"chunk_{idx}_{id(doc)}",
        )

        # Metadata detail
        with st.expander("📊 Metadata", expanded=False):
            display_meta = {
                k: v for k, v in meta.items()
                if k not in {"chunk_id", "start_index"}
            }
            st.json(display_meta)


def render_score_bar(label: str, score: float, max_score: float = 1.0) -> None:
    """
    Render a labelled progress bar for a score.

    Args:
        label:     Display label.
        score:     Raw score value.
        max_score: Maximum expected score (for normalisation).
    """
    normalised = min(score / max_score, 1.0) if max_score > 0 else 0.0
    color = _score_color(normalised)
    st.markdown(
        f"""
        <div style="margin-bottom:6px;">
            <span style="font-size:0.85em;color:#555;">{label}</span>
            <div style="background:#eee;border-radius:4px;height:12px;margin-top:2px;">
                <div style="background:{color};width:{normalised*100:.1f}%;
                            height:12px;border-radius:4px;"></div>
            </div>
            <span style="font-size:0.75em;color:#888;">{score:.4f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_evaluation_panel(retrieval_meta: Dict) -> None:
    """
    Render the full evaluation dashboard in a Streamlit expander.

    Args:
        retrieval_meta: Dict from RAGPipeline.query() with timing and counts.
    """
    with st.expander("🔬 Evaluation Dashboard", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Latency", f"{retrieval_meta.get('total_ms', 0)} ms")
        col2.metric("Retrieval", f"{retrieval_meta.get('retrieval_ms', 0)} ms")
        col3.metric("Re-ranking", f"{retrieval_meta.get('rerank_ms', 0)} ms")
        col4.metric("LLM", f"{retrieval_meta.get('llm_ms', 0)} ms")

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("Chunks Retrieved", retrieval_meta.get("num_retrieved", 0))
        c2.metric("After Re-ranking", retrieval_meta.get("num_reranked", 0))


def render_sources_panel(sources: List[str]) -> None:
    """
    Display a formatted list of source citations.

    Args:
        sources: List of "filename (p.N)" strings.
    """
    if not sources:
        return
    with st.expander("📚 Sources Used", expanded=True):
        for i, src in enumerate(sources, 1):
            st.markdown(f"**[{i}]** {src}")


def render_latency_metrics(retrieval_meta: Dict) -> None:
    """
    Compact inline latency display (no expander).
    """
    cols = st.columns(4)
    labels = ["Total", "Retrieval", "Re-rank", "LLM"]
    keys = ["total_ms", "retrieval_ms", "rerank_ms", "llm_ms"]
    for col, label, key in zip(cols, labels, keys):
        col.metric(label, f"{retrieval_meta.get(key, '—')} ms")


def render_settings_sidebar() -> Dict:
    """
    Render an interactive settings panel in the Streamlit sidebar.

    Returns:
        Dict of user-selected overrides.
    """
    st.sidebar.header("⚙️ Settings")

    with st.sidebar.expander("🔍 Retrieval", expanded=True):
        top_k_semantic = st.slider("Semantic top-k", 1, 20, 10, key="sk_sem")
        top_k_bm25 = st.slider("BM25 top-k", 1, 20, 10, key="sk_bm25")
        top_k_rerank = st.slider("Re-rank top-k", 1, 10, 5, key="sk_rer")
        fusion_method = st.selectbox(
            "Fusion method", ["rrf", "weighted"], key="fusion_method"
        )

    with st.sidebar.expander("🤖 LLM", expanded=False):
        model_name = st.text_input("Model name", value="llama3", key="llm_model")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05, key="temp")
        provider = st.selectbox(
            "Provider", ["ollama", "lmstudio", "openai"], key="provider"
        )
        base_url = st.text_input(
            "Base URL",
            value="http://localhost:11434/v1",
            key="base_url",
        )

    with st.sidebar.expander("🧩 Chunking", expanded=False):
        chunk_size = st.slider("Chunk size (chars)", 128, 1024, 512, 64, key="csize")
        chunk_overlap = st.slider("Chunk overlap", 0, 256, 64, 16, key="coverlap")

    with st.sidebar.expander("🔄 Re-ranking", expanded=False):
        reranker_enabled = st.toggle("Enable re-ranking", value=True, key="rer_on")

    return {
        "top_k_semantic": top_k_semantic,
        "top_k_bm25": top_k_bm25,
        "top_k_rerank": top_k_rerank,
        "fusion_method": fusion_method,
        "model_name": model_name,
        "temperature": temperature,
        "provider": provider,
        "base_url": base_url,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "reranker_enabled": reranker_enabled,
    }
