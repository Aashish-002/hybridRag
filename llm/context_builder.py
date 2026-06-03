"""
llm/context_builder.py
=======================
Assembles retrieved chunks into a clean context string for the LLM.

Purpose:
    - Convert a list of re-ranked Document chunks into a formatted string
      that the LLM can read and cite from.
    - Apply compression and deduplication to stay within token budget.
    - Preserve source attribution for citation tracking.

Workflow:
    CrossEncoderReranker → reranked_chunks
    → ContextBuilder.build(query, chunks)
    → (context_string, sources_list)

Context Format:
    [1] Source: document.pdf | Page: 3
    ───────────────────────────────────
    <chunk text here>

    [2] Source: report.docx | Page: 1
    ───────────────────────────────────
    <chunk text here>

Best Practices:
    - Number each chunk so the LLM can reference [1], [2] etc.
    - Include source and page for traceability
    - Deduplicate near-identical chunks before assembly
    - Hard-cap total context length to avoid exceeding LLM context window
    - Compress long chunks by truncating trailing sentences gracefully

Performance Considerations:
    - Context window: 4K–8K tokens is typical; 1 token ≈ 4 characters
    - Truncate chunks to 500 chars for large context windows
    - For very long documents, summarise chunks before assembly
"""

from __future__ import annotations

import hashlib
from typing import List, Tuple

from langchain_core.documents import Document

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# Approximate token budget: 1 token ≈ 4 chars.
# Reserve 512 tokens for the answer; use the rest for context.
_DEFAULT_MAX_CONTEXT_CHARS = 6000  # ~1500 tokens
_CHUNK_SEPARATOR = "\n" + "─" * 50 + "\n"


class ContextBuilder:
    """
    Builds a structured context string from retrieved document chunks.

    Args:
        max_context_chars: Hard cap on total context length in characters.
        deduplicate:       If True, remove near-duplicate chunks.
    """

    def __init__(
        self,
        max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
        deduplicate: bool = True,
    ) -> None:
        self.max_context_chars = max_context_chars
        self.deduplicate = deduplicate

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _content_hash(self, text: str) -> str:
        """MD5 of the first 200 chars — fast near-duplicate detector."""
        return hashlib.md5(text[:200].encode()).hexdigest()

    def _deduplicate(
        self, chunks: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """Remove chunks whose first 200 chars are identical."""
        seen: set = set()
        unique: List[Tuple[Document, float]] = []
        for doc, score in chunks:
            h = self._content_hash(doc.page_content)
            if h not in seen:
                seen.add(h)
                unique.append((doc, score))
        removed = len(chunks) - len(unique)
        if removed:
            logger.debug("Deduplication removed %d duplicate chunks.", removed)
        return unique

    def _format_chunk(self, idx: int, doc: Document) -> str:
        """
        Format a single chunk as a numbered, sourced block.

        Example output:
            [1] Source: paper.pdf | Page: 4 | Score: 0.923
            ──────────────────────────────────────────────────
            The transformer architecture was first proposed …
        """
        meta = doc.metadata
        source = meta.get("source", "Unknown")
        page = meta.get("page", "?")
        rerank = meta.get("rerank_score", meta.get("fused_score", "N/A"))
        score_str = f"{rerank:.4f}" if isinstance(rerank, float) else str(rerank)

        header = f"[{idx}] Source: {source} | Page: {page} | Score: {score_str}"
        return f"{header}\n{'─' * len(header)}\n{doc.page_content.strip()}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        query: str,
        chunks: List[Tuple[Document, float]],
    ) -> Tuple[str, List[str]]:
        """
        Assemble a context string and source citation list from chunks.

        Args:
            query:  User query (used for logging only).
            chunks: List of (Document, score) from the re-ranker.

        Returns:
            Tuple of:
              - context_str:  Formatted multi-chunk context for the LLM.
              - sources:      Deduplicated list of "source (page)" strings.
        """
        if not chunks:
            logger.warning("ContextBuilder.build() called with empty chunk list.")
            return "", []

        if self.deduplicate:
            chunks = self._deduplicate(chunks)

        context_parts: List[str] = []
        sources: List[str] = []
        seen_sources: set = set()
        total_chars = 0

        for idx, (doc, score) in enumerate(chunks, start=1):
            chunk_text = self._format_chunk(idx, doc)

            # Check if adding this chunk would exceed the budget
            if total_chars + len(chunk_text) > self.max_context_chars and idx > 1:
                logger.debug(
                    "Context budget reached at chunk %d/%d (%d chars).",
                    idx - 1, len(chunks), total_chars,
                )
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

            # Track unique sources
            meta = doc.metadata
            src = meta.get("source", "Unknown")
            page = meta.get("page")
            src_label = f"{src} (p.{page})" if page else src
            if src_label not in seen_sources:
                seen_sources.add(src_label)
                sources.append(src_label)

        context_str = _CHUNK_SEPARATOR.join(context_parts)

        logger.info(
            "Context assembled: %d chunks, %d chars, %d unique sources.",
            len(context_parts), total_chars, len(sources),
        )
        return context_str, sources

    def build_simple(
        self,
        chunks: List[Tuple[Document, float]],
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Build a plain text context without numbering or metadata headers.
        Useful for simpler prompts or when metadata is noisy.

        Args:
            chunks:    List of (Document, score).
            separator: String placed between chunks.

        Returns:
            Plain text context string.
        """
        if self.deduplicate:
            chunks = self._deduplicate(chunks)

        parts: List[str] = []
        total_chars = 0

        for doc, _ in chunks:
            text = doc.page_content.strip()
            if total_chars + len(text) > self.max_context_chars and parts:
                break
            parts.append(text)
            total_chars += len(text)

        return separator.join(parts)
