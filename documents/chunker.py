"""
documents/chunker.py
====================
Advanced chunking layer: splits raw Document pages into retrieval-sized chunks.

Purpose:
    - Convert large Document texts into small, overlapping chunks suitable
      for embedding and retrieval.
    - Two strategies:
        A. RecursiveCharacterTextSplitter — fast, rule-based, production default
        B. SemanticChunker               — slower, topic-aware, higher quality

Workflow:
    load_document() → [Document pages]
                    → DocumentChunker.chunk()
                    → [Document chunks with enriched metadata]

Best Practices:
    - Always store chunk_index and total_chunks in metadata for ordering
    - Overlap (chunk_overlap) prevents context loss at chunk boundaries
    - Semantic chunking is recommended for long-form documents with clear topics
    - Chunk size 256–512 chars works best for most retrieval models

Performance Considerations:
    - RecursiveCharacterTextSplitter is O(N) — fast even for large corpora
    - SemanticChunker calls the embedding model — budget ~1s per 10 chunks on CPU
    - For 1000-page PDFs, use RecursiveCharacterTextSplitter and batch-embed
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """
    Splits Document objects into smaller, overlapping chunks.

    Supports two strategies selectable via settings.chunking.use_semantic_chunking:
      - False (default): RecursiveCharacterTextSplitter
      - True:            SemanticChunker (requires embedding model)

    Args:
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.
        use_semantic:  Override settings.chunking.use_semantic_chunking.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        use_semantic: bool | None = None,
    ) -> None:
        cfg = settings.chunking
        self.chunk_size = chunk_size or cfg.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.chunk_overlap
        self.use_semantic = use_semantic if use_semantic is not None else cfg.use_semantic_chunking
        self.separators = cfg.separators

        # Build the appropriate splitter
        self._splitter = self._build_splitter()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_splitter(self):
        if self.use_semantic:
            return self._build_semantic_splitter()
        return self._build_recursive_splitter()

    def _build_recursive_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        RecursiveCharacterTextSplitter:
          - Tries each separator in order: paragraph → sentence → word → char
          - Falls back to the next separator if chunk would still be too large
          - Guarantees no chunk exceeds chunk_size characters
        """
        logger.debug(
            "Using RecursiveCharacterTextSplitter (size=%d, overlap=%d).",
            self.chunk_size, self.chunk_overlap,
        )
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            add_start_index=True,  # stores char offset in metadata
        )

    def _build_semantic_splitter(self):
        """
        SemanticChunker (LangChain experimental):
          - Encodes every sentence, detects cosine-similarity drops
          - Groups sentences into coherent topic blocks
          - Falls back to RecursiveCharacterTextSplitter if import fails
        """
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            from embeddings.embedder import get_embedder

            logger.debug(
                "Using SemanticChunker (breakpoint_threshold=%.2f).",
                settings.chunking.semantic_breakpoint_threshold,
            )
            embedder = get_embedder()
            return SemanticChunker(
                embeddings=embedder,
                breakpoint_threshold_type="cosine",
                breakpoint_threshold_amount=settings.chunking.semantic_breakpoint_threshold,
            )
        except ImportError:
            logger.warning(
                "langchain_experimental not installed. "
                "Falling back to RecursiveCharacterTextSplitter."
            )
            return self._build_recursive_splitter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of Document objects into chunks.

        Each output Document preserves the parent's metadata and adds:
          - chunk_index:  0-based position within the parent document
          - total_chunks: total number of chunks from the same source page

        Args:
            documents: List of loaded Document objects (one per page, etc.).

        Returns:
            List of chunk Documents ready for embedding.
        """
        if not documents:
            return []

        all_chunks: List[Document] = []

        for doc in documents:
            # Split this individual document
            raw_chunks: List[Document] = self._splitter.split_documents([doc])

            # Enrich metadata
            total = len(raw_chunks)
            for i, chunk in enumerate(raw_chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = total
                # Ensure a stable unique ID for this chunk
                src = chunk.metadata.get("source", "unknown")
                page = chunk.metadata.get("page", 0)
                chunk.metadata["chunk_id"] = f"{src}::p{page}::c{i}"

            all_chunks.extend(raw_chunks)

        logger.info(
            "Chunked %d document(s) into %d chunks "
            "(strategy=%s, size=%d, overlap=%d).",
            len(documents),
            len(all_chunks),
            "semantic" if self.use_semantic else "recursive",
            self.chunk_size,
            self.chunk_overlap,
        )
        return all_chunks
