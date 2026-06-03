"""
retrieval/hybrid_retriever.py
==============================
Orchestrates the full hybrid retrieval pipeline.

Purpose:
    Single entry-point that coordinates:
      1. Semantic search (FAISS)
      2. Keyword search (BM25)
      3. Score fusion (RRF or weighted)
    and returns a merged ranked list of chunks with rich score metadata.

Workflow:
    HybridRetriever.retrieve(query)
        ├── FAISSVectorStore.similarity_search_with_scores(query)
        ├── BM25Retriever.retrieve(query)
        └── RetrievalFusion.fuse(semantic, bm25)
        → [(Document, fused_score), …]

Best Practices:
    - Keep retrieval stateless (each call is independent)
    - Attach all intermediate scores to Document.metadata for explainability
    - Expose a `retrieve_with_metadata()` method for the evaluation dashboard

Performance Considerations:
    - FAISS search: O(N·D) ≈ 5–20ms for 50K chunks on CPU
    - BM25 search:  O(N·V) ≈ 10–50ms for 10K chunks on CPU
    - Both can run in parallel (see ThreadPoolExecutor approach below)
    - Total hybrid latency budget: 50–100ms before re-ranking
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from config.settings import settings
from retrieval.bm25_retriever import BM25Retriever
from retrieval.fusion import RetrievalFusion
from utils.logger import get_logger
from vectorstore.faiss_store import FAISSVectorStore

logger = get_logger(__name__)


class HybridRetriever:
    """
    Unified retrieval layer combining semantic and keyword search.

    Args:
        faiss_store:  Pre-built FAISSVectorStore (or None to auto-create).
        bm25_retriever: Pre-built BM25Retriever (or None to auto-create).
        fusion:       RetrievalFusion instance (or None to auto-create).
    """

    def __init__(
        self,
        faiss_store: Optional[FAISSVectorStore] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        fusion: Optional[RetrievalFusion] = None,
    ) -> None:
        self.faiss_store = faiss_store or FAISSVectorStore()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.fusion = fusion or RetrievalFusion()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def index_documents(self, documents: List[Document]) -> None:
        """
        Build both indexes from a list of chunked Documents.

        Runs FAISS embedding and BM25 tokenisation sequentially
        (FAISS embedding is the bottleneck; parallelising doesn't help).

        Args:
            documents: Chunked Document objects from DocumentChunker.
        """
        if not documents:
            logger.warning("index_documents() called with empty list.")
            return

        logger.info("Building FAISS index for %d chunks …", len(documents))
        self.faiss_store.add_documents(documents)

        logger.info("Building BM25 index for %d chunks …", len(documents))
        self.bm25_retriever.build_index(documents)

        logger.info("Both indexes ready.")

    def add_documents(self, documents: List[Document]) -> None:
        """Incrementally add new documents to both indexes."""
        self.faiss_store.add_documents(documents)
        self.bm25_retriever.add_documents(documents)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k_semantic: int | None = None,
        top_k_bm25: int | None = None,
    ) -> List[Tuple[Document, float]]:
        """
        Run hybrid retrieval and return fused results.

        Both searches run in parallel via a ThreadPoolExecutor for lower
        wall-clock latency (useful when one retriever is slow).

        Args:
            query:           Raw user query string.
            top_k_semantic:  Override for number of semantic results.
            top_k_bm25:      Override for number of BM25 results.

        Returns:
            List of (Document, fused_score) sorted descending by fused score.
            Each Document.metadata contains:
                - semantic_score  (if available)
                - bm25_score      (if available)
                - fused_score     (always present after fusion)
        """
        cfg = settings.retrieval
        k_sem = top_k_semantic or cfg.top_k_semantic
        k_bm25 = top_k_bm25 or cfg.top_k_bm25

        t0 = time.perf_counter()

        # Run both retrievers in parallel
        semantic_results: List[Tuple[Document, float]] = []
        bm25_results: List[Tuple[Document, float]] = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_semantic = executor.submit(
                self.faiss_store.similarity_search_with_scores, query, k_sem
            )
            future_bm25 = executor.submit(
                self.bm25_retriever.retrieve, query, k_bm25
            )

            try:
                semantic_results = future_semantic.result()
            except Exception as exc:
                logger.error("Semantic search failed: %s", exc, exc_info=True)

            try:
                bm25_results = future_bm25.result()
            except Exception as exc:
                logger.error("BM25 search failed: %s", exc, exc_info=True)

        # Annotate intermediate scores before fusion
        for doc, score in semantic_results:
            doc.metadata["semantic_score"] = round(score, 4)
        for doc, score in bm25_results:
            doc.metadata["bm25_score"] = round(score, 4)

        # Fuse results
        fused = self.fusion.fuse(semantic_results, bm25_results)

        # Annotate fused score
        for doc, score in fused:
            doc.metadata["fused_score"] = round(score, 6)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Hybrid retrieval: semantic=%d, bm25=%d → fused=%d chunks in %.3fs.",
            len(semantic_results), len(bm25_results), len(fused), elapsed,
        )
        return fused

    def retrieve_with_metadata(self, query: str) -> Dict:
        """
        Run retrieval and return a rich metadata dict for the eval dashboard.

        Returns:
            {
                "query":            str,
                "semantic_results": [(doc, score), ...],
                "bm25_results":     [(doc, score), ...],
                "fused_results":    [(doc, score), ...],
                "latency_ms":       float,
            }
        """
        t0 = time.perf_counter()

        cfg = settings.retrieval
        semantic_results = []
        bm25_results = []

        try:
            semantic_results = self.faiss_store.similarity_search_with_scores(
                query, k=cfg.top_k_semantic
            )
        except Exception as exc:
            logger.error("Semantic search error: %s", exc)

        try:
            bm25_results = self.bm25_retriever.retrieve(query, k=cfg.top_k_bm25)
        except Exception as exc:
            logger.error("BM25 search error: %s", exc)

        fused = self.fusion.fuse(semantic_results, bm25_results)
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "query": query,
            "semantic_results": semantic_results,
            "bm25_results": bm25_results,
            "fused_results": fused,
            "latency_ms": round(latency_ms, 2),
        }

    def is_ready(self) -> bool:
        """Return True only if both indexes have been built."""
        return self.faiss_store.is_ready() and self.bm25_retriever.is_ready()
