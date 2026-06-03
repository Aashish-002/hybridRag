"""
retrieval/bm25_retriever.py
============================
BM25 keyword-based retrieval using the rank-bm25 library.

Purpose:
    - Provide exact/partial keyword matching as a complement to dense
      semantic search. BM25 excels when the query uses specific terms,
      acronyms, IDs, or rare words that embeddings may miss.

Algorithm — BM25 (Best Match 25):
    score(q, d) = Σ IDF(t) · [TF(t,d)·(k1+1)] / [TF(t,d) + k1·(1 - b + b·|d|/avgdl)]

    Parameters:
        k1 = 1.5  — term-frequency saturation (higher → TF matters more)
        b  = 0.75 — document length normalisation (1.0 = full normalisation)

Workflow:
    DocumentChunker → chunks
    → BM25Retriever.build_index(chunks)       [tokenise + build BM25 corpus]
    → BM25Retriever.retrieve(query)           [returns scored chunks]

Best Practices:
    - Lowercase and tokenise consistently between index time and query time
    - Remove common English stopwords for better IDF discrimination
    - Persist the BM25 object + corpus to disk (pickle) for fast reload
    - For multilingual documents, use a language-appropriate tokeniser

Performance Considerations:
    - BM25 is CPU-bound and single-threaded in rank-bm25
    - Corpus of 10K chunks: ~50ms query time on modern CPU
    - For >100K chunks, consider Elasticsearch/OpenSearch for BM25 at scale
"""

from __future__ import annotations

import os
import pickle
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Simple tokeniser
# ---------------------------------------------------------------------------

# NLTK stopwords subset — avoids adding a heavy dependency
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "this", "that", "these", "those", "it", "its", "i", "we", "you",
    "he", "she", "they", "my", "our", "your", "his", "her", "their",
}


def _tokenise(text: str) -> List[str]:
    """
    Lowercase, strip punctuation, split, and remove stopwords.

    Keeping this function simple and deterministic ensures identical
    tokenisation at index time and query time.
    """
    text = text.lower()
    # Remove punctuation (but keep hyphens inside words)
    text = re.sub(r"[^\w\s-]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Keyword retriever backed by BM25Okapi from the rank-bm25 library.

    Attributes:
        corpus:    List of raw chunk strings (needed for score normalisation).
        documents: Parallel list of Document objects (same order as corpus).
        bm25:      Fitted BM25Okapi object.

    Args:
        k1:          BM25 k1 parameter (default 1.5).
        b:           BM25 b  parameter (default 0.75).
        persist_path: File path for pickle serialisation.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        persist_path: str = "data/vectorstore/bm25_index.pkl",
    ) -> None:
        self.k1 = k1
        self.b = b
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        self.corpus: List[str] = []
        self.documents: List[Document] = []
        self.bm25 = None  # BM25Okapi instance

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self, documents: List[Document]) -> None:
        """
        Build (or rebuild) the BM25 index from a list of Document chunks.

        Args:
            documents: Chunked Documents with page_content to index.
        """
        if not documents:
            logger.warning("build_index() called with empty document list.")
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError("rank-bm25 is required: pip install rank-bm25") from exc

        self.documents = documents
        self.corpus = [doc.page_content for doc in documents]
        tokenised_corpus = [_tokenise(text) for text in self.corpus]

        self.bm25 = BM25Okapi(tokenised_corpus, k1=self.k1, b=self.b)
        logger.info(
            "BM25 index built: %d documents, avgdl=%.1f tokens.",
            len(documents),
            sum(len(t) for t in tokenised_corpus) / max(len(tokenised_corpus), 1),
        )
        self.save()

    def add_documents(self, new_documents: List[Document]) -> None:
        """
        Incrementally add documents to the BM25 index.

        BM25Okapi does not support incremental updates natively,
        so we rebuild the full index with the merged corpus.

        Args:
            new_documents: New chunks to add.
        """
        if not new_documents:
            return
        merged = self.documents + new_documents
        logger.info(
            "Incrementally adding %d docs to BM25 (total will be %d).",
            len(new_documents), len(merged),
        )
        self.build_index(merged)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Pickle the BM25 object and parallel corpus/documents to disk."""
        payload = {
            "bm25": self.bm25,
            "corpus": self.corpus,
            "documents": self.documents,
            "k1": self.k1,
            "b": self.b,
        }
        with open(self.persist_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25 index saved to '%s' (%d docs).", self.persist_path, len(self.documents))

    def load(self) -> bool:
        """
        Load BM25 index from disk.

        Returns:
            True on success, False if no saved file found.
        """
        if not self.persist_path.exists():
            logger.info("No BM25 index found at '%s'.", self.persist_path)
            return False
        try:
            with open(self.persist_path, "rb") as f:
                payload = pickle.load(f)
            self.bm25 = payload["bm25"]
            self.corpus = payload["corpus"]
            self.documents = payload["documents"]
            self.k1 = payload.get("k1", self.k1)
            self.b = payload.get("b", self.b)
            logger.info("BM25 index loaded from '%s' (%d docs).", self.persist_path, len(self.documents))
            return True
        except Exception as exc:
            logger.error("Failed to load BM25 index: %s", exc, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int | None = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve the top-k most relevant chunks for a query using BM25.

        Scores are normalised to [0, 1] by dividing by the maximum score
        so they are comparable with cosine similarity scores.

        Args:
            query: Raw query string.
            k:     Number of results. Defaults to settings.retrieval.top_k_bm25.

        Returns:
            List of (Document, normalised_bm25_score) sorted descending.

        Raises:
            RuntimeError: if the index has not been built.
        """
        if self.bm25 is None:
            if not self.load():
                raise RuntimeError(
                    "BM25 index is empty. Call build_index() first."
                )

        top_k = k or settings.retrieval.top_k_bm25
        query_tokens = _tokenise(query)

        if not query_tokens:
            logger.warning("BM25 query tokenised to empty list. Returning empty results.")
            return []

        raw_scores = self.bm25.get_scores(query_tokens)  # numpy array, shape (N,)

        # Normalise to [0, 1]
        max_score = raw_scores.max()
        if max_score > 0:
            norm_scores = raw_scores / max_score
        else:
            norm_scores = raw_scores  # all zeros

        # Get indices of top-k scores (descending)
        top_indices = norm_scores.argsort()[::-1][:top_k]

        results: List[Tuple[Document, float]] = []
        for idx in top_indices:
            score = float(norm_scores[idx])
            if score > 0:  # skip zero-score results
                results.append((self.documents[idx], score))

        logger.debug(
            "BM25 retrieve: query='%s…', k=%d, results=%d, top_score=%.4f",
            query[:50],
            top_k,
            len(results),
            results[0][1] if results else 0.0,
        )
        return results

    def is_ready(self) -> bool:
        """Return True if the index is built or can be loaded from disk."""
        if self.bm25 is not None:
            return True
        return self.load()

    def get_stats(self) -> Dict[str, int]:
        return {
            "documents": len(self.documents),
            "corpus_size": len(self.corpus),
        }
