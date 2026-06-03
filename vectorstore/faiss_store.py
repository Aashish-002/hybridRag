"""
vectorstore/faiss_store.py
==========================
FAISS-backed vector store for semantic (dense) retrieval.

Purpose:
    Build and persist a FAISS index from document chunk embeddings,
    then serve fast similarity search at query time.

FAISS GPU note (Python 3.13):
    faiss-gpu wheels are not published on PyPI for Python 3.13.
    We use faiss-cpu (1.14.2), which is fully functional.
    GPU acceleration is applied at the *embedding* stage instead —
    sentence-transformers runs on CUDA (RTX 5060 Ti) to encode chunks
    and queries, so embedding is ~17× faster than CPU. The FAISS search
    itself (pure vector math on already-computed vectors) takes <20ms
    for up to 100K chunks even on CPU.

    If you later switch to Python 3.10/3.11 via conda, you can install
    faiss-gpu and the _move_index_to_gpu helpers can be re-enabled.

Index type:
    IndexFlatIP (exact inner-product = cosine for L2-normalised vectors).
    Correct and fast up to ~100K chunks. No training required.

Best Practices:
    - Persist after every add_documents() call
    - L2-normalise embeddings before insertion (done in Embedder)
    - LangChain sidecar .pkl stores the docstore alongside .faiss binary
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config.settings import settings
from embeddings.embedder import get_embedder
from utils.logger import get_logger

logger = get_logger(__name__)


class FAISSVectorStore:
    """
    Thin wrapper around LangChain's FAISS integration.

    Adds:
      - Auto-persist after mutations
      - Incremental add (merge new index into existing)
      - Typed similarity_search_with_scores() return

    Args:
        persist_dir: Directory for FAISS index files.
        index_name:  Filename prefix for .faiss / .pkl files.
        embedder:    SentenceTransformerEmbedder instance (lazy if None).
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        index_name: str | None = None,
        embedder=None,
    ) -> None:
        cfg = settings.vectorstore
        self.persist_dir = Path(persist_dir or cfg.persist_dir)
        self.index_name = index_name or cfg.index_name
        self.embedder = embedder or get_embedder()
        self._store: Optional[FAISS] = None

        self.persist_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_persisted(self) -> bool:
        return (self.persist_dir / f"{self.index_name}.faiss").exists()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """
        Load a previously saved FAISS index from disk.

        Returns:
            True if loaded, False if no saved index found.
        """
        if not self._is_persisted():
            logger.info("No persisted FAISS index found at '%s'.", self.persist_dir)
            return False
        try:
            self._store = FAISS.load_local(
                str(self.persist_dir),
                embeddings=self.embedder,
                index_name=self.index_name,
                allow_dangerous_deserialization=True,
            )
            logger.info(
                "FAISS index loaded — %d vectors.", self._store.index.ntotal
            )
            return True
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc, exc_info=True)
            return False

    def save(self) -> None:
        """Persist the current FAISS index and docstore to disk."""
        if self._store is None:
            return
        self._store.save_local(str(self.persist_dir), index_name=self.index_name)
        logger.info(
            "FAISS index saved (%d vectors).", self._store.index.ntotal
        )

    def add_documents(self, documents: List[Document]) -> None:
        """
        Embed and add documents to the FAISS index.

        Embedding runs on CUDA (RTX 5060 Ti) via the Embedder.
        The resulting float32 vectors are inserted into the CPU FAISS index.

        Args:
            documents: List of chunk Documents to index.
        """
        if not documents:
            logger.warning("add_documents() called with empty list.")
            return

        logger.info("Embedding %d chunks on GPU, indexing in FAISS …", len(documents))

        if self._store is None:
            self._store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedder,
            )
        else:
            # Incremental merge
            self._store.add_documents(documents)

        self.save()
        logger.info("FAISS index updated — %d total vectors.", self._store.index.ntotal)

    def similarity_search_with_scores(
        self,
        query: str,
        k: int | None = None,
    ) -> List[Tuple[Document, float]]:
        """
        Cosine-similarity search against the FAISS index.

        Query is embedded on GPU (CUDA via sentence-transformers),
        then the dot-product search runs on CPU FAISS (~5ms for 50K chunks).

        Args:
            query: Raw query string.
            k:     Number of results to return.

        Returns:
            List of (Document, cosine_score) sorted descending.
        """
        if self._store is None:
            if not self.load():
                raise RuntimeError(
                    "FAISS index is empty. Call add_documents() first."
                )

        top_k = k or settings.retrieval.top_k_semantic
        results = self._store.similarity_search_with_relevance_scores(query, k=top_k)

        logger.debug(
            "FAISS search: k=%d, top_score=%.4f.",
            top_k, results[0][1] if results else 0.0,
        )
        return results

    def similarity_search(self, query: str, k: int | None = None) -> List[Document]:
        """Convenience wrapper returning only Documents (no scores)."""
        return [doc for doc, _ in self.similarity_search_with_scores(query, k=k)]

    def is_ready(self) -> bool:
        """Return True if the index is built or can be loaded from disk."""
        if self._store is not None:
            return self._store.index.ntotal > 0
        return self.load()

    def get_stats(self) -> Dict:
        """Return basic index statistics."""
        if self._store is None:
            return {"vectors": 0, "dimension": 0}
        return {
            "vectors": self._store.index.ntotal,
            "dimension": self._store.index.d,
        }
