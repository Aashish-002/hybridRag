"""
retrieval/fusion.py
====================
Combines results from semantic (dense) and keyword (BM25) retrievers
into a single ranked list.

Purpose:
    Hybrid RAG outperforms pure semantic or pure keyword retrieval because:
    - Semantic search handles paraphrasing, synonyms, and concept matching
    - BM25 handles exact terms, acronyms, IDs, rare words
    Fusion merges both ranked lists to get the best of both worlds.

Two Fusion Strategies:
    A. Reciprocal Rank Fusion (RRF)  [default]
       score_rrf(d) = Σ 1 / (k + rank_i(d))
       - Rank-based: absolute scores don't matter, only relative ordering
       - Robust to score distribution mismatches between retrievers
       - k=60 is the standard constant from the original paper

    B. Weighted Score Fusion
       score(d) = α·semantic_score + β·bm25_score
       - Requires scores to be on comparable scales (both normalised to [0,1])
       - More tunable; use when you know retriever quality ratio upfront

Workflow:
    semantic_results [(doc, score)] + bm25_results [(doc, score)]
    → RetrievalFusion.fuse()
    → merged [(doc, fused_score)]  sorted descending

Best Practices:
    - Use RRF by default; switch to weighted only after evaluation
    - De-duplicate by chunk_id (same chunk from both retrievers = merge)
    - Always normalise BM25 scores before weighted fusion

Performance Considerations:
    - Fusion is O(N log N) — negligible compared to embedding/search
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from langchain_core.documents import Document

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_chunk_id(doc: Document) -> str:
    """
    Return a stable identifier for deduplication.
    Falls back to content hash if chunk_id metadata is absent.
    """
    cid = doc.metadata.get("chunk_id")
    if cid:
        return str(cid)
    import hashlib
    return hashlib.md5(doc.page_content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Fusion class
# ---------------------------------------------------------------------------

class RetrievalFusion:
    """
    Merges semantic and BM25 result lists using RRF or weighted fusion.

    Args:
        method:          'rrf' or 'weighted'.
        rrf_k:           RRF smoothing constant (default 60).
        semantic_weight: Weight for semantic scores in weighted mode.
        bm25_weight:     Weight for BM25 scores in weighted mode.
    """

    def __init__(
        self,
        method: str | None = None,
        rrf_k: int | None = None,
        semantic_weight: float | None = None,
        bm25_weight: float | None = None,
    ) -> None:
        cfg = settings.retrieval
        self.method = method or cfg.fusion_method
        self.rrf_k = rrf_k or cfg.rrf_k
        self.semantic_weight = semantic_weight or cfg.semantic_weight
        self.bm25_weight = bm25_weight or cfg.bm25_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        semantic_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """
        Fuse two ranked result lists into a single ranked list.

        Args:
            semantic_results: [(Document, cosine_score), …] sorted desc.
            bm25_results:     [(Document, bm25_norm_score), …] sorted desc.

        Returns:
            Merged list [(Document, fused_score), …] sorted descending.
        """
        if self.method == "rrf":
            return self._reciprocal_rank_fusion(semantic_results, bm25_results)
        elif self.method == "weighted":
            return self._weighted_fusion(semantic_results, bm25_results)
        else:
            raise ValueError(f"Unknown fusion method: '{self.method}'. Use 'rrf' or 'weighted'.")

    # ------------------------------------------------------------------
    # Strategy A — Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """
        RRF formula:  score(d) = Σ_r  1 / (k + rank_r(d))

        A document appearing at rank 1 in both lists gets:
            1/(60+1) + 1/(60+1) ≈ 0.033   (highest possible)

        A document appearing only in one list at rank 1 gets:
            1/(60+1) ≈ 0.016

        k=60 prevents top-rank documents from dominating too strongly.
        """
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Process semantic ranks (1-indexed)
        for rank, (doc, _score) in enumerate(semantic_results, start=1):
            cid = _get_chunk_id(doc)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            doc_map[cid] = doc

        # Process BM25 ranks (1-indexed)
        for rank, (doc, _score) in enumerate(bm25_results, start=1):
            cid = _get_chunk_id(doc)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            doc_map[cid] = doc

        # Sort by fused RRF score descending
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = [(doc_map[cid], score) for cid, score in fused]
        logger.debug(
            "RRF fusion: %d semantic + %d BM25 → %d unique chunks (k=%d).",
            len(semantic_results), len(bm25_results), len(results), self.rrf_k,
        )
        return results

    # ------------------------------------------------------------------
    # Strategy B — Weighted Score Fusion
    # ------------------------------------------------------------------

    def _weighted_fusion(
        self,
        semantic_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """
        Weighted fusion:  score(d) = α·semantic_score + β·bm25_score

        Both input scores should be normalised to [0, 1] before calling.
        Documents that appear in only one list get a score of 0 for the other.
        """
        combined: Dict[str, Dict] = {}

        for doc, score in semantic_results:
            cid = _get_chunk_id(doc)
            combined[cid] = {
                "doc": doc,
                "semantic": score,
                "bm25": combined.get(cid, {}).get("bm25", 0.0),
            }

        for doc, score in bm25_results:
            cid = _get_chunk_id(doc)
            if cid in combined:
                combined[cid]["bm25"] = score
            else:
                combined[cid] = {"doc": doc, "semantic": 0.0, "bm25": score}

        # Compute weighted sum
        scored = []
        for cid, entry in combined.items():
            fused_score = (
                self.semantic_weight * entry["semantic"]
                + self.bm25_weight * entry["bm25"]
            )
            # Attach individual scores for the evaluation dashboard
            entry["doc"].metadata["semantic_score"] = round(entry["semantic"], 4)
            entry["doc"].metadata["bm25_score"] = round(entry["bm25"], 4)
            entry["doc"].metadata["fused_score"] = round(fused_score, 4)
            scored.append((entry["doc"], fused_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            "Weighted fusion (α=%.2f, β=%.2f): %d semantic + %d BM25 → %d unique chunks.",
            self.semantic_weight, self.bm25_weight,
            len(semantic_results), len(bm25_results), len(scored),
        )
        return scored
