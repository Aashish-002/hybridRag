"""
pipeline.py
===========
Top-level orchestrator for the complete Hybrid RAG pipeline.

Purpose:
    Wire together every module into a single, easy-to-use RAGPipeline class.
    The Streamlit UI (app.py) creates one RAGPipeline instance and calls:
      - pipeline.ingest(file_bytes, filename)   → index a new document
      - pipeline.query(question)                → full RAG answer

Architecture Flow:
    ingest():
        load_from_bytes() → DocumentChunker → HybridRetriever.index_documents()

    query():
        HybridRetriever.retrieve()
        → CrossEncoderReranker.rerank()
        → ContextBuilder.build()
        → ResponseGenerator.generate_stream()
        → answer + sources + debug_info

Best Practices:
    - Stateless query path: each call is independent
    - Lazy-load expensive models (embedder, cross-encoder) on first use
    - Return a rich result dict so the UI can render scores and sources
    - Catch and log all errors; surface user-friendly messages
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from langchain_core.documents import Document

from config.settings import settings
from documents.chunker import DocumentChunker
from documents.loader import load_document, load_from_bytes
from llm.context_builder import ContextBuilder
from llm.generator import ResponseGenerator
from reranker.cross_encoder import CrossEncoderReranker
from retrieval.hybrid_retriever import HybridRetriever
from utils.helpers import format_sources
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """
    Unified Hybrid RAG pipeline.

    Instantiate once; call ingest() for each new document and query()
    for each user question.

    Args:
        chunker:    DocumentChunker instance (auto-created if None).
        retriever:  HybridRetriever instance (auto-created if None).
        reranker:   CrossEncoderReranker instance (auto-created if None).
        ctx_builder: ContextBuilder instance (auto-created if None).
        generator:  ResponseGenerator instance (auto-created if None).
    """

    def __init__(
        self,
        chunker: Optional[DocumentChunker] = None,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        ctx_builder: Optional[ContextBuilder] = None,
        generator: Optional[ResponseGenerator] = None,
    ) -> None:
        self.chunker = chunker or DocumentChunker()
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.ctx_builder = ctx_builder or ContextBuilder()
        self.generator = generator or ResponseGenerator()

        # Conversation memory: list of {"role": ..., "content": ...} dicts
        self.memory: List[dict] = []

        # Warm up GPU kernels so the first real query has no compile overhead
        self._warmup()
        logger.info("RAGPipeline initialised.")

    def _warmup(self) -> None:
        """
        Trigger lazy model loads and warm up CUDA kernels at startup.

        This prevents the first user query from paying the one-time cost of
        CUDA kernel compilation (~200–600ms on first use).
        """
        try:
            from embeddings.embedder import get_embedder
            emb = get_embedder()
            emb.embed_query("warmup")          # loads SentenceTransformer + CUDA
            if hasattr(self.reranker, "warmup"):
                self.reranker.warmup()         # loads CrossEncoder + CUDA
            logger.info("GPU warmup complete.")
        except Exception as exc:
            logger.warning("GPU warmup skipped: %s", exc)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, file_bytes: bytes, filename: str) -> Dict:
        """
        Ingest a document from raw bytes (e.g. from Streamlit uploader).

        Pipeline:
            bytes → load_from_bytes → chunk → index (FAISS + BM25)

        Args:
            file_bytes: Raw file bytes.
            filename:   Original filename (used for extension detection).

        Returns:
            Dict with keys: filename, num_pages, num_chunks, elapsed_ms
        """
        t0 = time.perf_counter()

        # 1. Save and load
        docs = load_from_bytes(
            file_bytes=file_bytes,
            filename=filename,
            save_dir=settings.documents_dir,
        )
        num_pages = len(docs)
        logger.info("Loaded '%s' → %d pages/sections.", filename, num_pages)

        # 2. Chunk
        chunks = self.chunker.chunk(docs)
        num_chunks = len(chunks)
        logger.info("Chunked into %d chunks.", num_chunks)

        # 3. Index (FAISS + BM25)
        self.retriever.index_documents(chunks)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Ingestion complete: '%s' → %d pages, %d chunks in %.0fms.",
            filename, num_pages, num_chunks, elapsed_ms,
        )
        return {
            "filename": filename,
            "num_pages": num_pages,
            "num_chunks": num_chunks,
            "elapsed_ms": round(elapsed_ms),
        }

    def ingest_file(self, file_path: str) -> Dict:
        """
        Ingest a document directly from a file path.

        Useful for batch ingestion scripts and the CLI.
        """
        path = Path(file_path)
        with open(path, "rb") as f:
            file_bytes = f.read()
        return self.ingest(file_bytes, path.name)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        use_memory: bool = True,
        stream: bool | None = None,
    ) -> Dict:
        """
        Run the full RAG pipeline for a user question.

        Pipeline:
            question
            → Hybrid retrieval (FAISS + BM25 + Fusion)
            → CrossEncoder re-ranking
            → Context assembly
            → LLM generation
            → answer + sources + debug metadata

        Args:
            question:    User's natural language question.
            use_memory:  Include conversation history in the LLM prompt.
            stream:      Override settings.llm.streaming.

        Returns:
            Dict with keys:
              - answer:         str  (full answer)
              - sources:        List[str]  (citation strings)
              - chunks:         List[Tuple[Document, float]]  (re-ranked)
              - retrieval_meta: Dict  (scores, latency breakdown)
        """
        t0 = time.perf_counter()

        # 1. Hybrid retrieval
        t_ret = time.perf_counter()
        fused_results = self.retriever.retrieve(question)
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        if not fused_results:
            logger.warning("No results retrieved for query: '%s'", question)
            return {
                "answer": "I could not find this information in the provided documents.",
                "sources": [],
                "chunks": [],
                "retrieval_meta": {"retrieval_ms": round(retrieval_ms)},
            }

        # 2. Re-ranking
        t_rer = time.perf_counter()
        reranked = self.reranker.rerank(question, fused_results)
        rerank_ms = (time.perf_counter() - t_rer) * 1000

        # 3. Context assembly
        context_str, sources = self.ctx_builder.build(question, reranked)

        # 4. LLM generation
        t_llm = time.perf_counter()
        should_stream = stream if stream is not None else settings.llm.streaming

        if should_stream:
            # Return a streaming generator for Streamlit; accumulate for memory
            answer_parts: List[str] = []
            def _streaming_answer():
                for token in self.generator.generate_stream(context_str, question):
                    answer_parts.append(token)
                    yield token
            answer = _streaming_answer()
        else:
            if use_memory:
                answer = self.generator.generate_with_memory(
                    context_str, question, self.memory
                )
            else:
                answer = self.generator.generate(context_str, question)

        llm_ms = (time.perf_counter() - t_llm) * 1000
        total_ms = (time.perf_counter() - t0) * 1000

        # 5. Update conversation memory (non-streaming only)
        if not should_stream and isinstance(answer, str):
            self.memory.append({"role": "user", "content": question})
            self.memory.append({"role": "assistant", "content": answer})
            # Keep only the last K pairs
            k = settings.conversation_memory_k * 2
            if len(self.memory) > k:
                self.memory = self.memory[-k:]

        logger.info(
            "Query complete in %.0fms "
            "(retrieval=%.0fms, rerank=%.0fms, llm=%.0fms). "
            "%d sources.",
            total_ms, retrieval_ms, rerank_ms, llm_ms, len(sources),
        )

        return {
            "answer": answer,
            "sources": sources,
            "chunks": reranked,
            "retrieval_meta": {
                "retrieval_ms": round(retrieval_ms),
                "rerank_ms": round(rerank_ms),
                "llm_ms": round(llm_ms),
                "total_ms": round(total_ms),
                "num_retrieved": len(fused_results),
                "num_reranked": len(reranked),
            },
        }

    def query_stream(self, question: str) -> Iterator[str]:
        """
        Convenience method: run full RAG pipeline and stream answer tokens.

        Useful for Streamlit's st.write_stream() integration.

        Args:
            question: User question.

        Yields:
            Answer tokens as they are generated.
        """
        fused_results = self.retriever.retrieve(question)
        if not fused_results:
            yield "I could not find this information in the provided documents."
            return

        reranked = self.reranker.rerank(question, fused_results)
        context_str, _ = self.ctx_builder.build(question, reranked)
        yield from self.generator.generate_stream(context_str, question)

    def clear_memory(self) -> None:
        """Reset conversation history."""
        self.memory.clear()
        logger.info("Conversation memory cleared.")

    def is_ready(self) -> bool:
        """Return True if at least one document has been indexed."""
        return self.retriever.is_ready()
