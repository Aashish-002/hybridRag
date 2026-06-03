"""
config/settings.py
==================
Central configuration for the Hybrid RAG application.

Purpose:
    Single source of truth for all tunable parameters across the pipeline.
    Loaded at startup; override via environment variables or a .env file.

Workflow:
    app.py → imports settings singleton → passes to each module constructor

Best Practices:
    - Use pydantic BaseSettings for type safety and env-var overrides
    - Never hard-code credentials; pull from environment
    - Group settings by concern (embeddings, retrieval, llm, etc.)

Performance Considerations:
    - Singleton pattern avoids repeated disk reads
    - Pydantic validation catches misconfiguration at startup, not at runtime
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class EmbeddingSettings(BaseSettings):
    model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace sentence-transformer model for dense embeddings.",
        # Other good options: "all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5"
    )
    device: str = Field(
        default="cuda",
        description="Inference device: 'cuda' (GPU) or 'cpu'. RTX 5060 Ti detected.",
    )
    batch_size: int = Field(
        default=128,
        description="Batch size when encoding many chunks. 128 is optimal for RTX 5060 Ti (16 GB).",
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="L2-normalize vectors so dot-product == cosine similarity.",
    )

    model_config = {"env_prefix": "EMBED_", "extra": "ignore"}


class ChunkingSettings(BaseSettings):
    chunk_size: int = Field(default=512, description="Max characters per chunk.")
    chunk_overlap: int = Field(default=64, description="Character overlap between adjacent chunks.")
    separators: List[str] = Field(
        default=["\n\n", "\n", ". ", " ", ""],
        description="Priority-ordered separators for RecursiveCharacterTextSplitter.",
    )
    use_semantic_chunking: bool = Field(
        default=False,
        description="Enable semantic chunking (slower, but respects topic boundaries).",
    )
    semantic_breakpoint_threshold: float = Field(
        default=0.8,
        description="Cosine similarity below which a new semantic chunk starts.",
    )

    model_config = {"env_prefix": "CHUNK_", "extra": "ignore"}


class RetrievalSettings(BaseSettings):
    top_k_semantic: int = Field(default=10, description="Candidate chunks from vector search.")
    top_k_bm25: int = Field(default=10, description="Candidate chunks from BM25 search.")
    top_k_rerank: int = Field(default=5, description="Final chunks kept after re-ranking.")
    fusion_method: Literal["rrf", "weighted"] = Field(
        default="rrf",
        description="Score fusion strategy: 'rrf' = Reciprocal Rank Fusion, 'weighted' = linear blend.",
    )
    rrf_k: int = Field(
        default=60,
        description="RRF constant k. Higher → ranks matter less, scores become more equal.",
    )
    semantic_weight: float = Field(default=0.6, description="Weight for semantic score in weighted fusion.")
    bm25_weight: float = Field(default=0.4, description="Weight for BM25 score in weighted fusion.")
    similarity_threshold: float = Field(
        default=0.0,
        description="Minimum score after fusion to include a chunk. 0.0 = disabled.",
    )

    model_config = {"env_prefix": "RETRIEVAL_", "extra": "ignore"}


class RerankerSettings(BaseSettings):
    model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="CrossEncoder model for query-document relevance scoring.",
    )
    enabled: bool = Field(default=True, description="Set False to skip re-ranking (faster, less accurate).")
    device: str = Field(default="cuda", description="'cuda' (GPU) or 'cpu'. RTX 5060 Ti detected.")

    model_config = {"env_prefix": "RERANKER_", "extra": "ignore"}


class LLMSettings(BaseSettings):
    provider: Literal["lmstudio", "ollama", "openai"] = Field(
        default="lmstudio",
        description="LLM backend. All three expose OpenAI-compatible /v1/chat/completions.",
    )
    base_url: str = Field(
        default="http://localhost:1234/v1",
        description=(
            "API base URL.\n"
            "  LM Studio  → http://localhost:1234/v1\n"
            "  Ollama     → http://localhost:11434/v1\n"
            "  OpenAI     → https://api.openai.com/v1"
        ),
    )
    api_key: str = Field(
        default="lm-studio",
        description="API key. LM Studio accepts any non-empty string.",
    )
    model_name: str = Field(
        default="meta-llama-3-8b-instruct",
        description="Model identifier exactly as shown in LM Studio's loaded model list.",
    )
    temperature: float = Field(default=0.1, description="0 = deterministic, 1 = creative.")
    max_tokens: int = Field(default=1024, description="Max tokens in the generated answer.")
    timeout: int = Field(default=120, description="HTTP request timeout in seconds.")
    streaming: bool = Field(default=True, description="Stream response tokens to the UI.")

    model_config = {"env_prefix": "LLM_", "extra": "ignore"}


class VectorStoreSettings(BaseSettings):
    persist_dir: str = Field(
        default="data/vectorstore",
        description="Directory where the FAISS index is saved to / loaded from.",
    )
    index_name: str = Field(default="hybrid_rag_index", description="FAISS index filename prefix.")

    model_config = {"env_prefix": "VS_", "extra": "ignore"}


# ---------------------------------------------------------------------------
# Top-level aggregator
# ---------------------------------------------------------------------------

class AppSettings(BaseSettings):
    """Aggregates all sub-configs into a single importable object."""

    app_title: str = "Hybrid RAG — Intelligent Document Assistant"
    app_version: str = "1.0.0"
    log_level: str = Field(default="INFO", description="Python logging level.")
    data_dir: str = Field(default="data")
    documents_dir: str = Field(default="data/documents")
    max_upload_size_mb: int = Field(default=50)
    allowed_extensions: List[str] = Field(default=[".pdf", ".docx", ".txt"])
    conversation_memory_k: int = Field(
        default=5,
        description="Number of recent (question, answer) pairs kept in conversation memory.",
    )

    # Nested sub-configs — each reads its own prefixed env vars independently
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        # Ignore unknown env vars (e.g. KMP_DUPLICATE_LIB_OK, system vars)
        "extra": "ignore",
    }


# ---------------------------------------------------------------------------
# Singleton — `from config.settings import settings` throughout the project
# ---------------------------------------------------------------------------
settings = AppSettings()
