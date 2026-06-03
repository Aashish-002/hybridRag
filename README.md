# 🔍 Hybrid RAG — Intelligent Document Assistant

A production-ready **Hybrid Retrieval-Augmented Generation** system combining
**Semantic Search (FAISS)** + **Keyword Search (BM25)** + **CrossEncoder Re-ranking**
+ **Local/Cloud LLM generation**.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                   Query Processing                       │
└─────────────────────────────────────────────────────────┘
    │
    ├─────────────────────┬──────────────────────┐
    ▼                     ▼                      │
┌──────────────┐   ┌──────────────┐             │
│  FAISS Vector│   │  BM25 Keyword│             │
│  Search      │   │  Search      │             │
│  (Semantic)  │   │  (Exact)     │             │
└──────┬───────┘   └──────┬───────┘             │
       │                  │                     │
       └─────────┬─────────┘                    │
                 ▼                              │
    ┌────────────────────────┐                  │
    │  Retrieval Fusion      │                  │
    │  RRF or Weighted Blend │                  │
    └────────────┬───────────┘                  │
                 │                              │
                 ▼                              │
    ┌────────────────────────┐                  │
    │  CrossEncoder Re-rank  │                  │
    │  ms-marco-MiniLM-L-6   │                  │
    └────────────┬───────────┘                  │
                 │                              │
                 ▼                              │
    ┌────────────────────────┐                  │
    │  Context Builder       │◄─────────────────┘
    │  + Dedup + Citations   │
    └────────────┬───────────┘
                 │
                 ▼
    ┌────────────────────────┐
    │  LLM Generation        │
    │  Ollama / LM Studio /  │
    │  OpenAI-compatible     │
    └────────────┬───────────┘
                 │
                 ▼
           Final Response
           + Source Citations
           + Score Breakdown
```

---

## Project Structure

```
hybrid_rag/
│
├── app.py                   # Streamlit UI entry-point
├── pipeline.py              # Top-level orchestrator
├── requirements.txt
├── .env.example
│
├── config/
│   └── settings.py          # Pydantic settings (all config in one place)
│
├── documents/
│   ├── loader.py            # PDF / DOCX / TXT ingestion
│   └── chunker.py           # Recursive + semantic chunking
│
├── embeddings/
│   └── embedder.py          # SentenceTransformer wrapper
│
├── vectorstore/
│   └── faiss_store.py       # FAISS index build / search / persist
│
├── retrieval/
│   ├── bm25_retriever.py    # BM25 keyword retrieval
│   ├── fusion.py            # RRF + weighted score fusion
│   └── hybrid_retriever.py  # Orchestrates both retrievers
│
├── reranker/
│   └── cross_encoder.py     # CrossEncoder re-ranking
│
├── llm/
│   ├── generator.py         # LLM response generation
│   └── context_builder.py   # Context assembly + dedup
│
├── ui/
│   └── components.py        # Reusable Streamlit components
│
├── utils/
│   ├── logger.py            # Centralised logging
│   └── helpers.py           # Shared utilities
│
└── data/
    ├── documents/           # Uploaded files saved here
    ├── vectorstore/         # FAISS index + BM25 pickle
    └── logs/                # Rotating log files
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) **or** [LM Studio](https://lmstudio.ai) running locally  
  (or an OpenAI API key)

### 2. Clone and install

```bash
git clone <your-repo-url>
cd hybrid_rag

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure

```bash
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux

# Edit .env to match your setup
# Minimum: set LLM_PROVIDER, LLM_BASE_URL, LLM_MODEL_NAME
```

### 4. Start your LLM

**Ollama (recommended):**
```bash
ollama pull llama3
ollama serve
```

**LM Studio:**
- Open LM Studio → Local Server → Start Server
- Change `LLM_BASE_URL=http://localhost:1234/v1` in `.env`

**OpenAI:**
- Set `LLM_PROVIDER=openai`
- Set `LLM_BASE_URL=https://api.openai.com/v1`
- Set `LLM_API_KEY=sk-...`

### 5. Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Usage

1. **Upload documents** — drag PDF / DOCX / TXT files into the sidebar
2. **Click "Index"** — the system embeds and indexes all chunks
3. **Ask questions** — type in the chat box
4. **Review results** — check the "Retrieved Chunks" and "Evaluation" tabs

---

## Configuration Reference

All settings are in `config/settings.py` and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `EMBED_DEVICE` | `cpu` | `cpu` or `cuda` |
| `CHUNK_CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `RETRIEVAL_TOP_K_SEMANTIC` | `10` | FAISS candidates |
| `RETRIEVAL_TOP_K_BM25` | `10` | BM25 candidates |
| `RETRIEVAL_TOP_K_RERANK` | `5` | Final chunks after re-rank |
| `RETRIEVAL_FUSION_METHOD` | `rrf` | `rrf` or `weighted` |
| `RERANKER_ENABLED` | `true` | Toggle CrossEncoder |
| `LLM_PROVIDER` | `ollama` | `ollama`, `lmstudio`, `openai` |
| `LLM_MODEL_NAME` | `llama3` | Model identifier |
| `LLM_TEMPERATURE` | `0.1` | Generation temperature |

---

## GPU Acceleration

```bash
# 1. Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Install faiss-gpu instead of faiss-cpu
pip uninstall faiss-cpu
pip install faiss-gpu

# 3. Update .env
EMBED_DEVICE=cuda
RERANKER_DEVICE=cuda
```

---

## Supported LLM Models

| Provider | Models |
|---|---|
| Ollama | llama3, qwen3, gemma3, mistral, phi3, deepseek-r1 |
| LM Studio | Any GGUF model loaded in LM Studio |
| OpenAI | gpt-4o, gpt-4o-mini, gpt-3.5-turbo |

---

## Module Overview

| Module | Responsibility |
|---|---|
| `documents/loader.py` | Parse PDF/DOCX/TXT → LangChain Documents |
| `documents/chunker.py` | Recursive + semantic text splitting |
| `embeddings/embedder.py` | SentenceTransformer dense vectors |
| `vectorstore/faiss_store.py` | FAISS ANN index, persist/load |
| `retrieval/bm25_retriever.py` | BM25Okapi keyword retrieval |
| `retrieval/fusion.py` | RRF + weighted score fusion |
| `retrieval/hybrid_retriever.py` | Parallel semantic+BM25 orchestration |
| `reranker/cross_encoder.py` | CrossEncoder relevance re-ranking |
| `llm/context_builder.py` | Context assembly, dedup, citations |
| `llm/generator.py` | OpenAI-compatible LLM generation |
| `pipeline.py` | End-to-end pipeline orchestrator |
| `app.py` | Streamlit UI |

---

## License

MIT
