"""
utils/helpers.py
================
Utility functions shared across multiple modules.

Purpose:
    Text cleaning, metadata extraction helpers, timing decorators,
    and other stateless functions that don't belong to a single module.

Best Practices:
    - All functions are pure (no side-effects) where possible
    - Type-annotated for IDE auto-complete and static analysis
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalize and clean raw extracted text.

    Steps:
      1. Unicode NFC normalisation (handles ligatures, accents, etc.)
      2. Remove non-printable / control characters (except newlines/tabs)
      3. Collapse excessive whitespace and blank lines
      4. Strip leading/trailing whitespace
    """
    # 1. Unicode normalise
    text = unicodedata.normalize("NFC", text)

    # 2. Remove control characters except \n and \t
    text = re.sub(r"[^\S\n\t ]+", " ", text)          # multiple spaces → single
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)  # non-printable

    # 3. Collapse more than 2 consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Strip
    return text.strip()


def remove_duplicate_chunks(chunks: List[Dict[str, Any]], key: str = "content") -> List[Dict[str, Any]]:
    """
    Remove near-duplicate chunks based on content hash.

    Args:
        chunks: List of chunk dicts with at least a `key` field.
        key:    Dict key whose value is used for deduplication.

    Returns:
        De-duplicated list preserving original order.
    """
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for chunk in chunks:
        h = hashlib.md5(chunk[key].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    return unique


def truncate_text(text: str, max_chars: int = 200) -> str:
    """Truncate text to `max_chars` characters, appending '…' if cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …"


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def get_file_hash(path: str | Path) -> str:
    """Return SHA-256 hex digest of a file (used for dedup/incremental indexing)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def human_readable_size(num_bytes: int) -> str:
    """Convert bytes to a human-readable string (KB / MB / GB)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:,.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------

def timeit(func: Callable) -> Callable:
    """
    Decorator that measures and logs wall-clock execution time.

    Usage:
        @timeit
        def my_slow_function(): ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        # Import here to avoid circular dependency
        from utils.logger import get_logger
        get_logger(__name__).debug("%s() completed in %.3fs", func.__name__, elapsed)
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def extract_file_metadata(path: str | Path) -> Dict[str, Any]:
    """
    Extract basic file metadata without reading content.

    Returns dict with: name, extension, size_bytes, size_human, hash
    """
    p = Path(path)
    stat = p.stat()
    return {
        "name": p.name,
        "stem": p.stem,
        "extension": p.suffix.lower(),
        "size_bytes": stat.st_size,
        "size_human": human_readable_size(stat.st_size),
        "hash": get_file_hash(p),
        "path": str(p.resolve()),
    }


def format_sources(chunks: List[Dict[str, Any]]) -> str:
    """
    Build a deduplicated, human-readable source citation string.

    Args:
        chunks: List of chunk dicts containing 'metadata' with 'source' key.

    Returns:
        Formatted string like "Sources: doc1.pdf (p.3), doc2.txt (p.7)"
    """
    sources = []
    seen = set()
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        src = meta.get("source", "Unknown")
        page = meta.get("page", None)
        label = f"{src}" + (f" (p.{page})" if page else "")
        if label not in seen:
            seen.add(label)
            sources.append(label)
    if not sources:
        return ""
    return "**Sources:** " + " · ".join(sources)


# ---------------------------------------------------------------------------
# GPU diagnostics
# ---------------------------------------------------------------------------

def get_gpu_info() -> Dict[str, Any]:
    """
    Return current GPU status for display in the Streamlit sidebar.

    Returns a dict with keys: available, name, total_mb, allocated_mb,
    free_mb, utilization_pct. All values are 0 / False when no GPU.
    """
    info: Dict[str, Any] = {
        "available": False,
        "name": "CPU only",
        "total_mb": 0,
        "allocated_mb": 0,
        "reserved_mb": 0,
        "free_mb": 0,
    }
    try:
        import torch
        if not torch.cuda.is_available():
            return info
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory // (1024 ** 2)
        allocated = torch.cuda.memory_allocated(0) // (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) // (1024 ** 2)
        info.update({
            "available": True,
            "name": props.name,
            "compute": f"{props.major}.{props.minor}",
            "total_mb": total,
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "free_mb": total - reserved,
        })
    except Exception:
        pass
    return info
