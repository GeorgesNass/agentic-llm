'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Retrieval public API: ingestion, chunking, embeddings and RAG search. Uses vector stores from retrieval_store."
'''

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.core.errors import NotFoundError, RetrievalError, StorageError
from src.llm.embeddings import EmbeddingsResult, embed_texts
from src.orchestrator.vector_store import get_vector_store, get_vector_store_info
from src.utils.logging_utils import get_logger, log_execution_time
from src.utils.safe_utils import _safe_str
from src.utils.env_utils import _get_env_int
from src.utils.io_utils import _list_text_files, safe_read_text_file
from src.utils.llm_utils import _dedupe_chunks
from src.utils.text_utils import _normalize_text

logger = get_logger(__name__)

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(frozen=True)
class Chunk:
    """
        Text chunk with metadata

        Args:
            chunk_id: Stable chunk id
            text: Chunk content
            source: Source name or path
            start_char: Start char offset
            end_char: End char offset
    """

    chunk_id: str
    text: str
    source: str
    start_char: int
    end_char: int

@dataclass(frozen=True)
class SearchResult:
    """
        Retrieval search result

        Args:
            chunk_id: Chunk id
            text: Chunk text
            source: Source path
            score: Similarity score
            metadata: Metadata dict
    """

    chunk_id: str
    text: str
    source: str
    score: float
    metadata: Dict[str, Any]

## ============================================================
## CHUNKING
## ============================================================
def chunk_text(
    text: str,
    *,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Chunk]:
    """
        Chunk text into overlapping character windows

        Args:
            text: Input text
            source: Source identifier or file path
            chunk_size: Window size in chars
            chunk_overlap: Overlap size in chars

        Returns:
            List of Chunk
    """

    ## Normalize before chunking
    normalized = _normalize_text(text)

    ## Validate params
    if chunk_size <= 0:
        raise RetrievalError(
            message="Invalid chunk_size",
            error_code="retrieval_error",
            details={"chunk_size": chunk_size},
            origin="retrieval",
            cause=None,
            http_status=400,
            is_retryable=False,
        )

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise RetrievalError(
            message="Invalid chunk_overlap",
            error_code="retrieval_error",
            details={"chunk_overlap": chunk_overlap, "chunk_size": chunk_size},
            origin="retrieval",
            cause=None,
            http_status=400,
            is_retryable=False,
        )

    ## Sliding window
    chunks: List[Chunk] = []
    start = 0
    idx = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        part = normalized[start:end].strip()

        ## Skip empty chunks
        if part:
            chunk_id = f"{Path(source).name}:{idx}:{start}-{end}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=part,
                    source=source,
                    start_char=start,
                    end_char=end,
                )
            )

        ## Move forward with overlap
        start = end - chunk_overlap
        if start < 0:
            start = 0

        idx += 1

        ## Stop if we reached end
        if end >= len(normalized):
            break

    return chunks

## ============================================================
## PUBLIC INGESTION
## ============================================================
@log_execution_time
def ingest_folder_to_vector_store(
    folder: str | Path,
    *,
    embedding_provider: str,
    embedding_model: Optional[str],
    use_gpu: bool,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> Dict[str, Any]:
    """
        Ingest a folder into the vector store

        High-level workflow:
            1) List files
            2) Read each file
            3) Chunk content
            4) Embed chunks
            5) Store vectors + metadata

        Env variables:
            CHUNK_SIZE
            CHUNK_OVERLAP

        Args:
            folder: Folder path
            embedding_provider: Embedding provider
            embedding_model: Optional model override
            use_gpu: GPU for local embeddings
            chunk_size: Chunk size override
            chunk_overlap: Chunk overlap override

        Returns:
            Summary dict
    """

    folder_path = Path(folder).expanduser().resolve()

    ## Resolve chunk params
    cs = chunk_size if chunk_size is not None else _get_env_int("CHUNK_SIZE", 900)
    ov = chunk_overlap if chunk_overlap is not None else _get_env_int("CHUNK_OVERLAP", 120)

    ## Validate folder
    if not folder_path.exists():
        raise NotFoundError(
            message="Ingestion folder not found",
            error_code="not_found",
            details={"folder": str(folder_path)},
            origin="retrieval",
            cause=None,
            http_status=404,
            is_retryable=False,
        )

    ## List files
    files = _list_text_files(folder_path)
    if not files:
        info = get_vector_store_info()
        return {
            "folder": str(folder_path),
            "files": 0,
            "chunks": 0,
            "inserted": 0,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model or "",
            "vector_backend": info.get("backend", ""),
            "vector_store_dir": info.get("store_dir", ""),
            "index_name": info.get("index_name", ""),
        }

    ## Read and chunk
    all_chunks: List[Chunk] = []
    for f in files:
        text = safe_read_text_file(f)
        all_chunks.extend(chunk_text(text, source=str(f), chunk_size=cs, chunk_overlap=ov))

    ## Dedupe to reduce noise
    all_chunks = _dedupe_chunks(all_chunks)

    if not all_chunks:
        info = get_vector_store_info()
        return {
            "folder": str(folder_path),
            "files": int(len(files)),
            "chunks": 0,
            "inserted": 0,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model or "",
            "vector_backend": info.get("backend", ""),
            "vector_store_dir": info.get("store_dir", ""),
            "index_name": info.get("index_name", ""),
        }

    ## Embed
    texts = [c.text for c in all_chunks]
    try:
        emb: EmbeddingsResult = embed_texts(
            texts=texts,
            provider=embedding_provider,  # type: ignore[arg-type]
            model=embedding_model,
            use_gpu=use_gpu,
            normalize=True,
        )
    except Exception as exc:
        raise RetrievalError(
            message="Failed to compute embeddings for ingestion",
            error_code="retrieval_error",
            details={"provider": embedding_provider, "count": int(len(texts)), "cause": _safe_str(exc)},
            origin="retrieval",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc

    ## Convert to numpy
    vectors = np.asarray(emb.vectors, dtype=np.float32)
    if vectors.ndim != 2 or int(vectors.shape[0]) != int(len(all_chunks)):
        raise RetrievalError(
            message="Embeddings output has invalid shape",
            error_code="retrieval_error",
            details={"shape": list(vectors.shape), "chunks": int(len(all_chunks))},
            origin="retrieval",
            cause=None,
            http_status=500,
            is_retryable=False,
        )

    ## Insert
    store = get_vector_store()

    try:
        inserted = int(store.add(vectors=vectors, chunks=all_chunks))
    except Exception as exc:
        raise RetrievalError(
            message="Failed to insert embeddings into vector store",
            error_code="retrieval_error",
            details={"cause": _safe_str(exc)},
            origin="retrieval",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc

    ## Build summary
    info = get_vector_store_info()
    return {
        "folder": str(folder_path),
        "files": int(len(files)),
        "chunks": int(len(all_chunks)),
        "inserted": inserted,
        "embedding_provider": embedding_provider,
        "embedding_model": emb.model,
        "vector_backend": info.get("backend", ""),
        "vector_store_dir": info.get("store_dir", ""),
        "index_name": info.get("index_name", ""),
    }

## ============================================================
## PUBLIC SEARCH
## ============================================================
@log_execution_time
def rag_search(
    query: str,
    *,
    top_k: int = 5,
    embedding_provider: str = "local",
    embedding_model: Optional[str] = None,
    use_gpu: bool = False,
) -> List[SearchResult]:
    """
        Search relevant chunks for a query

        Args:
            query: Query string
            top_k: Top results
            embedding_provider: Provider for query embedding
            embedding_model: Optional model override
            use_gpu: GPU usage for local embeddings

        Returns:
            List of SearchResult
    """

    ## Normalize query
    q = _normalize_text(query)
    if not q:
        return []

    ## Embed query
    emb = embed_texts(
        texts=[q],
        provider=embedding_provider,  # type: ignore[arg-type]
        model=embedding_model,
        use_gpu=use_gpu,
        normalize=True,
    )

    if not emb.vectors:
        return []

    query_vector = np.asarray(emb.vectors[0], dtype=np.float32)

    ## Search store
    store = get_vector_store()
    raw_results = store.search(query_vector=query_vector, top_k=int(top_k))

    ## Convert to typed results
    results: List[SearchResult] = []
    for r in raw_results:
        if not isinstance(r, dict):
            continue
        results.append(
            SearchResult(
                chunk_id=str(r.get("chunk_id", "")),
                text=str(r.get("text", "")),
                source=str(r.get("source", "")),
                score=float(r.get("score", 0.0)),
                metadata=r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {},
            )
        )

    ## Sort by score desc
    results.sort(key=lambda x: x.score, reverse=True)

    return results