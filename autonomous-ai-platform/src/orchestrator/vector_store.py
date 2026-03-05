'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Vector store implementations and factory for retrieval (FAISS or Chroma) with disk persistence and structured errors."
'''

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.core.config import config
from src.core.errors import DependencyError, RetrievalError, StorageError
from src.utils.logging_utils import get_logger
from src.utils.io_utils import ensure_directory
from src.utils.env_utils import (
    _resolve_index_name,
    _resolve_vector_store_dir,
    _resolve_vector_store_backend,
)

logger = get_logger(__name__)

## ============================================================
## DATA MODELS
## ============================================================
class ChunkLike:
    """
        Protocol-like minimal chunk interface

        Args:
            chunk_id: Stable id
            text: Chunk content
            source: Source id
            start_char: Start offset
            end_char: End offset
    """

    chunk_id: str
    text: str
    source: str
    start_char: int
    end_char: int


class SearchResultLike:
    """
        Protocol-like minimal search result interface
    """

## ============================================================
## FAISS STORE
## ============================================================
class FaissStore:
    """
        Minimal FAISS vector store with disk persistence

        Design:
            - Expects normalized vectors for cosine similarity using IndexFlatIP
            - Persists index and metadata side-by-side in store_dir
    """

    def __init__(self, store_dir: Path, index_name: str) -> None:
        ## Resolve and create persistence directory
        self.store_dir = Path(store_dir).expanduser().resolve()
        self.index_name = index_name
        ensure_directory(self.store_dir)

        ## Persisted files
        self.index_path = self.store_dir / f"{self.index_name}.faiss"
        self.meta_path = self.store_dir / f"{self.index_name}.json"

        ## In-memory structures
        self._index: Optional[Any] = None
        self._meta: Dict[str, Dict[str, Any]] = {}

        ## Load if existing
        self._load_if_exists()

    def _import_faiss(self) -> Any:
        """
            Import faiss lazily

            Returns:
                faiss module
        """

        try:
            import faiss  # type: ignore

            return faiss
        except Exception as exc:
            raise DependencyError(
                message="FAISS dependency not available",
                error_code="dependency_error",
                details={"pip_package": "faiss-cpu or faiss-gpu"},
                origin="retrieval_store",
                cause=exc,
                http_status=500,
                is_retryable=False,
            ) from exc

    def _load_if_exists(self) -> None:
        """
            Load index and metadata if they exist

            Returns:
                None
        """

        faiss = self._import_faiss()

        ## Load metadata first
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise StorageError(
                    message="Failed to load FAISS metadata",
                    error_code="storage_error",
                    details={"meta_path": str(self.meta_path)},
                    origin="retrieval_store",
                    cause=exc,
                    http_status=500,
                    is_retryable=True,
                ) from exc

        ## Load index
        if self.index_path.exists():
            try:
                self._index = faiss.read_index(str(self.index_path))
            except Exception as exc:
                raise StorageError(
                    message="Failed to load FAISS index",
                    error_code="storage_error",
                    details={"index_path": str(self.index_path)},
                    origin="retrieval_store",
                    cause=exc,
                    http_status=500,
                    is_retryable=True,
                ) from exc

    def _persist(self) -> None:
        """
            Persist index and metadata

            Returns:
                None
        """

        faiss = self._import_faiss()

        ## Persist only if index exists
        if self._index is None:
            return

        ## Persist index
        try:
            faiss.write_index(self._index, str(self.index_path))
        except Exception as exc:
            raise StorageError(
                message="Failed to persist FAISS index",
                error_code="storage_error",
                details={"index_path": str(self.index_path)},
                origin="retrieval_store",
                cause=exc,
                http_status=500,
                is_retryable=True,
            ) from exc

        ## Persist metadata
        try:
            self.meta_path.write_text(
                json.dumps(self._meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            raise StorageError(
                message="Failed to persist FAISS metadata",
                error_code="storage_error",
                details={"meta_path": str(self.meta_path)},
                origin="retrieval_store",
                cause=exc,
                http_status=500,
                is_retryable=True,
            ) from exc

    def add(self, vectors: np.ndarray, chunks: Sequence[ChunkLike]) -> int:
        """
            Add vectors and chunk metadata into store

            Args:
                vectors: Numpy array (n, d) float32
                chunks: Chunk objects

            Returns:
                Insert count
        """

        faiss = self._import_faiss()

        ## Validate shapes
        if vectors.ndim != 2:
            raise RetrievalError(
                message="Vectors must be 2D numpy array",
                error_code="retrieval_error",
                details={"shape": list(vectors.shape)},
                origin="retrieval_store",
                cause=None,
                http_status=400,
                is_retryable=False,
            )

        if len(chunks) != int(vectors.shape[0]):
            raise RetrievalError(
                message="Vectors count does not match chunks count",
                error_code="retrieval_error",
                details={"vectors": int(vectors.shape[0]), "chunks": int(len(chunks))},
                origin="retrieval_store",
                cause=None,
                http_status=400,
                is_retryable=False,
            )

        ## Init index if needed
        if self._index is None:
            dim = int(vectors.shape[1])
            self._index = faiss.IndexFlatIP(dim)

        ## Add vectors
        try:
            self._index.add(vectors.astype(np.float32))
        except Exception as exc:
            raise RetrievalError(
                message="Failed to add vectors to FAISS index",
                error_code="retrieval_error",
                details={"count": int(vectors.shape[0])},
                origin="retrieval_store",
                cause=exc,
                http_status=500,
                is_retryable=True,
            ) from exc

        ## Persist chunk metadata keyed by internal integer id
        base_id = len(self._meta)
        for i, ch in enumerate(chunks):
            internal_id = str(base_id + i)
            self._meta[internal_id] = {
                "chunk_id": getattr(ch, "chunk_id", ""),
                "text": getattr(ch, "text", ""),
                "source": getattr(ch, "source", ""),
                "start_char": getattr(ch, "start_char", 0),
                "end_char": getattr(ch, "end_char", 0),
            }

        ## Persist to disk
        self._persist()

        return int(vectors.shape[0])

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """
            Search for nearest vectors

            Args:
                query_vector: Numpy array (d,) float32
                top_k: Number of results

            Returns:
                List of dict results
        """

        ## Return empty if not ready
        if self._index is None or not self._meta:
            return []

        ## Validate query vector
        if query_vector.ndim != 1:
            raise RetrievalError(
                message="Query vector must be 1D numpy array",
                error_code="retrieval_error",
                details={"shape": list(query_vector.shape)},
                origin="retrieval_store",
                cause=None,
                http_status=400,
                is_retryable=False,
            )

        ## Run FAISS search
        try:
            q = query_vector.astype(np.float32).reshape(1, -1)
            scores, ids = self._index.search(q, int(top_k))
        except Exception as exc:
            raise RetrievalError(
                message="FAISS search failed",
                error_code="retrieval_error",
                details={"top_k": int(top_k)},
                origin="retrieval_store",
                cause=exc,
                http_status=500,
                is_retryable=True,
            ) from exc

        ## Build results list
        results: List[Dict[str, Any]] = []
        for i in range(min(int(top_k), int(ids.shape[1]))):
            internal_id = int(ids[0, i])
            score = float(scores[0, i])

            meta = self._meta.get(str(internal_id))
            if not meta:
                continue

            results.append(
                {
                    "chunk_id": str(meta.get("chunk_id", "")),
                    "text": str(meta.get("text", "")),
                    "source": str(meta.get("source", "")),
                    "score": score,
                    "metadata": meta,
                }
            )

        return results

## ============================================================
## CHROMA STORE
## ============================================================
class ChromaStore:
    """
        Minimal Chroma vector store (persistent)

        Notes:
            - Stores documents and metadata
            - Uses embeddings passed by caller
    """

    def __init__(self, store_dir: Path, index_name: str) -> None:
        self.store_dir = Path(store_dir).expanduser().resolve()
        self.index_name = index_name
        ensure_directory(self.store_dir)

        self._collection: Optional[Any] = None
        self._init_collection()

    def _init_collection(self) -> None:
        """
            Initialize Chroma collection

            Returns:
                None
        """

        try:
            import chromadb  # type: ignore
        except Exception as exc:
            raise DependencyError(
                message="Chroma dependency not available",
                error_code="dependency_error",
                details={"pip_package": "chromadb"},
                origin="retrieval_store",
                cause=exc,
                http_status=500,
                is_retryable=False,
            ) from exc

        try:
            client = chromadb.PersistentClient(path=str(self.store_dir))
            self._collection = client.get_or_create_collection(name=self.index_name)
        except Exception as exc:
            raise StorageError(
                message="Failed to initialize Chroma collection",
                error_code="storage_error",
                details={"store_dir": str(self.store_dir), "index_name": self.index_name},
                origin="retrieval_store",
                cause=exc,
                http_status=500,
                is_retryable=True,
            ) from exc

    def add(self, vectors: np.ndarray, chunks: Sequence[ChunkLike]) -> int:
        """
            Add vectors and docs into Chroma

            Args:
                vectors: Numpy array (n, d)
                chunks: Chunk objects

            Returns:
                Insert count
        """

        if self._collection is None:
            return 0

        if vectors.ndim != 2:
            raise RetrievalError(
                message="Vectors must be 2D numpy array",
                error_code="retrieval_error",
                details={"shape": list(vectors.shape)},
                origin="retrieval_store",
                cause=None,
                http_status=400,
                is_retryable=False,
            )

        if len(chunks) != int(vectors.shape[0]):
            raise RetrievalError(
                message="Vectors count does not match chunks count",
                error_code="retrieval_error",
                details={"vectors": int(vectors.shape[0]), "chunks": int(len(chunks))},
                origin="retrieval_store",
                cause=None,
                http_status=400,
                is_retryable=False,
            )

        ## Build Chroma payloads
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        embeddings = vectors.astype(np.float32).tolist()

        for ch in chunks:
            ids.append(str(getattr(ch, "chunk_id", "")))
            documents.append(str(getattr(ch, "text", "")))
            metadatas.append(
                {
                    "source": str(getattr(ch, "source", "")),
                    "start_char": int(getattr(ch, "start_char", 0)),
                    "end_char": int(getattr(ch, "end_char", 0)),
                }
            )

        ## Add to collection
        try:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            return int(len(ids))
        except Exception as exc:
            raise RetrievalError(
                message="Failed to add vectors to Chroma",
                error_code="retrieval_error",
                details={"count": int(len(ids))},
                origin="retrieval_store",
                cause=exc,
                http_status=500,
                is_retryable=True,
            ) from exc

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """
            Search in Chroma

            Args:
                query_vector: Query vector (d,)
                top_k: Top k

            Returns:
                List of dict results
        """

        if self._collection is None:
            return []

        if query_vector.ndim != 1:
            raise RetrievalError(
                message="Query vector must be 1D numpy array",
                error_code="retrieval_error",
                details={"shape": list(query_vector.shape)},
                origin="retrieval_store",
                cause=None,
                http_status=400,
                is_retryable=False,
            )

        ## Query collection
        try:
            res = self._collection.query(
                query_embeddings=[query_vector.astype(np.float32).tolist()],
                n_results=int(top_k),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise RetrievalError(
                message="Chroma search failed",
                error_code="retrieval_error",
                details={"top_k": int(top_k)},
                origin="retrieval_store",
                cause=exc,
                http_status=500,
                is_retryable=True,
            ) from exc

        ## Extract results arrays
        docs = res.get("documents", [[]])
        metas = res.get("metadatas", [[]])
        dists = res.get("distances", [[]])
        ids = res.get("ids", [[]])

        results: List[Dict[str, Any]] = []
        for i in range(min(int(top_k), len(ids[0]) if ids and ids[0] else 0)):
            dist = float(dists[0][i]) if dists and dists[0] else 0.0
            score = 1.0 / (1.0 + dist)

            md = metas[0][i] if metas and metas[0] else {}
            results.append(
                {
                    "chunk_id": str(ids[0][i]),
                    "text": str(docs[0][i]) if docs and docs[0] else "",
                    "source": str(md.get("source", "")),
                    "score": score,
                    "metadata": md,
                }
            )

        return results

## ============================================================
## FACTORY
## ============================================================
def get_vector_store() -> Any:
    """
        Vector store factory

        Env variables:
            VECTOR_STORE_BACKEND
            VECTOR_STORE_DIR
            VECTOR_INDEX_NAME

        Returns:
            Store instance
    """

    ## Resolve backend and paths
    backend = _resolve_vector_store_backend()
    store_dir = _resolve_vector_store_dir()
    index_name = _resolve_index_name()

    ## Ensure persistence directory exists
    ensure_directory(store_dir)

    ## Create store instance
    if backend == "chroma":
        return ChromaStore(store_dir=store_dir, index_name=index_name)

    return FaissStore(store_dir=store_dir, index_name=index_name)

def get_vector_store_info() -> Dict[str, Any]:
    """
        Expose vector store info for UI/debug

        Returns:
            Dict info
    """

    return {
        "backend": _resolve_vector_store_backend(),
        "store_dir": str(_resolve_vector_store_dir()),
        "index_name": _resolve_index_name(),
    }