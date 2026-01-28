'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "RAG core logic for rag-drive-gcp: chunking, indexing, and retrieval using Vertex embeddings."
'''

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.core.vertex import build_rag_prompt, embed_texts, generate_text
from src.model.settings import RagAnswer, get_settings
from src.utils.logging_utils import get_logger
from src.utils.utils import ensure_directories

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("rag")


## ============================================================
## DATA CLASSES
## ============================================================
@dataclass
class Chunk:
    """
        Text chunk container

        Attributes:
            text (str): Chunk text
            source (str): Source identifier (file name or URI)
            index (int): Chunk index in source
    """

    text: str
    source: str
    index: int

@dataclass
class RAGIndex:
    """
        In-memory RAG index

        Attributes:
            embeddings (np.ndarray): Embedding matrix (n_chunks x dim)
            chunks (List[Chunk]): Corresponding text chunks
    """

    embeddings: np.ndarray
    chunks: List[Chunk]

## ============================================================
## CHUNKING
## ============================================================
def chunk_text(
    text: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Chunk]:
    """
        Split text into overlapping chunks

        Args:
            text (str): Full text to split
            source (str): Source identifier
            chunk_size (int): Max characters per chunk
            chunk_overlap (int): Overlap between chunks

        Returns:
            List[Chunk]: List of text chunks
    """
    
    chunks: List[Chunk] = []
    start = 0
    idx = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk_text_value = text[start:end].strip()

        if chunk_text_value:
            chunks.append(
                Chunk(
                    text=chunk_text_value,
                    source=source,
                    index=idx,
                )
            )
            idx += 1

        start = end - chunk_overlap
        if start < 0:
            start = 0

    logger.debug(f"Generated {len(chunks)} chunk(s) from source={source}")
    return chunks

## ============================================================
## INDEX BUILDING
## ============================================================
def build_rag_index_from_texts(
    texts: List[Tuple[str, str]],
) -> RAGIndex:
    """
        Build an in-memory RAG index from raw texts

        Args:
            texts (List[Tuple[str, str]]): List of (source, text) pairs

        Returns:
            RAGIndex: Built RAG index
    """
    
    settings = get_settings()

    all_chunks: List[Chunk] = []
    for source, text in texts:
        all_chunks.extend(
            chunk_text(
                text=text,
                source=source,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        )

    if not all_chunks:
        raise ValueError("No chunks generated from input texts.")

    logger.info(f"Total chunks to embed: {len(all_chunks)}")

    embeddings = embed_texts([c.text for c in all_chunks])
    emb_matrix = np.array(embeddings, dtype=float)

    return RAGIndex(
        embeddings=emb_matrix,
        chunks=all_chunks,
    )

## ============================================================
## RETRIEVAL
## ============================================================
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
        Compute cosine similarity between matrix a and vector b

        Args:
            a (np.ndarray): Matrix (n x d)
            b (np.ndarray): Vector (d,)

        Returns:
            np.ndarray: Similarity scores (n,)
    """
    
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)

def retrieve_top_k(
    index: RAGIndex,
    query: str,
    top_k: int,
) -> List[Chunk]:
    """
        Retrieve top-K most relevant chunks for a query

        Args:
            index (RAGIndex): RAG index
            query (str): User query
            top_k (int): Number of chunks to retrieve

        Returns:
            List[Chunk]: Top-K retrieved chunks
    """
    
    query_embedding = embed_texts([query])[0]
    scores = _cosine_similarity(index.embeddings, np.array(query_embedding))

    top_indices = np.argsort(scores)[::-1][:top_k]
    retrieved = [index.chunks[i] for i in top_indices]

    logger.info(f"Retrieved {len(retrieved)} chunk(s) for query.")
    return retrieved

## ============================================================
## END-TO-END QUERY
## ============================================================
def run_rag_query(
    index: RAGIndex,
    question: str,
) -> RagAnswer:
    """
        Run a full RAG query: retrieve context and generate answer

        Args:
            index (RAGIndex): RAG index
            question (str): User question

        Returns:
            RagAnswer: Generated answer with sources
    """
    
    settings = get_settings()

    retrieved_chunks = retrieve_top_k(
        index=index,
        query=question,
        top_k=settings.top_k,
    )

    context_texts = [c.text for c in retrieved_chunks]
    prompt = build_rag_prompt(question=question, context_chunks=context_texts)

    generation = generate_text(prompt)

    sources = [
        {
            "source": c.source,
            "chunk_index": c.index,
        }
        for c in retrieved_chunks
    ]

    return RagAnswer(
        question=question,
        answer=generation.response_text,
        sources=sources,
    )

## ============================================================
## SERIALIZATION HELPERS (OPTIONAL)
## ============================================================
def save_rag_index(index: RAGIndex, output_dir: Path) -> None:
    """
        Save RAG index locally (embeddings + metadata)

        Args:
            index (RAGIndex): RAG index to save
            output_dir (Path): Directory to save artifacts
    """
    
    ensure_directories(output_dir)

    np.save(output_dir / "embeddings.npy", index.embeddings)

    meta = [
        {"text": c.text, "source": c.source, "index": c.index}
        for c in index.chunks
    ]
    with open(output_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"RAG index saved to {output_dir}")

def load_rag_index(input_dir: Path) -> RAGIndex:
    """
        Load a RAG index from local artifacts

        Args:
            input_dir (Path): Directory containing embeddings.npy and chunks.json

        Returns:
            RAGIndex: Loaded RAG index
    """
    
    embeddings = np.load(input_dir / "embeddings.npy")

    with open(input_dir / "chunks.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    chunks = [
        Chunk(text=m["text"], source=m["source"], index=m["index"])
        for m in meta
    ]

    logger.info(f"RAG index loaded from {input_dir}")
    return RAGIndex(embeddings=embeddings, chunks=chunks)