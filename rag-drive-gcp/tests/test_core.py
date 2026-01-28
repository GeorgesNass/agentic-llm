'''
__author__ = "Georges Nassopoulos"
__version__ = "1.0.0"
__status__ = "Dev"
__desc__ = "Minimal unit tests for rag-drive-gcp core components."
'''

import numpy as np

from src.core.rag import Chunk, RAGIndex, chunk_text, retrieve_top_k
from src.io.gcs import build_gcs_object_path
from src.model.settings import get_settings

## ============================================================
## SETTINGS
## ============================================================
def test_settings_can_be_loaded():
    """
        Ensure application settings load correctly
    """
    
    settings = get_settings()
    assert settings is not None

## ============================================================
## CHUNKING
## ============================================================
def test_chunking_produces_multiple_chunks():
    """
        Ensure chunking splits long text into multiple chunks
    """
    
    text = "A" * 2000

    chunks = chunk_text(
        text=text,
        source="unit-test",
        chunk_size=500,
        chunk_overlap=100,
    )

    assert len(chunks) > 1

## ============================================================
## GCS PATH BUILDER
## ============================================================
def test_gcs_object_path_builder():
    """
        Ensure GCS object path builder works as expected
    """
    
    path = build_gcs_object_path("texts", "file.txt")
    assert path == "texts/file.txt"

## ============================================================
## RAG RETRIEVAL
## ============================================================
def test_retrieve_top_k_returns_expected_chunks():
    """
        Ensure top-k retrieval returns expected results
    """
    
    chunks = [
        Chunk(text="hello world", source="test", index=0),
    ]

    embeddings = np.array([[1.0, 0.0]])
    index = RAGIndex(
        embeddings=embeddings,
        chunks=chunks,
    )

    results = retrieve_top_k(
        index=index,
        query="hello",
        top_k=1,
    )

    assert len(results) == 1