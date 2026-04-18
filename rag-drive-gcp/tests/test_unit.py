'''
__author__ = "Georges Nassopoulos"
__version__ = "1.0.0"
__status__ = "Dev"
__desc__ = "Minimal unit tests for rag-drive-gcp core components."
'''

import numpy as np

from src.core.data_consistency import run_data_consistency
from src.core.data_quality import run_data_quality
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
    assert hasattr(settings, "top_k")

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


def test_chunking_empty_text():
    """
        Ensure chunking handles empty text
    """

    chunks = chunk_text(
        text="",
        source="unit-test",
        chunk_size=500,
        chunk_overlap=100,
    )

    assert chunks == []

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

def test_retrieve_top_k_empty_index():
    """
        Ensure retrieval handles empty index
    """

    index = RAGIndex(
        embeddings=np.empty((0, 2)),
        chunks=[],
    )

    results = retrieve_top_k(
        index=index,
        query="hello",
        top_k=1,
    )

    assert results == []
    
## ============================================================
## DATA CONSISTENCY TESTS (RAG)
## ============================================================
def test_data_consistency_valid_rag():
    """
        Validate correct RAG payload
    """

    data = {
        "query": "hypertension",
        "chunks": ["text 1", "text 2"],
        "embeddings": [0.1] * 128,
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is True
    assert result["errors"] == 0

def test_data_consistency_empty_query():
    """
        Detect empty query
    """

    data = {
        "query": "",
        "embeddings": [0.1] * 128,
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is False

def test_data_consistency_invalid_chunks():
    """
        Detect invalid chunks structure
    """

    data = {
        "query": "test",
        "chunks": "not_a_list",
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is False

def test_data_consistency_invalid_embeddings():
    """
        Detect invalid embeddings
    """

    data = {
        "query": "test",
        "embeddings": ["bad", "data"],
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is False
    
## ============================================================
## DATA QUALITY TESTS
## ============================================================
def test_data_quality_valid_chunks():
    """
        Validate correct chunks
    """

    texts = ["this is a valid chunk"]

    result = run_data_quality(texts=texts)

    assert result["is_valid"] is True

def test_data_quality_empty_chunk():
    """
        Detect empty chunk
    """

    texts = [""]

    result = run_data_quality(texts=texts)

    assert result["is_valid"] is False

def test_data_quality_length_anomaly():
    """
        Detect abnormal chunk length
    """

    texts = ["short", "this is a very very very very very long chunk of text"]

    result = run_data_quality(texts=texts, method="zscore")

    assert any(issue["rule"] == "chunk_length_anomaly" for issue in result["issues"])

def test_data_quality_scoring():
    """
        Ensure scoring is computed
    """

    texts = ["valid chunk"]

    result = run_data_quality(texts=texts)

    assert "score" in result

def test_data_quality_strict_mode():
    """
        Strict mode should raise error
    """

    texts = [""]

    import pytest

    with pytest.raises(Exception):
        run_data_quality(texts=texts, strict=True)