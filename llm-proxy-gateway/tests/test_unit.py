'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unit tests for costing utilities, scan helpers, and lightweight validation paths (no real HTTP calls)."
'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

## ============================================================
## IMPORTS UNDER TEST
## ============================================================
from src.utils.costing_utils import (
    ScanStats,
    cost_usd_for_chat,
    cost_usd_for_embeddings,
    estimate_input_tokens_for_embeddings,
    get_pricing_row,
    iter_char_chunks,
    resolve_model_for_provider,
)
from src.utils.tokeniser_utils import approximate_token_count

## ------------------------------------------------------------
## NOTE
## ------------------------------------------------------------
## scan_* helpers are expected to be moved into src/utils/utils.py
## If you have not moved them yet, adapt imports accordingly
## ------------------------------------------------------------
from src.utils.utils import (
    scan_folder_txt_inputs,
    scan_text_input_for_chat,
    scan_text_input_for_embeddings,
)

## ============================================================
## FIXTURES
## ============================================================
@pytest.fixture()
def models_catalog_min() -> dict[str, Any]:
    """
        Minimal models catalog fixture

        Returns:
            Models catalog dictionary
    """
    
    return {
        "defaults": {
            "openai": "gpt-4o-mini",
            "google": "gemini-1.5-pro",
            "xai": "grok-2",
        }
    }

@pytest.fixture()
def pricing_catalog_layout_a() -> dict[str, Any]:
    """
        Pricing catalog fixture using layout A

        Layout A:
            pricing_catalog[provider][model] = {...}

        Returns:
            Pricing catalog dictionary
    """
    
    return {
        "openai": {
            "gpt-4o-mini": {
                "input_per_1k": 0.001,
                "output_per_1k": 0.002,
                "embedding_per_1k": 0.0002,
            },
            "default": {
                "input_per_1k": 0.01,
                "output_per_1k": 0.02,
                "embedding_per_1k": 0.002,
            },
        }
    }

@pytest.fixture()
def pricing_catalog_layout_b() -> dict[str, Any]:
    """
        Pricing catalog fixture using layout B

        Layout B:
            pricing_catalog["providers"][provider][model] = {...}

        Returns:
            Pricing catalog dictionary
    """
    
    return {
        "providers": {
            "openai": {
                "gpt-4o-mini": {
                    "input_per_1k": 0.001,
                    "output_per_1k": 0.002,
                    "embedding_per_1k": 0.0002,
                }
            }
        }
    }

## ============================================================
## TOKEN HEURISTICS
## ============================================================
def test_approximate_token_count_empty() -> None:
    """
        Ensure token approximation returns 0 on empty input

        Returns:
            None
    """
    
    assert approximate_token_count("") == 0
    assert approximate_token_count("   ") == 0

def test_approximate_token_count_non_empty() -> None:
    """
        Ensure token approximation returns >= 1 for non-empty text

        Returns:
            None
    """
    
    assert approximate_token_count("hello") >= 1

## ============================================================
## CHUNKING
## ============================================================
def test_iter_char_chunks_valid() -> None:
    """
        Validate chunking produces overlapping windows

        Returns:
            None
    """
    
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = iter_char_chunks(text=text, chunk_size=10, chunk_overlap=2)

    ## Expect multiple chunks and non-empty content
    assert len(chunks) >= 2
    assert all(isinstance(c, str) and c.strip() != "" for c in chunks)

def test_iter_char_chunks_reject_invalid_params() -> None:
    """
        Ensure chunking validation rejects invalid params

        Returns:
            None
    """
    
    text = "hello world"

    with pytest.raises(Exception):
        iter_char_chunks(text=text, chunk_size=0, chunk_overlap=0)

    with pytest.raises(Exception):
        iter_char_chunks(text=text, chunk_size=10, chunk_overlap=-1)

    with pytest.raises(Exception):
        iter_char_chunks(text=text, chunk_size=10, chunk_overlap=10)

def test_estimate_input_tokens_for_embeddings_positive() -> None:
    """
        Ensure embeddings token estimation returns a non-negative integer

        Returns:
            None
    """
    
    text = "This is a long enough text to be chunked and token-counted"
    tokens = estimate_input_tokens_for_embeddings(text=text, chunk_size=20, chunk_overlap=5)

    assert isinstance(tokens, int)
    assert tokens >= 0

## ============================================================
## PRICING RESOLUTION
## ============================================================
def test_resolve_model_for_provider_requested_overrides(models_catalog_min: dict[str, Any]) -> None:
    """
        Requested model must override defaults

        Args:
            models_catalog_min: Fixture

        Returns:
            None
    """
    
    model = resolve_model_for_provider(
        provider="openai",
        requested_model="my-model",
        models_catalog=models_catalog_min,
    )
    
    assert model == "my-model"

def test_resolve_model_for_provider_uses_default(models_catalog_min: dict[str, Any]) -> None:
    """
        If requested model is empty, defaults must be used when available

        Args:
            models_catalog_min: Fixture

        Returns:
            None
    """
    
    model = resolve_model_for_provider(
        provider="openai",
        requested_model=None,
        models_catalog=models_catalog_min,
    )
    
    assert model == "gpt-4o-mini"

def test_get_pricing_row_layout_a(pricing_catalog_layout_a: dict[str, Any]) -> None:
    """
        Validate pricing lookup supports layout A

        Args:
            pricing_catalog_layout_a: Fixture

        Returns:
            None
    """
    
    row = get_pricing_row(
        provider="openai",
        model="gpt-4o-mini",
        pricing_catalog=pricing_catalog_layout_a,
    )

    assert row.input_per_1k == 0.001
    assert row.output_per_1k == 0.002
    assert row.embedding_per_1k == 0.0002

def test_get_pricing_row_layout_b(pricing_catalog_layout_b: dict[str, Any]) -> None:
    """
        Validate pricing lookup supports layout B

        Args:
            pricing_catalog_layout_b: Fixture

        Returns:
            None
    """
    
    row = get_pricing_row(
        provider="openai",
        model="gpt-4o-mini",
        pricing_catalog=pricing_catalog_layout_b,
    )

    assert row.input_per_1k == 0.001
    assert row.output_per_1k == 0.002
    assert row.embedding_per_1k == 0.0002

def test_get_pricing_row_fallback_default(pricing_catalog_layout_a: dict[str, Any]) -> None:
    """
        If model is missing, provider default pricing can be used

        Args:
            pricing_catalog_layout_a: Fixture

        Returns:
            None
    """
    
    row = get_pricing_row(
        provider="openai",
        model="missing-model",
        pricing_catalog=pricing_catalog_layout_a,
    )

    assert row.input_per_1k == 0.01
    assert row.output_per_1k == 0.02
    assert row.embedding_per_1k == 0.002

## ============================================================
## COST MATH
## ============================================================
def test_cost_usd_for_chat_simple(pricing_catalog_layout_a: dict[str, Any]) -> None:
    """
        Validate chat cost computation returns expected float

        Args:
            pricing_catalog_layout_a: Fixture

        Returns:
            None
    """
    
    row = get_pricing_row(
        provider="openai",
        model="gpt-4o-mini",
        pricing_catalog=pricing_catalog_layout_a,
    )

    cost = cost_usd_for_chat(pricing=row, input_tokens=1000, output_tokens=500)

    ## 1000 tokens input = 1 unit * 0.001
    ## 500 tokens output = 0.5 unit * 0.002
    assert cost == pytest.approx(0.001 + 0.001, rel=1e-9)

def test_cost_usd_for_embeddings_simple(pricing_catalog_layout_a: dict[str, Any]) -> None:
    """
        Validate embeddings cost computation returns expected float

        Args:
            pricing_catalog_layout_a: Fixture

        Returns:
            None
    """
    
    row = get_pricing_row(
        provider="openai",
        model="gpt-4o-mini",
        pricing_catalog=pricing_catalog_layout_a,
    )

    cost = cost_usd_for_embeddings(pricing=row, input_tokens=2000)

    ## 2000 tokens = 2 units * 0.0002
    assert cost == pytest.approx(0.0004, rel=1e-9)

## ============================================================
## SCAN HELPERS
## ============================================================
def test_scan_text_input_for_chat() -> None:
    """
        Validate scan stats for chat text input

        Returns:
            None
    """
    
    scan = scan_text_input_for_chat("hello world")

    assert isinstance(scan, ScanStats)
    assert scan.n_files == 0
    assert scan.n_chars == len("hello world")
    assert scan.input_tokens >= 1

def test_scan_text_input_for_embeddings() -> None:
    """
        Validate scan stats for embeddings text input

        Returns:
            None
    """
    
    scan = scan_text_input_for_embeddings(
        text="hello world embeddings input",
        chunk_size=10,
        chunk_overlap=2,
    )

    assert isinstance(scan, ScanStats)
    assert scan.n_files == 0
    assert scan.n_chars == len("hello world embeddings input")
    assert scan.input_tokens >= 1

def test_scan_folder_txt_inputs_chat(tmp_path: Path) -> None:
    """
        Validate folder scan in chat mode

        Args:
            tmp_path: Pytest temp folder fixture

        Returns:
            None
    """
    
    ## Create small txt inputs
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("world", encoding="utf-8")

    scan, per_file = scan_folder_txt_inputs(
        folder=tmp_path,
        recursive=False,
        max_chars_per_file=200_000,
        mode="chat",
        chunk_size=1000,
        chunk_overlap=200,
    )

    assert isinstance(scan, ScanStats)
    assert scan.n_files == 2
    assert scan.n_chars > 0
    assert scan.input_tokens > 0

    ## Per-file breakdown should match
    assert len(per_file) == 2
    assert all("file_path" in r for r in per_file)
    assert all("approx_tokens" in r for r in per_file)

def test_scan_folder_txt_inputs_embeddings(tmp_path: Path) -> None:
    """
        Validate folder scan in embeddings mode

        Args:
            tmp_path: Pytest temp folder fixture

        Returns:
            None
    """
    ## Create txt inputs
    (tmp_path / "a.txt").write_text("hello embeddings" * 50, encoding="utf-8")
    (tmp_path / "nested").mkdir(parents=True, exist_ok=True)
    (tmp_path / "nested" / "b.txt").write_text("more embeddings" * 50, encoding="utf-8")

    scan, per_file = scan_folder_txt_inputs(
        folder=tmp_path,
        recursive=True,
        max_chars_per_file=200_000,
        mode="embeddings",
        chunk_size=50,
        chunk_overlap=10,
    )

    assert scan.n_files == 2
    assert scan.input_tokens > 0
    assert len(per_file) == 2

def test_scan_folder_txt_inputs_invalid_mode(tmp_path: Path) -> None:
    """
        Validate invalid mode is rejected

        Args:
            tmp_path: Pytest temp folder fixture

        Returns:
            None
    """
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")

    with pytest.raises(Exception):
        scan_folder_txt_inputs(
            folder=tmp_path,
            recursive=False,
            max_chars_per_file=200_000,
            mode="invalid",
            chunk_size=1000,
            chunk_overlap=200,
        )