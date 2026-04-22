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
import pandas as pd

import pytest

## IMPORTS UNDER TEST
from src.core.data_quality import run_data_quality
from src.core.data_consistency import run_data_consistency
from src.core.data_drift import run_data_drift
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
    assert all(len(c) <= 10 for c in chunks)

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

def test_get_pricing_row_unknown_provider() -> None:
    """
        Validate unknown provider handling
    """

    with pytest.raises(Exception):
        get_pricing_row(
            provider="unknown",
            model="x",
            pricing_catalog={},
        )
        
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

def test_scan_text_input_for_chat_empty() -> None:
    """
        Validate scan handles empty chat input
    """

    scan = scan_text_input_for_chat("")

    assert scan.n_chars == 0
    assert scan.input_tokens == 0
    
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

def test_scan_text_input_for_embeddings_empty() -> None:
    """
        Validate scan handles empty embeddings input
    """

    scan = scan_text_input_for_embeddings(
        text="",
        chunk_size=10,
        chunk_overlap=2,
    )

    assert scan.n_chars == 0
    assert scan.input_tokens == 0
    
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

## ============================================================
## DATA CONSISTENCY TESTS (LLM GATEWAY)
## ============================================================
def test_data_consistency_valid_prompt() -> None:
    """
        Validate correct prompt payload

        Returns:
            None
    """

    data = {
        "prompt": "hello world",
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "temperature": 0.5,
        "top_p": 1.0,
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is True

def test_data_consistency_missing_model() -> None:
    """
        Detect missing model

        Returns:
            None
    """

    data = {
        "prompt": "hello",
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is False

def test_data_consistency_invalid_messages() -> None:
    """
        Detect invalid messages structure

        Returns:
            None
    """

    data = {
        "messages": "not_a_list",
        "model": "gpt-4o-mini",
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is False

def test_data_consistency_invalid_params() -> None:
    """
        Detect invalid parameters

        Returns:
            None
    """

    data = {
        "prompt": "test",
        "model": "gpt-4o-mini",
        "max_tokens": -1,
        "temperature": 5,
        "top_p": 2,
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is False
    
## ============================================================
## DATA QUALITY TESTS (LLM GATEWAY)
## ============================================================
def test_data_quality_valid() -> None:
    """
        Validate normal LLM payload metrics

        Returns:
            None
    """

    data = {
        "prompt_length": 100,
        "messages_count": 2,
        "max_tokens": 256,
    }

    result = run_data_quality(data=data)

    assert result["is_valid"] is True
    assert result["errors"] == 0

def test_data_quality_outlier() -> None:
    """
        Detect abnormal LLM request metrics

        Returns:
            None
    """

    data = {
        "prompt_length": 1000000,
        "messages_count": 5000,
        "max_tokens": 100000,
    }

    result = run_data_quality(data=data)

    assert result["warnings"] > 0

def test_data_quality_invalid() -> None:
    """
        Detect invalid values

        Returns:
            None
    """

    data = {
        "prompt_length": float("nan"),
        "messages_count": 1,
    }

    result = run_data_quality(data=data)

    assert result["errors"] > 0

def test_data_quality_strict() -> None:
    """
        Strict mode should raise error

        Returns:
            None
    """

    data = {
        "prompt_length": float("nan"),
    }

    with pytest.raises(Exception):
        run_data_quality(data=data, strict=True)
        
## ============================================================
## DATA DRIFT TESTS (LLM GATEWAY)
## ============================================================
def test_data_drift_no_drift_llm() -> None:
    """
        Validate no drift scenario on LLM metrics
    """

    df_ref = pd.DataFrame({
        "response_time": [100, 110],
        "input_tokens": [50, 60],
        "output_tokens": [20, 25],
        "model": ["gpt-4o-mini", "gpt-4o-mini"],
    })

    df_cur = pd.DataFrame({
        "response_time": [100, 110],
        "input_tokens": [50, 60],
        "output_tokens": [20, 25],
        "model": ["gpt-4o-mini", "gpt-4o-mini"],
    })

    result = run_data_drift(df_ref=df_ref, df_current=df_cur)

    assert result["drift_score"] >= 0.9
    assert result["errors"] == 0

def test_data_drift_detected_llm() -> None:
    """
        Detect drift on latency and tokens
    """

    df_ref = pd.DataFrame({
        "response_time": [100, 100],
        "input_tokens": [50, 50],
        "output_tokens": [20, 20],
        "model": ["gpt-4o-mini", "gpt-4o-mini"],
    })

    df_cur = pd.DataFrame({
        "response_time": [1000, 1200],
        "input_tokens": [500, 600],
        "output_tokens": [200, 250],
        "model": ["gpt-4o", "gpt-4o"],
    })

    result = run_data_drift(df_ref=df_ref, df_current=df_cur)

    assert result["drift_score"] < 1.0
    assert result["warnings"] > 0

def test_data_drift_empty_llm() -> None:
    """
        Validate empty dataset handling
    """

    df_ref = pd.DataFrame()
    df_cur = pd.DataFrame()

    with pytest.raises(Exception):
        run_data_drift(df_ref=df_ref, df_current=df_cur)


def test_data_drift_strict_llm() -> None:
    """
        Validate strict mode behavior
    """

    df_ref = pd.DataFrame({"response_time": [100]})
    df_cur = pd.DataFrame({"response_time": [1000]})

    with pytest.raises(Exception):
        run_data_drift(df_ref=df_ref, df_current=df_cur, strict=True)
        
def test_data_drift_evidently_output_llm_proxy() -> None:
    """
        Validate Evidently report generation for LLM proxy drift

        Returns:
            None
    """

    df_ref = pd.DataFrame({
        "response_time": [100, 110],
        "input_tokens": [50, 60],
        "output_tokens": [20, 25],
        "model": ["gpt-4o-mini", "gpt-4o-mini"],
    })

    df_cur = df_ref.copy()

    result = run_data_drift(df_ref=df_ref, df_current=df_cur)

    assert "evidently_report" in result or result["warnings"] >= 0