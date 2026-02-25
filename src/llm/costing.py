'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Cost simulation API: load catalogs, scan inputs (text/folder), and compute provider/model cost estimates."
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from src.core.errors import (
    log_and_raise_missing_path,
    log_and_raise_validation_error,
)
from src.utils.costing_utils import (
    ScanStats,
    cost_usd_for_chat,
    cost_usd_for_embeddings,
    estimate_input_tokens_for_embeddings,
    get_pricing_row,
    resolve_model_for_provider,
)
from src.utils.logging_utils import get_logger, log_execution_time_and_path
from src.utils.tokeniser_utils import approximate_token_count
from src.utils.utils import (
    list_txt_files,
    resolve_path,
    safe_load_json,
    safe_read_text,
)

LOGGER = get_logger(__name__)

## ============================================================
## CATALOG LOADING
## ============================================================
@log_execution_time_and_path
def load_catalogs(
    models_catalog_path: str | Path,
    pricing_catalog_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
        Load models and pricing catalogs from JSON files

        High-level workflow:
            1) Resolve paths to absolute paths
            2) Validate both files exist
            3) Load JSON safely with utf-8
            4) Return both dictionaries

        Args:
            models_catalog_path: Path to models_catalog.json
            pricing_catalog_path: Path to pricing_catalog.json

        Returns:
            Tuple(models_catalog, pricing_catalog)
    """
    
    ## Resolve and validate paths
    models_path = resolve_path(models_catalog_path)
    pricing_path = resolve_path(pricing_catalog_path)

    if not models_path.exists():
        log_and_raise_missing_path(models_path, context="models_catalog")

    if not pricing_path.exists():
        log_and_raise_missing_path(pricing_path, context="pricing_catalog")

    ## Load JSON catalogs
    models_catalog = safe_load_json(models_path)
    pricing_catalog = safe_load_json(pricing_path)

    return models_catalog, pricing_catalog

## ============================================================
## INPUT SCANNING
## ============================================================
def _scan_text_input_for_chat(text: str) -> ScanStats:
    """
        Scan a direct text input for chat mode

        High-level workflow:
            1) Count characters
            2) Estimate input tokens using heuristic tokenization
            3) Return ScanStats

        Args:
            text: Prompt text

        Returns:
            ScanStats
    """
    
    ## Estimate tokens from prompt text
    n_chars = len(text)
    input_tokens = approximate_token_count(text)

    return ScanStats(
        n_files=0,
        n_chars=n_chars,
        input_tokens=input_tokens,
    )

def _scan_text_input_for_embeddings(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> ScanStats:
    """
        Scan a direct text input for embeddings mode

        High-level workflow:
            1) Count characters
            2) Chunk the text to approximate embedding segmentation
            3) Sum token counts across chunks

        Args:
            text: Input text
            chunk_size: Chunk size in characters
            chunk_overlap: Chunk overlap in characters

        Returns:
            ScanStats
    """
    
    ## Estimate tokens using chunking heuristic
    n_chars = len(text)
    input_tokens = estimate_input_tokens_for_embeddings(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return ScanStats(
        n_files=0,
        n_chars=n_chars,
        input_tokens=input_tokens,
    )

def _scan_folder_txt_inputs(
    folder: str | Path,
    recursive: bool,
    max_chars_per_file: int,
    mode: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[ScanStats, list[dict[str, Any]]]:
    """
        Scan a folder for .txt files and estimate tokens

        High-level workflow:
            1) Resolve and validate folder path
            2) List .txt files (recursive optional)
            3) For each file:
                - Read up to max_chars_per_file
                - Estimate tokens based on mode
                - Append a per-file summary row
            4) Return global stats + per-file rows

        Args:
            folder: Folder path
            recursive: Recursive scan flag
            max_chars_per_file: Max chars to read per file
            mode: "chat" or "embeddings"
            chunk_size: Embeddings chunk size
            chunk_overlap: Embeddings chunk overlap

        Returns:
            Tuple(ScanStats, per_file_rows)
    """
    
    folder_path = resolve_path(folder)
    if not folder_path.exists():
        log_and_raise_missing_path(folder_path, context="_scan_folder_txt_inputs")

    ## List files
    files = list_txt_files(folder_path, recursive=recursive)

    total_chars = 0
    total_tokens = 0
    per_file: list[dict[str, Any]] = []

    ## Scan each file
    for fp in files:
        ## Read with a hard cap to avoid huge files exploding runtime
        text = safe_read_text(fp, max_chars=max_chars_per_file)
        n_chars = len(text)

        ## Compute tokens depending on mode
        if mode == "chat":
            approx_tokens = approximate_token_count(text)
        elif mode == "embeddings":
            approx_tokens = estimate_input_tokens_for_embeddings(
                text=text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            log_and_raise_validation_error(
                reason=f"Invalid mode={mode}",
                context="_scan_folder_txt_inputs",
            )

        total_chars += n_chars
        total_tokens += approx_tokens

        ## Build per-file row
        per_file.append(
            {
                "file_path": str(fp),
                "n_chars": int(n_chars),
                "approx_tokens": int(approx_tokens),
            }
        )

    return (
        ScanStats(
            n_files=len(files),
            n_chars=total_chars,
            input_tokens=total_tokens,
        ),
        per_file,
    )

## ============================================================
## PUBLIC API
## ============================================================
@log_execution_time_and_path
def simulate_cost(
    mode: str,
    providers: list[str],
    requested_model: Optional[str],
    text: Optional[str],
    path: Optional[str],
    recursive: bool,
    max_chars_per_file: int,
    expected_output_tokens: int,
    chunk_size: int,
    chunk_overlap: int,
    models_catalog: dict[str, Any],
    pricing_catalog: dict[str, Any],
    include_per_file: bool = False,
) -> dict[str, Any]:
    """
        Simulate costs for chat completions or embeddings

        High-level workflow:
            1) Validate inputs (text XOR path, providers not empty)
            2) Compute ScanStats from text or folder
            3) For each provider:
                - Resolve effective model
                - Resolve pricing row
                - Compute estimated USD cost
            4) Build a response with summary + results (+ per_file optional)

        Notes:
            - This module is a pure costing layer
            - Orchestration chaining (cost -> embeddings -> completion) happens in pipeline.py

        Args:
            mode: "chat" or "embeddings"
            providers: List of provider identifiers
            requested_model: Optional model override
            text: Optional direct text
            path: Optional folder path with txt files
            recursive: Recursive scan for folder mode
            max_chars_per_file: Max chars to read per file
            expected_output_tokens: Assumed output tokens for chat simulation
            chunk_size: Chunk size for embeddings simulation
            chunk_overlap: Chunk overlap for embeddings simulation
            models_catalog: Loaded models catalog dict
            pricing_catalog: Loaded pricing catalog dict
            include_per_file: Include per-file breakdown

        Returns:
            Dict with summary, results, warnings, and optional per_file
    """
    
    ## Validate mutually exclusive inputs
    if (text is None or text.strip() == "") and (path is None or path.strip() == ""):
        log_and_raise_validation_error(
            reason="Either 'text' or 'path' must be provided",
            context="simulate_cost",
        )

    if text is not None and text.strip() != "" and path is not None and path.strip() != "":
        log_and_raise_validation_error(
            reason="'text' and 'path' are mutually exclusive",
            context="simulate_cost",
        )

    if not providers:
        log_and_raise_validation_error(
            reason="providers list must not be empty",
            context="simulate_cost",
        )

    warnings: list[str] = []

    ## Build scan stats
    per_file_rows: list[dict[str, Any]] = []
    if text is not None and text.strip() != "":
        ## Scan direct text
        if mode == "chat":
            scan = _scan_text_input_for_chat(text)
        elif mode == "embeddings":
            scan = _scan_text_input_for_embeddings(text, chunk_size, chunk_overlap)
        else:
            log_and_raise_validation_error(
                reason=f"Invalid mode={mode}",
                context="simulate_cost",
            )
    else:
        ## Scan folder
        scan, per_file_rows = _scan_folder_txt_inputs(
            folder=Path(path or ""),
            recursive=recursive,
            max_chars_per_file=max_chars_per_file,
            mode=mode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    ## Simulate per provider
    results: list[dict[str, Any]] = []
    for provider in providers:
        ## Resolve model and pricing for provider
        effective_model = resolve_model_for_provider(
            provider=provider,
            requested_model=requested_model,
            models_catalog=models_catalog,
        )
        pricing = get_pricing_row(
            provider=provider,
            model=effective_model,
            pricing_catalog=pricing_catalog,
        )

        ## Compute cost depending on mode
        if mode == "chat":
            cost_usd = cost_usd_for_chat(
                pricing=pricing,
                input_tokens=scan.input_tokens,
                output_tokens=expected_output_tokens,
            )
            total_tokens = int(scan.input_tokens + expected_output_tokens)
            output_assumed = int(expected_output_tokens)

        elif mode == "embeddings":
            cost_usd = cost_usd_for_embeddings(
                pricing=pricing,
                input_tokens=scan.input_tokens,
            )
            total_tokens = int(scan.input_tokens)
            output_assumed = 0

        else:
            log_and_raise_validation_error(
                reason=f"Invalid mode={mode}",
                context="simulate_cost",
            )

        ## Build result row
        results.append(
            {
                "provider": provider,
                "model": effective_model,
                "input_tokens": int(scan.input_tokens),
                "output_tokens_assumed": int(output_assumed),
                "total_tokens": int(total_tokens),
                "estimated_cost_usd": float(round(cost_usd, 8)),
                "estimation_mode": "approx",
                "price_source": "config",
            }
        )

    response: dict[str, Any] = {
        "summary": {
            "mode": mode,
            "n_files": int(scan.n_files),
            "n_chars": int(scan.n_chars),
        },
        "results": results,
        "warnings": warnings,
    }

    ## Attach optional per-file breakdown
    if include_per_file:
        response["per_file"] = per_file_rows

    LOGGER.info(
        "Cost simulation completed | mode=%s | files=%s | chars=%s | providers=%s",
        mode,
        scan.n_files,
        scan.n_chars,
        len(providers),
    )

    return response