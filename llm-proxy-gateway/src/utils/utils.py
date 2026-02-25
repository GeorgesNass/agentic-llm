'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Generic helpers: environment parsing, safe file loading, path utilities, exports and basic statistics."
'''

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, List

from src.utils.logging_utils import get_logger
from src.core.errors import (
    log_and_raise_missing_path,
    log_and_raise_validation_error,
)
from src.utils.costing_utils import ScanStats
from src.core.errors import log_and_raise_missing_file

LOGGER = get_logger(__name__)

## ============================================================
## ENVIRONMENT HELPERS
## ============================================================
def get_env_str(key: str, default: str = "") -> str:
    """
        Retrieve a string environment variable

        Args:
            key: Environment variable name
            default: Default value if missing

        Returns:
            Resolved string value
    """
    
    return os.getenv(key, default)

def get_env_int(key: str, default: int = 0) -> int:
    """
        Retrieve an integer environment variable

        Args:
            key: Environment variable name
            default: Default value if missing

        Returns:
            Resolved integer value
    """
    
    raw = os.getenv(key)

    ## Return default if variable not defined
    if raw is None:
        return default

    ## Parse integer safely
    try:
        return int(raw)
    except ValueError:
        return default

def get_env_bool(key: str, default: bool = False) -> bool:
    """
        Retrieve a boolean environment variable

        Args:
            key: Environment variable name
            default: Default value if missing

        Returns:
            Resolved boolean value
    """
    
    raw = os.getenv(key)

    ## Return default if variable not defined
    if raw is None:
        return default

    return raw.strip().lower() in {"1", "true", "yes", "y"}

## ============================================================
## PATH HELPERS
## ============================================================
def resolve_path(path_str: str | Path) -> Path:
    """
        Resolve a path to an absolute Path

        Args:
            path_str: Raw path

        Returns:
            Absolute resolved Path
    """
    
    return Path(path_str).expanduser().resolve()

def ensure_parent_dir(path: str | Path) -> None:
    """
        Ensure parent directory exists

        Args:
            path: File path
    """
    
    p = resolve_path(path)

    ## Create parent folder if needed
    p.parent.mkdir(parents=True, exist_ok=True)

def ensure_dir(path: str | Path) -> None:
    """
        Ensure a directory exists

        Args:
            path: Directory path
    """
    
    p = resolve_path(path)

    ## Create directory if needed
    p.mkdir(parents=True, exist_ok=True)

## ============================================================
## FILE HELPERS
## ============================================================
def safe_read_text(
    path: str | Path,
    encoding: str = "utf-8",
    max_chars: int = 200_000,
) -> str:
    """
        Safely read a text file with size limits

        Args:
            path: File path
            encoding: Text encoding
            max_chars: Maximum number of characters to read

        Returns:
            File content
    """
    
    file_path = resolve_path(path)

    ## Read file with upper bound
    with file_path.open("r", encoding=encoding, errors="replace") as f:
        content = f.read(max_chars)

    return content

def safe_load_json(path: str | Path) -> dict[str, Any]:
    """
        Safely load a JSON file

        Args:
            path: JSON file path

        Returns:
            Parsed JSON dictionary
    """
    
    json_path = resolve_path(path)

    ## Load JSON safely
    with json_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    return data

def list_txt_files(folder: str | Path, recursive: bool = True) -> List[Path]:
    """
        List .txt files in a folder

        Args:
            folder: Folder path
            recursive: Whether to scan recursively

        Returns:
            List of text file paths
    """
    
    folder_path = resolve_path(folder)
    pattern = "**/*.txt" if recursive else "*.txt"

    ## Collect text files
    return sorted([p for p in folder_path.glob(pattern) if p.is_file()])

## ============================================================
## TEXT HELPERS
## ============================================================
def normalize_text_basic(text: str) -> str:
    """
        Normalize text for simple matching

        High-level workflow:
            1) Lowercase
            2) Strip spaces
            3) Collapse repeated whitespace

        Args:
            text: Raw text

        Returns:
            Normalized text
    """
    lowered = text.lower().strip()

    ## Replace multiple whitespace with single space
    import re
    return re.sub(r"\s+", " ", lowered)

## ============================================================
## TOKEN ESTIMATION HELPERS
## ============================================================
def approximate_token_count(text: str) -> int:
    """
        Approximate token count using a char-based heuristic

        Notes:
            - Rough approximation: 1 token ~ 4 characters
            - This is used for cost simulation only

        Args:
            text: Input text

        Returns:
            Approximate token count
    """
    
    stripped = text.strip()
    if stripped == "":
        return 0

    ## Heuristic token estimate
    return max(1, (len(stripped) + 3) // 4)

def _iter_char_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
        Split text into overlapping character chunks

        Args:
            text: Input text
            chunk_size: Chunk size in characters
            chunk_overlap: Overlap in characters

        Returns:
            List of chunk strings
    """
    
    if chunk_size <= 0:
        log_and_raise_validation_error(
            reason=f"chunk_size must be > 0, got {chunk_size}",
            context="chunking",
        )

    if chunk_overlap < 0:
        log_and_raise_validation_error(
            reason=f"chunk_overlap must be >= 0, got {chunk_overlap}",
            context="chunking",
        )

    if chunk_overlap >= chunk_size:
        log_and_raise_validation_error(
            reason="chunk_overlap must be < chunk_size",
            context="chunking",
        )

    ## Early exit for short text
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = chunk_size - chunk_overlap

    ## Build overlapping windows
    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip() != "":
            chunks.append(chunk)

        if end >= len(text):
            break

    return chunks

def estimate_input_tokens_for_embeddings(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """
        Estimate total input tokens for embeddings by chunking text

        Args:
            text: Raw input text
            chunk_size: Chunk size in characters
            chunk_overlap: Chunk overlap in characters

        Returns:
            Total estimated tokens
    """
    
    ## Chunk text and sum approximate tokens
    chunks = _iter_char_chunks(text, chunk_size, chunk_overlap)
    return sum(approximate_token_count(c) for c in chunks)

## ============================================================
## EXPORT HELPERS
## ============================================================
def export_dicts_to_csv(
    rows: List[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """
        Export a list of dictionaries to CSV

        Args:
            rows: List of dictionaries
            output_path: Destination path

        Returns:
            Output path
    """
    
    if not rows:
        raise ValueError("No rows provided for CSV export")

    out_path = resolve_path(output_path)
    ensure_parent_dir(out_path)

    fieldnames = list(rows[0].keys())

    ## Write CSV file
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path

def compute_basic_stats(values: List[float]) -> dict[str, float]:
    """
        Compute basic statistics

        Args:
            values: List of numeric values

        Returns:
            Dict with min, max, mean
    """
    
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}

    vmin = min(values)
    vmax = max(values)
    mean = sum(values) / float(len(values))

    return {
        "min": float(vmin),
        "max": float(vmax),
        "mean": float(mean),
    }
    
## ============================================================
## NUMERIC HELPERS
## ============================================================
def as_float(value: float | None) -> float:
    """
        Convert optional float to float safely

        Args:
            value: Optional float

        Returns:
            Float value or 0.0
    """
    
    if value is None:
        return 0.0
    
    return float(value)

def mean(values: list[float]) -> float:
    """
        Compute mean safely

        Args:
            values: List of numeric values

        Returns:
            Mean value or 0.0 if empty
    """
    
    if not values:
        return 0.0

    return float(sum(values) / float(len(values)))  

## ============================================================
## INPUT HELPERS
## ============================================================
def _parse_providers(raw: str) -> list[str]:
    """
        Parse providers list from comma-separated string

        Args:
            raw: Comma-separated providers string

        Returns:
            List of providers
    """
    
    parts = [p.strip() for p in raw.split(",")]
    
    return [p for p in parts if p != ""]

def _load_messages(messages_json_path: str) -> list[dict[str, Any]]:
    """
        Load OpenAI-like messages list from a JSON file

        Args:
            messages_json_path: Path to JSON file

        Returns:
            List of messages dictionaries
    """
    path = Path(messages_json_path).expanduser().resolve()
    if not path.exists():
        log_and_raise_missing_file(path, context="_load_lines")

    data = json.loads(safe_read_text(path))
    if not isinstance(data, list):
        raise ValueError("messages-json must contain a JSON list")

    return data

def _load_lines(path_str: str) -> list[str]:
    """
        Load a text file into lines

        Args:
            path_str: File path

        Returns:
            List of stripped lines
    """
    
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        log_and_raise_missing_file(path, context="_load_lines")

    content = safe_read_text(path)
    
    return [line.strip() for line in content.splitlines() if line.strip() != ""]

## ============================================================
## SCAN HELPERS
## ============================================================
def scan_text_input_for_chat(text: str) -> ScanStats:
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
    
def scan_text_input_for_embeddings(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> ScanStats:
    """
        Scan a direct text input for embeddings mode

        High-level workflow:
            1) Count characters
            2) Chunk the text
            3) Estimate tokens per chunk
            4) Aggregate total tokens

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

def scan_folder_txt_inputs(
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
                - Append per-file summary
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
        log_and_raise_missing_path(folder_path, context="scan_folder_txt_inputs")

    ## List files
    files = list_txt_files(folder_path, recursive=recursive)

    total_chars = 0
    total_tokens = 0
    per_file: list[dict[str, Any]] = []

    ## Scan each file
    for fp in files:
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
                context="scan_folder_txt_inputs",
            )

        total_chars += n_chars
        total_tokens += approx_tokens

        ## Per-file breakdown
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