'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized data consistency checks for RAG pipeline (query, chunks, embeddings, metadata, cross-source, business rules, quality)."
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger
from src.utils.data_utils import (
    normalize_data,
    validate_schema,
    validate_types,
    compare_sources,
    check_business_rules,
    compute_quality_score,
    detect_duplicates,
)

try:
    from src.core.errors import ValidationError, DataError
except Exception:
    ValidationError = ValueError
    DataError = RuntimeError

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("data_consistency")

## ============================================================
## ISSUE HANDLING
## ============================================================
def _create_issue(
    rule: str,
    level: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
        Create standardized issue object

        Args:
            rule: Rule name
            level: Severity level
            message: Description
            details: Optional metadata

        Returns:
            Issue dictionary
    """

    ## Build issue
    issue = {
        "rule": rule,
        "level": level,
        "message": message,
        "details": details or {},
    }

    logger.debug(f"Issue created: {rule}")

    return issue

def _add_issue(
    issues: List[Dict[str, Any]],
    rule: str,
    level: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
        Append issue and log it

        Args:
            issues: Issue list
            rule: Rule name
            level: Severity
            message: Description
            details: Metadata
    """

    ## Create + append
    issue = _create_issue(rule, level, message, details)
    issues.append(issue)

    ## Log
    if level == "error":
        logger.error(f"{rule} - {message}")
    else:
        logger.warning(f"{rule} - {message}")

## ============================================================
## VALIDATIONS
## ============================================================
def _validate_file(
    file_path: Optional[str | Path],
    issues: List[Dict[str, Any]],
) -> Optional[Path]:
    """
        Validate file input

        Args:
            file_path: Path input
            issues: Issue list

        Returns:
            Path or None
    """

    if file_path is None:
        return None

    path = Path(file_path)

    ## Check exists
    if not path.exists():
        logger.error(f"File not found: {path}")
        _add_issue(issues, "file_exists", "error", "File does not exist", {"file": str(path)})
        return None

    ## Check type
    if not path.is_file():
        logger.error(f"Invalid file path: {path}")
        _add_issue(issues, "file_type", "error", "Path is not a file")
        return None

    return path

def _validate_query(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate user query

        Args:
            data: Input data
            issues: Issue list
    """

    query = data.get("query", "")

    ## Normalize
    normalized = normalize_data({"query": query}).get("query", "")
    data["query"] = normalized

    ## Empty
    if not normalized:
        logger.error("Empty query")
        _add_issue(issues, "query_empty", "error", "Query is empty")

    ## Too short
    if len(normalized) < 3:
        logger.warning("Query too short")
        _add_issue(issues, "query_short", "warning", "Query too short")

def _validate_chunks(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate retrieved chunks

        Args:
            data: Input data
            issues: Issue list
    """

    chunks = data.get("chunks")

    if chunks is None:
        return

    ## Check type
    if not isinstance(chunks, list):
        logger.error("Chunks must be list")
        _add_issue(issues, "chunks_type", "error", "Chunks must be list")
        return

    ## Check empty
    if len(chunks) == 0:
        logger.warning("No chunks retrieved")
        _add_issue(issues, "chunks_empty", "warning", "No chunks found")
        return

    ## Check each chunk
    for idx, chunk in enumerate(chunks):

        if not isinstance(chunk, str):
            logger.error(f"Invalid chunk at index {idx}")
            _add_issue(issues, "chunk_type", "error", f"Chunk {idx} must be string")
            continue

        if len(chunk.strip()) == 0:
            logger.warning(f"Empty chunk at index {idx}")
            _add_issue(issues, "chunk_empty", "warning", f"Chunk {idx} is empty")

def _validate_embeddings(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate embeddings consistency

        Args:
            data: Input data
            issues: Issue list
    """

    embeddings = data.get("embeddings")

    if embeddings is None:
        return

    ## Type
    if not isinstance(embeddings, list):
        logger.error("Embeddings must be list")
        _add_issue(issues, "embedding_type", "error", "Embeddings must be list")
        return

    ## Empty
    if len(embeddings) == 0:
        logger.error("Empty embeddings")
        _add_issue(issues, "embedding_empty", "error", "Embeddings empty")
        return

    ## Numeric check
    if not all(isinstance(v, (int, float)) for v in embeddings):
        logger.error("Invalid embedding values")
        _add_issue(issues, "embedding_values", "error", "Embeddings must be numeric")

    ## Dim check
    if len(embeddings) < 10:
        logger.warning("Embedding dim too small")
        _add_issue(issues, "embedding_dim", "warning", "Embedding dimension too small")

def _validate_structure(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate schema and types

        Args:
            data: Input data
            issues: Issue list
    """

    ## Schema
    for s in validate_schema(data):
        _add_issue(issues, s["rule"], s["level"], s["message"])

    ## Types
    for t in validate_types(data):
        _add_issue(issues, t["rule"], t["level"], t["message"])

def _validate_cross_source(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate cross-source consistency

        Args:
            data: Input data
            issues: Issue list
    """

    results = compare_sources(data)

    for r in results:
        _add_issue(issues, r["rule"], r["level"], r["message"], r.get("details"))

def _validate_business(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Apply business rules

        Args:
            data: Input data
            issues: Issue list
    """

    results = check_business_rules(data)

    for r in results:
        _add_issue(issues, r["rule"], r["level"], r["message"], r.get("details"))

def _validate_duplicates(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Detect duplicates

        Args:
            data: Input data
            issues: Issue list
    """

    duplicates = detect_duplicates(data)

    if duplicates:
        logger.warning("Duplicates detected")
        _add_issue(issues, "duplicates", "warning", "Duplicate values detected", {"values": duplicates})

## ============================================================
## QUALITY
## ============================================================
def _compute_quality(data: Dict[str, Any]) -> float:
    """
        Compute quality score

        Args:
            data: Input data

        Returns:
            Score
    """

    score = compute_quality_score(data)
    logger.debug(f"Quality score: {score}")
    return score

## ============================================================
## MAIN ENTRYPOINT
## ============================================================
def run_data_consistency(
    data: Dict[str, Any],
    file_path: Optional[str | Path] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
        Run full RAG consistency pipeline

        Args:
            data: Input data
            file_path: Optional file path
            strict: Raise error if inconsistency

        Returns:
            Result dictionary
    """

    issues: List[Dict[str, Any]] = []

    try:
        ## Normalize
        data = normalize_data(data)

        ## File
        path = _validate_file(file_path, issues)

        ## Query
        _validate_query(data, issues)

        ## Chunks
        _validate_chunks(data, issues)

        ## Embeddings
        _validate_embeddings(data, issues)

        ## Structure
        _validate_structure(data, issues)

        ## Cross-source
        _validate_cross_source(data, issues)

        ## Business rules
        _validate_business(data, issues)

        ## Duplicates
        _validate_duplicates(data, issues)

        ## Quality
        quality_score = _compute_quality(data)

        errors = [i for i in issues if i["level"] == "error"]

        result = {
            "is_consistent": len(errors) == 0,
            "errors": len(errors),
            "warnings": len(issues) - len(errors),
            "quality_score": quality_score,
            "issues": issues,
            "file": str(path) if path else None,
        }

        logger.info(f"Consistency result: {result['is_consistent']}")

        if strict and errors:
            logger.error("Strict mode failure")
            raise ValidationError("Data consistency failed")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        raise DataError("Consistency pipeline failed") from exc