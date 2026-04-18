'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data quality checks for RAG pipelines: chunk validation, z-score, IQR and anomaly scoring."
'''

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.utils.logging_utils import get_logger
from src.utils.stats_utils import compute_mean_std, compute_iqr_bounds

try:
    from src.core.errors import ValidationError, DataError
except Exception:
    ValidationError = ValueError
    DataError = RuntimeError

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("data_quality")

## ============================================================
## HELPERS
## ============================================================
def _add_issue(
    issues: List[Dict[str, Any]],
    rule: str,
    level: str,
    message: str,
    details: Dict[str, Any] | None = None,
) -> None:
    """
        Append issue and log it

        Args:
            issues: Issue container
            rule: Rule identifier
            level: error or warning
            message: Issue message
            details: Optional metadata

        Returns:
            None
    """

    issue = {
        "rule": rule,
        "level": level,
        "message": message,
        "details": details or {},
    }

    issues.append(issue)

    ## log
    if level == "error":
        logger.error(f"{rule} - {message}")
    else:
        logger.warning(f"{rule} - {message}")

def _safe_token_count(text: str) -> int:
    """
        Compute token count safely

        Args:
            text: Input text

        Returns:
            Token count
    """

    ## simple whitespace tokenization
    return len(str(text).split())

## ============================================================
## DETECTION METHODS
## ============================================================
def _detect_zscore(arr: np.ndarray, threshold: float) -> np.ndarray:
    """
        Detect anomalies using z-score

        Args:
            arr: Input array
            threshold: Z-score threshold

        Returns:
            Boolean mask
    """

    mean, std = compute_mean_std(arr)

    if std == 0:
        return np.zeros_like(arr, dtype=bool)

    z = (arr - mean) / std

    return np.abs(z) > threshold

def _detect_iqr(arr: np.ndarray, multiplier: float) -> np.ndarray:
    """
        Detect anomalies using IQR

        Args:
            arr: Input array
            multiplier: IQR multiplier

        Returns:
            Boolean mask
    """

    lower, upper = compute_iqr_bounds(arr, multiplier)

    return (arr < lower) | (arr > upper)

## ============================================================
## MAIN FUNCTION
## ============================================================
def run_data_quality(
    texts: List[str],
    method: str = "zscore",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    strict: bool = False,
) -> Dict[str, Any]:
    """
        Run data quality checks for RAG chunks

        High-level workflow:
            1) Validate chunk inputs
            2) Detect empty / missing chunks
            3) Analyze chunk length distribution
            4) Detect statistical anomalies
            5) Compute global score

        Args:
            texts: List of chunk texts
            method: Detection method (zscore or iqr)
            z_threshold: Z-score threshold
            iqr_multiplier: IQR multiplier
            strict: Raise error if invalid

        Returns:
            Result dictionary
    """

    issues: List[Dict[str, Any]] = []

    try:
        ## BASIC VALIDATION
        if not texts:
            raise ValidationError("Empty dataset")

        ## EMPTY TEXT CHECK
        empty_mask = [text is None or str(text).strip() == "" for text in texts]

        if any(empty_mask):
            _add_issue(
                issues,
                "empty_chunk",
                "error",
                "Empty or missing chunk detected",
                {"count": int(sum(empty_mask))},
            )

        ## TOKEN LENGTH ANALYSIS
        lengths = np.array([_safe_token_count(text) for text in texts], dtype=float)

        ## detect anomalies
        if method == "zscore":
            anomaly_mask = _detect_zscore(lengths, z_threshold)
        elif method == "iqr":
            anomaly_mask = _detect_iqr(lengths, iqr_multiplier)
        else:
            raise ValidationError("Invalid anomaly method")

        if anomaly_mask.any():
            _add_issue(
                issues,
                "chunk_length_anomaly",
                "warning",
                "Abnormal chunk length detected",
                {"count": int(anomaly_mask.sum())},
            )

        ## GLOBAL SCORE
        error_count = len([issue for issue in issues if issue["level"] == "error"])
        total = len(texts)

        score = 1.0 - (error_count / max(total, 1))

        result = {
            "is_valid": error_count == 0,
            "errors": error_count,
            "warnings": len(issues) - error_count,
            "score": score,
            "issues": issues,
        }

        logger.info(f"Data quality score: {score}")

        ## STRICT MODE
        if strict and error_count > 0:
            raise ValidationError("Data quality failed")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Data quality failure: {exc}")
        raise DataError("Data quality pipeline failed") from exc