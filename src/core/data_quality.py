'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Anomaly detection for local-finetuning: tensors, gradients, z-score, IQR, NaN/inf checks."
'''

from __future__ import annotations

from typing import Any, Dict, List, Union

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
def _to_numpy(data: Union[List[Any], np.ndarray]) -> np.ndarray:
    """
        Convert input data to numpy array

        Args:
            data: Input list or numpy array

        Returns:
            Flattened numpy array
    """

    ## convert to numpy and flatten
    return np.asarray(data, dtype=float).flatten()

def _add_issue(
    issues: List[Dict[str, Any]],
    rule: str,
    level: str,
    message: str,
    details: Dict[str, Any] | None = None,
) -> None:
    """
        Append issue to list and log it

        Args:
            issues: Issues container
            rule: Rule identifier
            level: error or warning
            message: Issue message
            details: Optional details

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

    ## logging
    if level == "error":
        logger.error(f"{rule} - {message}")
    else:
        logger.warning(f"{rule} - {message}")

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

    ## compute stats
    mean, std = compute_mean_std(arr)

    ## avoid division by zero
    if std == 0:
        return np.zeros_like(arr, dtype=bool)

    ## compute z-score
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

    ## compute bounds
    lower, upper = compute_iqr_bounds(arr, multiplier)

    return (arr < lower) | (arr > upper)

## ============================================================
## MAIN FUNCTION
## ============================================================
def run_data_quality(
    data: Union[List[Any], np.ndarray],
    method: str = "zscore",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    strict: bool = False,
) -> Dict[str, Any]:
    """
        Run anomaly detection on finetuning tensors

        High-level workflow:
            1) Convert input to numpy
            2) Detect NaN / inf values
            3) Detect statistical anomalies
            4) Detect gradient explosion
            5) Compute global score

        Args:
            data: Tensor-like input (weights, gradients, loss)
            method: Detection method (zscore or iqr)
            z_threshold: Z-score threshold
            iqr_multiplier: IQR multiplier
            strict: Raise error if invalid

        Returns:
            Result dictionary
    """

    issues: List[Dict[str, Any]] = []

    try:
        ## PREPARE DATA
        arr = _to_numpy(data)

        if arr.size == 0:
            raise ValidationError("Empty input data")

        ## INVALID VALUES (NaN / INF)
        invalid_mask = np.isnan(arr) | np.isinf(arr)

        if invalid_mask.any():
            _add_issue(
                issues,
                "invalid_values",
                "error",
                "NaN or inf detected",
                {"count": int(invalid_mask.sum())},
            )

        ## STATISTICAL ANOMALIES
        if method == "zscore":
            anomaly_mask = _detect_zscore(arr, z_threshold)
        elif method == "iqr":
            anomaly_mask = _detect_iqr(arr, iqr_multiplier)
        else:
            raise ValidationError("Invalid anomaly method")

        if anomaly_mask.any():
            _add_issue(
                issues,
                "statistical_anomaly",
                "warning",
                "Outliers detected",
                {"count": int(anomaly_mask.sum())},
            )

        ## GRADIENT EXPLOSION CHECK
        max_abs = float(np.max(np.abs(arr)))

        if max_abs > 1e4:
            _add_issue(
                issues,
                "gradient_explosion",
                "warning",
                "Very large values detected",
                {"max_abs": max_abs},
            )

        ## GLOBAL SCORE
        global_mask = invalid_mask | anomaly_mask

        score = 1.0 - float(np.mean(global_mask))

        errors = [i for i in issues if i["level"] == "error"]

        result = {
            "is_valid": len(errors) == 0,
            "errors": len(errors),
            "warnings": len(issues) - len(errors),
            "score": score,
            "issues": issues,
        }

        logger.info(f"Data quality score: {score}")

        ## STRICT MODE
        if strict and errors:
            raise ValidationError("Data quality failed")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Data quality failure: {exc}")
        raise DataError("Data quality pipeline failed") from exc