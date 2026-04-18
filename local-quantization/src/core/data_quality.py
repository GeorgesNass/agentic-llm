'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Lightweight anomaly detection for local-quantization: tensor checks, z-score, IQR, NaN/inf and range validation."
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
        Convert input to numpy array

        Args:
            data: List or numpy array

        Returns:
            Numpy array
    """

    ## convert safely
    return np.asarray(data, dtype=float)

def _add_issue(
    issues: List[Dict[str, Any]],
    rule: str,
    level: str,
    message: str,
    details: Dict[str, Any] | None = None,
) -> None:
    """
        Append issue and log

        Args:
            issues: List of issues
            rule: Rule name
            level: error or warning
            message: Message
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

    ## log
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
            arr: Numpy array
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
            arr: Numpy array
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
    data: Union[List[Any], np.ndarray],
    method: str = "zscore",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    strict: bool = False,
) -> Dict[str, Any]:
    """
        Run anomaly detection on tensor data

        High-level workflow:
            1) Convert input to numpy
            2) Detect invalid values (NaN / inf)
            3) Detect anomalies (z-score / IQR)
            4) Compute score

        Args:
            data: Tensor-like input
            method: zscore or iqr
            z_threshold: Threshold
            iqr_multiplier: Multiplier
            strict: Raise error if invalid

        Returns:
            Result dictionary
    """

    issues: List[Dict[str, Any]] = []

    try:
        ## convert input
        arr = _to_numpy(data)

        ## flatten for analysis
        flat = arr.flatten()

        ## detect NaN / inf
        invalid_mask = np.isnan(flat) | np.isinf(flat)

        if invalid_mask.any():
            _add_issue(
                issues,
                "invalid_values",
                "error",
                "NaN or inf detected",
                {"count": int(invalid_mask.sum())},
            )

        ## select method
        if method == "zscore":
            anomaly_mask = _detect_zscore(flat, z_threshold)
        elif method == "iqr":
            anomaly_mask = _detect_iqr(flat, iqr_multiplier)
        else:
            raise ValidationError("Invalid method")

        ## anomalies
        if anomaly_mask.any():
            _add_issue(
                issues,
                "anomaly_detected",
                "warning",
                "Statistical anomalies detected",
                {"count": int(anomaly_mask.sum())},
            )

        ## range sanity check (important for quantization)
        if flat.size > 0:
            min_val = float(np.min(flat))
            max_val = float(np.max(flat))

            if abs(min_val) > 1e6 or abs(max_val) > 1e6:
                _add_issue(
                    issues,
                    "extreme_values",
                    "warning",
                    "Extreme values detected",
                    {"min": min_val, "max": max_val},
                )

        ## global mask
        global_mask = invalid_mask | anomaly_mask

        ## score
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

        ## strict mode
        if strict and errors:
            raise ValidationError("Data quality failed")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Data quality failure: {exc}")
        raise DataError("Data quality pipeline failed") from exc