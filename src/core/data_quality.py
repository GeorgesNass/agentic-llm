'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Anomaly detection for llm-proxy-gateway: z-score, IQR, API payload metrics (tokens, latency, cost)."
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

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
## ISSUE HANDLING
## ============================================================
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
            issues: Issue container
            rule: Rule name
            level: Severity level
            message: Description
            details: Optional metadata
    """

    ## build issue
    issue = {
        "rule": rule,
        "level": level,
        "message": message,
        "details": details or {},
    }

    ## append
    issues.append(issue)

    ## log
    if level == "error":
        logger.error(f"{rule} - {message}")
    else:
        logger.warning(f"{rule} - {message}")

## ============================================================
## Z-SCORE DETECTION
## ============================================================
def _detect_zscore(series: pd.Series, threshold: float) -> pd.Series:
    """
        Detect anomalies using z-score

        High-level workflow:
            1) Compute mean and std
            2) Compute z-score
            3) Flag anomalies

        Args:
            series: Numerical series
            threshold: Z-score threshold

        Returns:
            Boolean mask
    """

    ## compute stats
    mean, std = compute_mean_std(series)

    if std == 0:
        return pd.Series(False, index=series.index)

    ## compute z-score
    z = (series - mean) / std

    return z.abs() > threshold

## ============================================================
## IQR DETECTION
## ============================================================
def _detect_iqr(series: pd.Series, multiplier: float) -> pd.Series:
    """
        Detect anomalies using IQR

        High-level workflow:
            1) Compute IQR bounds
            2) Flag anomalies

        Args:
            series: Numerical series
            multiplier: IQR multiplier

        Returns:
            Boolean mask
    """

    ## compute bounds
    lower, upper = compute_iqr_bounds(series, multiplier)

    ## detect anomalies
    return (series < lower) | (series > upper)

## ============================================================
## MAIN ENTRYPOINT
## ============================================================
def run_data_quality(
    data: Union[pd.DataFrame, Dict[str, Any], List[Any], np.ndarray],
    method: str = "zscore",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    strict: bool = False,
) -> Dict[str, Any]:
    """
        Run anomaly detection for LLM gateway metrics

        High-level workflow:
            1) Normalize input payload
            2) Detect invalid values (NaN / inf)
            3) Detect anomalies (z-score / IQR)
            4) Compute anomaly score
            5) Return structured result

        Design choice:
            - Optimized for API metrics (tokens, latency, cost)
            - Lightweight validation before LLM routing

        Args:
            data: Input dataset (dict / DataFrame / array)
            method: Detection method (zscore / iqr)
            z_threshold: Z-score threshold
            iqr_multiplier: IQR multiplier
            strict: Raise error if anomalies detected

        Returns:
            Result dictionary with issues and score
    """

    issues: List[Dict[str, Any]] = []

    try:
        ## normalize input
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        ## select numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            logger.warning("No numeric columns found")
            return {
                "is_valid": True,
                "errors": 0,
                "warnings": 0,
                "score": 1.0,
                "issues": [],
            }

        global_mask = pd.Series(False, index=df.index)

        ## iterate columns
        for col in columns:

            ## cast
            series = df[col].astype(float)

            ## detect invalid values
            invalid_mask = series.isna() | np.isinf(series)

            if invalid_mask.any():
                _add_issue(
                    issues,
                    "invalid_values",
                    "error",
                    f"Invalid values in {col}",
                    {"count": int(invalid_mask.sum())},
                )

            ## anomaly detection
            if method == "zscore":
                mask = _detect_zscore(series, z_threshold)
            elif method == "iqr":
                mask = _detect_iqr(series, iqr_multiplier)
            else:
                raise ValidationError("Invalid anomaly method")

            if mask.any():
                _add_issue(
                    issues,
                    "anomaly_detected",
                    "warning",
                    f"Anomalies detected in {col}",
                    {"count": int(mask.sum())},
                )

            ## aggregate mask
            global_mask = global_mask | mask | invalid_mask

        ## compute score
        score = 1.0 - float(global_mask.mean())

        ## split errors
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
            logger.error("Strict mode failure")
            raise ValidationError("Data quality failed")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Data quality failure: {exc}")
        raise DataError("Data quality pipeline failed") from exc