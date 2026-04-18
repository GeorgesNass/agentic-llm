'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Lightweight anomaly detection for autonomous-ai-platform: z-score, IQR, payload numeric checks and scoring."
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

        Returns:
            None
    """

    ## build issue
    issue = {
        "rule": rule,
        "level": level,
        "message": message,
        "details": details or {},
    }

    ## append issue
    issues.append(issue)

    ## log issue
    if level == "error":
        logger.error(f"{rule} - {message}")
    else:
        logger.warning(f"{rule} - {message}")

## ============================================================
## DETECTIONS
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
            Boolean mask of anomalies
    """

    ## compute statistics
    mean, std = compute_mean_std(series)

    ## avoid division by zero
    if std == 0:
        return pd.Series(False, index=series.index)

    ## compute z-score
    z = (series - mean) / std

    ## return anomaly mask
    return z.abs() > threshold

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
            Boolean mask of anomalies
    """

    ## compute bounds
    lower, upper = compute_iqr_bounds(series, multiplier)

    ## detect anomalies
    return (series < lower) | (series > upper)

## ============================================================
## MAIN ENTRYPOINT
## ============================================================
def run_data_quality(
    data: Union[Dict[str, Any], List[Any], np.ndarray, pd.DataFrame],
    method: str = "zscore",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    strict: bool = False,
) -> Dict[str, Any]:
    """
        Run lightweight anomaly detection for runtime payload

        High-level workflow:
            1) Normalize input payload
            2) Extract numeric values
            3) Detect invalid values (NaN / inf)
            4) Detect anomalies (z-score / IQR)
            5) Compute score and flags

        Design choice:
            - Optimized for API / agent runtime payloads
            - No heavy DataFrame logic
            - Fast validation before orchestration

        Args:
            data: Input payload (dict / list / array / DataFrame)
            method: Detection method (zscore / iqr)
            z_threshold: Z-score threshold
            iqr_multiplier: IQR multiplier
            strict: Raise error if anomalies detected

        Returns:
            Structured result dictionary
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

        ## extract numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        ## no numeric data
        if not numeric_cols:
            logger.warning("No numeric payload detected")

            return {
                "is_valid": True,
                "errors": 0,
                "warnings": 0,
                "score": 1.0,
                "issues": [],
            }

        global_mask = pd.Series(False, index=df.index)

        ## iterate numeric columns
        for col in numeric_cols:

            ## convert to float
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

            ## select detection method
            if method == "zscore":
                anomaly_mask = _detect_zscore(series, z_threshold)
            elif method == "iqr":
                anomaly_mask = _detect_iqr(series, iqr_multiplier)
            else:
                raise ValidationError("Invalid anomaly method")

            ## flag anomalies
            if anomaly_mask.any():
                _add_issue(
                    issues,
                    "anomaly_detected",
                    "warning",
                    f"Anomalies detected in {col}",
                    {"count": int(anomaly_mask.sum())},
                )

            ## aggregate mask
            global_mask = global_mask | anomaly_mask | invalid_mask

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

        ## strict mode handling
        if strict and errors:
            logger.error("Strict mode failure")
            raise ValidationError("Data quality failed")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Data quality failure: {exc}")
        raise DataError("Data quality pipeline failed") from exc