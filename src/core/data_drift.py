'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data drift detection for local-quantization: weights, model size and quantization metrics monitoring."
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logging_utils import get_logger
from src.utils.drift_utils import (
    compute_ks_test,
    compute_weight_stats,
    generate_evidently_report
)

try:
    from src.core.errors import ValidationError, DataError
except Exception:
    ValidationError = ValueError
    DataError = RuntimeError

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("data_drift")

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

        High-level workflow:
            1) Build issue dictionary
            2) Attach optional metadata
            3) Return structured issue

        Args:
            rule: Rule name
            level: Severity level
            message: Description
            details: Optional metadata

        Returns:
            Issue dictionary
    """

    issue = {
        "rule": rule,
        "level": level,
        "message": message,
        "details": details or {},
    }

    logger.debug(f"Issue created: {rule} - {level}")

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

        High-level workflow:
            1) Create issue object
            2) Append to list
            3) Log severity

        Args:
            issues: Issue container
            rule: Rule name
            level: Severity level
            message: Description
            details: Optional metadata

        Returns:
            None
    """

    issue = _create_issue(rule, level, message, details)
    issues.append(issue)

    if level == "error":
        logger.error(f"{rule} - {message}")
    else:
        logger.warning(f"{rule} - {message}")

## ============================================================
## DRIFT DETECTION
## ============================================================
def _detect_weights_drift(
    ref: np.ndarray,
    cur: np.ndarray,
    threshold: float,
    issues: List[Dict[str, Any]],
) -> float:
    """
        Detect drift on model weights

        High-level workflow:
            1) Flatten weights arrays
            2) Run KS test
            3) Compare p-value with threshold

        Args:
            ref: Reference weights
            cur: Current weights
            threshold: p-value threshold
            issues: Issue container

        Returns:
            p_value from KS test
    """

    ref_flat = ref.flatten()
    cur_flat = cur.flatten()

    stat, p_value = compute_ks_test(ref_flat, cur_flat)

    if p_value < threshold:
        _add_issue(
            issues,
            "weights_drift",
            "warning",
            "Drift detected on model weights",
            {"p_value": float(p_value)},
        )

    return float(p_value)

def _detect_size_drift(
    ref_size: float,
    cur_size: float,
    threshold: float,
    issues: List[Dict[str, Any]],
) -> float:
    """
        Detect drift on model size

        High-level workflow:
            1) Compute relative difference
            2) Compare with threshold
            3) Raise issue if exceeded

        Args:
            ref_size: Reference model size
            cur_size: Current model size
            threshold: Size drift threshold
            issues: Issue container

        Returns:
            relative difference
    """

    if ref_size == 0:
        return 0.0

    diff = abs(cur_size - ref_size) / ref_size

    if diff > threshold:
        _add_issue(
            issues,
            "model_size_drift",
            "warning",
            "Drift detected on model size",
            {"ref": ref_size, "current": cur_size, "diff": diff},
        )

    return float(diff)

## ============================================================
## MAIN ENTRYPOINT
## ============================================================
def run_data_drift(
    weights_ref: np.ndarray,
    weights_current: np.ndarray,
    model_size_ref: float,
    model_size_current: float,
    p_value_threshold: float = 0.05,
    size_threshold: float = 0.2,
    strict: bool = False,
) -> Dict[str, Any]:
    """
        Run data drift detection for local quantization

        High-level workflow:
            1) Validate inputs
            2) Detect drift on weights distribution
            3) Detect drift on model size
            4) Aggregate issues and compute score

        Design choice:
            - KS test used for weights comparison
            - Relative difference used for model size
            - Equal contribution to global drift score

        Args:
            weights_ref: Reference model weights
            weights_current: Quantized model weights
            model_size_ref: Reference model size
            model_size_current: Quantized model size
            p_value_threshold: Statistical threshold
            size_threshold: Size drift threshold
            strict: Raise error if drift detected

        Returns:
            Dictionary with drift results
    """

    issues: List[Dict[str, Any]] = []

    try:
        if weights_ref.size == 0 or weights_current.size == 0:
            raise ValidationError("Empty weights provided")

        drift_flags: List[bool] = []

        ## WEIGHTS DRIFT
        p_value = _detect_weights_drift(
            weights_ref,
            weights_current,
            p_value_threshold,
            issues,
        )
        drift_flags.append(p_value < p_value_threshold)

        ## MODEL SIZE DRIFT
        size_diff = _detect_size_drift(
            model_size_ref,
            model_size_current,
            size_threshold,
            issues,
        )
        drift_flags.append(size_diff > size_threshold)

        ## compute score
        drift_score = 1.0 - (sum(drift_flags) / len(drift_flags))

        errors = [i for i in issues if i["level"] == "error"]

        result = {
            "is_drift_ok": len(errors) == 0,
            "errors": len(errors),
            "warnings": len(issues) - len(errors),
            "drift_score": drift_score,
            "issues": issues,
        }

        logger.info(f"Drift score: {drift_score}")

        ## EVIDENTLY REPORT
        try:
            df_ref = np.array(weights_ref).reshape(-1, 1)
            df_cur = np.array(weights_current).reshape(-1, 1)

            report_paths = generate_evidently_report(
                pd.DataFrame(df_ref, columns=["weights"]),
                pd.DataFrame(df_cur, columns=["weights"]),
            )

            result["evidently_report"] = report_paths

        except Exception as e:
            logger.warning(f"Evidently failed: {e}")
            
        if strict and drift_score < 1.0:
            raise ValidationError("Data drift detected")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        raise DataError("Data drift pipeline failed") from exc