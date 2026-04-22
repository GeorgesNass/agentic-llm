'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data drift detection for local-finetuning: dataset, labels, text length and training metrics monitoring."
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


from src.utils.logging_utils import get_logger
from src.utils.drift_utils import (
    compute_ks_test,
    compute_chi2_test,
    compute_text_stats,
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
            1) Create issue
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
def _detect_numeric_drift(
    ref: pd.Series,
    cur: pd.Series,
    column: str,
    threshold: float,
    issues: List[Dict[str, Any]],
) -> float:
    """
        Detect drift for numeric feature

        High-level workflow:
            1) Compute KS test
            2) Compare p-value
            3) Raise issue if drift detected

        Args:
            ref: Reference series
            cur: Current series
            column: Column name
            threshold: p-value threshold
            issues: Issue container

        Returns:
            p_value
    """

    stat, p_value = compute_ks_test(ref, cur)

    if p_value < threshold:
        _add_issue(
            issues,
            "drift_numeric",
            "warning",
            f"Drift detected in {column}",
            {"p_value": float(p_value)},
        )

    return float(p_value)

def _detect_categorical_drift(
    ref: pd.Series,
    cur: pd.Series,
    column: str,
    threshold: float,
    issues: List[Dict[str, Any]],
) -> float:
    """
        Detect drift for categorical feature

        High-level workflow:
            1) Compute Chi-square test
            2) Compare p-value
            3) Raise issue if drift detected

        Args:
            ref: Reference series
            cur: Current series
            column: Column name
            threshold: p-value threshold
            issues: Issue container

        Returns:
            p_value
    """

    stat, p_value = compute_chi2_test(ref, cur)

    if p_value < threshold:
        _add_issue(
            issues,
            "drift_categorical",
            "warning",
            f"Drift detected in {column}",
            {"p_value": float(p_value)},
        )

    return float(p_value)

## ============================================================
## MAIN ENTRYPOINT
## ============================================================
def run_data_drift(
    df_ref: pd.DataFrame,
    df_current: pd.DataFrame,
    p_value_threshold: float = 0.05,
    strict: bool = False,
) -> Dict[str, Any]:
    """
        Run data drift detection for local-finetuning

        High-level workflow:
            1) Validate datasets
            2) Detect drift on numeric features
            3) Detect drift on labels
            4) Detect drift on text features
            5) Aggregate issues and compute score

        Design choice:
            - KS test for numeric drift
            - Chi-square for categorical drift
            - Text drift via length comparison
            - Equal weight for all drift signals

        Args:
            df_ref: Reference dataset (training data)
            df_current: New dataset (incoming data)
            p_value_threshold: Statistical threshold
            strict: Raise error if drift detected

        Returns:
            Dictionary with drift results
    """

    issues: List[Dict[str, Any]] = []

    try:
        if df_ref.empty or df_current.empty:
            raise ValidationError("Empty datasets provided")

        drift_flags: List[bool] = []

        ## NUMERIC FEATURES DRIFT
        for col in df_ref.columns:
            if col not in df_current.columns:
                continue

            if pd.api.types.is_numeric_dtype(df_ref[col]):
                p_value = _detect_numeric_drift(
                    df_ref[col],
                    df_current[col],
                    col,
                    p_value_threshold,
                    issues,
                )
                drift_flags.append(p_value < p_value_threshold)

        ## LABEL DRIFT
        if "label" in df_ref.columns:
            p_value = _detect_categorical_drift(
                df_ref["label"],
                df_current["label"],
                "label",
                p_value_threshold,
                issues,
            )
            drift_flags.append(p_value < p_value_threshold)

        ## TEXT FEATURES DRIFT
        ref_text = compute_text_stats(df_ref)
        cur_text = compute_text_stats(df_current)

        for col in ref_text.columns:
            p_value = _detect_numeric_drift(
                ref_text[col],
                cur_text[col],
                col,
                p_value_threshold,
                issues,
            )
            drift_flags.append(p_value < p_value_threshold)

        ## compute score
        drift_score = 1.0 - (sum(drift_flags) / len(drift_flags)) if drift_flags else 1.0

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
            report_paths = generate_evidently_report(df_ref, df_current)
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