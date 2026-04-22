'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data drift detection for rag-drive-gcp: embeddings, chunks, metadata and Evidently reporting."
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger
from src.utils.drift_utils import (
    compute_ks_test,
    compute_chi2_test,
    compute_text_stats,
    generate_evidently_report,
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

        Args:
            rule: Rule name
            level: Severity level
            message: Description
            details: Optional metadata

        Returns:
            Issue dictionary
    """

    return {
        "rule": rule,
        "level": level,
        "message": message,
        "details": details or {},
    }

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
            1) Create standardized issue object
            2) Append issue to issues list
            3) Log message depending on severity level

        Args:
            issues: Issue container
            rule: Rule name
            level: Severity level (warning | error)
            message: Description of the issue
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
## EMBEDDING STATS
## ============================================================
def _compute_embedding_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
        Compute embedding statistics

        High-level workflow:
            1) Extract embedding vectors
            2) Compute norms
            3) Compute mean value

        Args:
            df: Input dataset

        Returns:
            DataFrame with embedding stats
    """

    data: Dict[str, pd.Series] = {}

    if "embedding" in df.columns:

        ## ensure array
        emb = df["embedding"].apply(
            lambda x: np.array(x) if isinstance(x, (list, tuple)) else np.array([])
        )

        ## norm
        data["embedding_norm"] = emb.apply(lambda x: np.linalg.norm(x) if x.size else 0.0)

        ## mean value
        data["embedding_mean"] = emb.apply(lambda x: float(np.mean(x)) if x.size else 0.0)

    return pd.DataFrame(data)

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
        Detect numeric drift using KS test
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
        Detect categorical drift using Chi-square test

        High-level workflow:
            1) Compute Chi-square test on categorical distributions
            2) Compare p-value with configured threshold
            3) Add warning issue if drift is detected

        Args:
            ref: Reference series
            cur: Current series
            column: Column name
            threshold: Statistical p-value threshold
            issues: Issue container

        Returns:
            p_value from Chi-square test
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
        Run data drift detection for RAG system

        High-level workflow:
            1) Validate datasets
            2) Detect drift on embeddings
            3) Detect drift on chunk text features
            4) Detect drift on metadata (source, mime type)
            5) Compute global drift score
            6) Generate Evidently report

        Args:
            df_ref: Reference dataset
            df_current: Current dataset
            p_value_threshold: Statistical threshold
            strict: Raise error if drift detected

        Returns:
            Drift result dictionary
    """

    issues: List[Dict[str, Any]] = []

    try:
        if df_ref.empty or df_current.empty:
            raise ValidationError("Empty datasets provided")

        drift_flags: List[bool] = []

        ## EMBEDDING DRIFT
        ref_emb = _compute_embedding_stats(df_ref)
        cur_emb = _compute_embedding_stats(df_current)

        for col in ref_emb.columns:
            if col not in cur_emb.columns:
                continue

            p_value = _detect_numeric_drift(
                ref_emb[col],
                cur_emb[col],
                col,
                p_value_threshold,
                issues,
            )
            drift_flags.append(p_value < p_value_threshold)

        ## TEXT / CHUNK DRIFT
        ref_text = compute_text_stats(df_ref)
        cur_text = compute_text_stats(df_current)

        for col in ref_text.columns:
            if col not in cur_text.columns:
                continue

            p_value = _detect_numeric_drift(
                ref_text[col],
                cur_text[col],
                col,
                p_value_threshold,
                issues,
            )
            drift_flags.append(p_value < p_value_threshold)

        ## SOURCE DRIFT
        if "source" in df_ref.columns and "source" in df_current.columns:
            p_value = _detect_categorical_drift(
                df_ref["source"],
                df_current["source"],
                "source",
                p_value_threshold,
                issues,
            )
            drift_flags.append(p_value < p_value_threshold)

        ## MIME TYPE DRIFT
        if "mime_type" in df_ref.columns and "mime_type" in df_current.columns:
            p_value = _detect_categorical_drift(
                df_ref["mime_type"],
                df_current["mime_type"],
                "mime_type",
                p_value_threshold,
                issues,
            )
            drift_flags.append(p_value < p_value_threshold)

        ## GLOBAL SCORE
        drift_score = 1.0 - (sum(drift_flags) / len(drift_flags)) if drift_flags else 1.0

        errors = [i for i in issues if i["level"] == "error"]

        result = {
            "is_drift_ok": len(errors) == 0,
            "errors": len(errors),
            "warnings": len(issues) - len(errors),
            "drift_score": drift_score,
            "issues": issues,
        }

        ## EVIDENTLY REPORT
        try:
            report_paths = generate_evidently_report(df_ref, df_current)
            result["evidently_report"] = report_paths
        except Exception as e:
            logger.warning(f"Evidently failed: {e}")

        logger.info(f"Drift score: {drift_score}")

        if strict and drift_score < 1.0:
            raise ValidationError("Data drift detected")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        raise DataError("Data drift pipeline failed") from exc