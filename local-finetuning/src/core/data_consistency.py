'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data consistency for local finetuning: NLP + dataset + training parameters."
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger
from src.utils.data_utils import (
    normalize_data,
    validate_schema,
    validate_types,
    check_business_rules,
    compute_quality_score,
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

    issue = {
        "rule": rule,
        "level": level,
        "message": message,
        "details": details or {},
    }

    issues.append(issue)

    ## Log issue
    if level == "error":
        logger.error(f"{rule} - {message}")
    else:
        logger.warning(f"{rule} - {message}")

## ============================================================
## VALIDATIONS
## ============================================================
def _validate_text(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate text input

        Args:
            data: Input data
            issues: Issue list
    """

    text = data.get("text", "")

    ## Normalize text
    normalized = normalize_data({"text": text}).get("text", "")
    data["text"] = normalized

    ## Empty check
    if not normalized:
        _add_issue(issues, "text_empty", "error", "Text is empty")

    ## Length check
    if len(normalized) < 3:
        _add_issue(issues, "text_short", "warning", "Text too short")

def _validate_dataset(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate dataset structure

        Args:
            data: Input data
            issues: Issue list
    """

    dataset = data.get("dataset")

    ## Missing dataset
    if dataset is None:
        _add_issue(issues, "dataset_missing", "error", "Dataset is required")
        return

    ## Type check
    if not isinstance(dataset, list):
        _add_issue(issues, "dataset_type", "error", "Dataset must be a list")
        return

    ## Empty dataset
    if len(dataset) == 0:
        _add_issue(issues, "dataset_empty", "error", "Dataset is empty")
        return

    ## Validate first sample structure
    sample = dataset[0]

    if not isinstance(sample, dict):
        _add_issue(issues, "dataset_sample_type", "error", "Dataset items must be dict")
        return

    if "text" not in sample:
        _add_issue(issues, "dataset_text_missing", "error", "Dataset item missing text field")

    if "label" not in sample:
        _add_issue(issues, "dataset_label_missing", "error", "Dataset item missing label field")

def _validate_model(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate model name

        Args:
            data: Input data
            issues: Issue list
    """

    model_name = data.get("model_name")

    ## Missing
    if not model_name:
        _add_issue(issues, "model_missing", "error", "Model name is required")
        return

    ## Type
    if not isinstance(model_name, str):
        _add_issue(issues, "model_type", "error", "Model name must be string")

def _validate_training_params(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate training parameters

        Args:
            data: Input data
            issues: Issue list
    """

    batch_size = data.get("batch_size")
    epochs = data.get("epochs")

    ## Batch size
    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            _add_issue(issues, "batch_size", "error", "batch_size must be positive int")

    ## Epochs
    if epochs is not None:
        if not isinstance(epochs, int) or epochs <= 0:
            _add_issue(issues, "epochs", "error", "epochs must be positive int")

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

    for s in validate_schema(data):
        _add_issue(issues, s["rule"], s["level"], s["message"])

    for t in validate_types(data):
        _add_issue(issues, t["rule"], t["level"], t["message"])

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
        _add_issue(issues, r["rule"], r["level"], r["message"])

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

    return compute_quality_score(data)

## ============================================================
## MAIN ENTRYPOINT
## ============================================================
def run_data_consistency(
    data: Dict[str, Any],
    strict: bool = False,
) -> Dict[str, Any]:
    """
        Run full consistency pipeline

        Args:
            data: Input data
            strict: Raise error if inconsistency

        Returns:
            Result dictionary
    """

    issues: List[Dict[str, Any]] = []

    try:
        ## Normalize
        data = normalize_data(data)

        ## Validate text
        _validate_text(data, issues)

        ## Validate dataset
        _validate_dataset(data, issues)

        ## Validate model
        _validate_model(data, issues)

        ## Validate training params
        _validate_training_params(data, issues)

        ## Validate structure
        _validate_structure(data, issues)

        ## Business rules
        _validate_business(data, issues)

        ## Quality score
        quality_score = _compute_quality(data)

        errors = [i for i in issues if i["level"] == "error"]

        result = {
            "is_consistent": len(errors) == 0,
            "errors": len(errors),
            "warnings": len(issues) - len(errors),
            "quality_score": quality_score,
            "issues": issues,
        }

        if strict and errors:
            raise ValidationError("Data consistency failed")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        raise DataError("Consistency pipeline failed") from exc