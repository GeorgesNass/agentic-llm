'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data consistency for local quantization: NLP + model + quantization parameters."
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
        Validate prompt text

        Args:
            data: Input data
            issues: Issue list
    """

    text = data.get("text", "")

    ## Normalize
    normalized = normalize_data({"text": text}).get("text", "")
    data["text"] = normalized

    ## Empty
    if not normalized:
        _add_issue(issues, "text_empty", "error", "Text is empty")

    ## Too short
    if len(normalized) < 3:
        _add_issue(issues, "text_short", "warning", "Text too short")

def _validate_model(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate model parameters

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

def _validate_quantization(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate quantization parameters

        Args:
            data: Input data
            issues: Issue list
    """

    bits = data.get("bits")
    dtype = data.get("dtype")
    device = data.get("device")

    ## Bits check
    if bits is not None:
        if bits not in [2, 3, 4, 8]:
            _add_issue(issues, "quant_bits", "error", "Invalid quantization bits")

    ## Dtype check
    if dtype is not None:
        if dtype not in ["float16", "bfloat16", "int8"]:
            _add_issue(issues, "quant_dtype", "error", "Invalid dtype")

    ## Device check
    if device is not None:
        if device not in ["cpu", "cuda"]:
            _add_issue(issues, "quant_device", "error", "Invalid device")

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

        ## NLP
        _validate_text(data, issues)

        ## Model
        _validate_model(data, issues)

        ## Quantization
        _validate_quantization(data, issues)

        ## Structure
        _validate_structure(data, issues)

        ## Business
        _validate_business(data, issues)

        ## Quality
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