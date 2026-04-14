'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data consistency for LLM proxy gateway: prompt, messages, model, params, tokens and request coherence"
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger
from src.utils.data_utils import (
    normalize_data,
    validate_schema,
    validate_types,
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
            level: Severity level
            message: Description
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

    ## Log depending on severity
    if level == "error":
        logger.error(f"{rule} - {message}")
    else:
        logger.warning(f"{rule} - {message}")

## ============================================================
## PROMPT VALIDATION
## ============================================================
def _validate_prompt(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate prompt or messages

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    prompt = data.get("prompt")
    messages = data.get("messages")

    ## Must have at least one input
    if not prompt and not messages:
        _add_issue(
            issues,
            "prompt_missing",
            "error",
            "Either prompt or messages must be provided",
        )

    ## Normalize + validate prompt
    if prompt:
        normalized = normalize_data({"prompt": prompt}).get("prompt", "")
        data["prompt"] = normalized

        if not normalized:
            _add_issue(issues, "prompt_empty", "error", "Prompt is empty")

    ## Validate chat messages structure
    if messages:
        if not isinstance(messages, list):
            _add_issue(issues, "messages_type", "error", "Messages must be a list")
            return

        for idx, msg in enumerate(messages):

            ## Each message must be a dict
            if not isinstance(msg, dict):
                _add_issue(
                    issues,
                    "message_format",
                    "error",
                    "Message must be dict",
                    {"index": idx},
                )
                continue

            role = msg.get("role")
            content = msg.get("content")

            ## Validate role
            if role not in ["system", "user", "assistant"]:
                _add_issue(
                    issues,
                    "message_role",
                    "error",
                    "Invalid role",
                    {"index": idx},
                )

            ## Validate content
            if not isinstance(content, str) or not content.strip():
                _add_issue(
                    issues,
                    "message_content",
                    "error",
                    "Content must be non-empty string",
                    {"index": idx},
                )

def _validate_model(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate model field

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    model = data.get("model")

    ## Model is mandatory
    if not model:
        _add_issue(issues, "model_missing", "error", "Model is required")
        return

    ## Must be string
    if not isinstance(model, str):
        _add_issue(issues, "model_type", "error", "Model must be string")

def _validate_params(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate generation parameters

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    max_tokens = data.get("max_tokens")
    temperature = data.get("temperature")
    top_p = data.get("top_p")

    ## Validate max_tokens
    if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
        _add_issue(issues, "max_tokens", "error", "Invalid max_tokens")

    ## Validate temperature range
    if temperature is not None and (
        not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2
    ):
        _add_issue(issues, "temperature", "error", "Invalid temperature")

    ## Validate top_p range
    if top_p is not None and (
        not isinstance(top_p, (int, float)) or top_p <= 0 or top_p > 1
    ):
        _add_issue(issues, "top_p", "error", "Invalid top_p")

def _validate_structure(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate schema and types

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    ## Schema validation
    for s in validate_schema(data):
        _add_issue(issues, s["rule"], s["level"], s["message"])

    ## Type validation
    for t in validate_types(data):
        _add_issue(issues, t["rule"], t["level"], t["message"])

## ============================================================
## QUALITY
## ============================================================
def _compute_quality(
    data: Dict[str, Any],
) -> float:
    """
        Compute quality score

        Args:
            data: Input data

        Returns:
            float
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
        Run data consistency pipeline for LLM gateway

        Args:
            data: Input data
            strict: Raise error if inconsistency

        Returns:
            Dict[str, Any]
    """

    issues: List[Dict[str, Any]] = []

    try:
        ## Normalize input payload
        data = normalize_data(data)

        ## Validate prompt/messages
        _validate_prompt(data, issues)

        ## Validate model
        _validate_model(data, issues)

        ## Validate generation params
        _validate_params(data, issues)

        ## Validate schema/types
        _validate_structure(data, issues)

        ## Compute quality score
        quality_score = _compute_quality(data)

        errors = [i for i in issues if i["level"] == "error"]

        result = {
            "is_consistent": len(errors) == 0,
            "errors": len(errors),
            "warnings": len(issues) - len(errors),
            "quality_score": quality_score,
            "issues": issues,
        }

        ## Strict mode blocks pipeline
        if strict and errors:
            raise ValidationError("Data consistency failed")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        raise DataError("Consistency pipeline failed") from exc