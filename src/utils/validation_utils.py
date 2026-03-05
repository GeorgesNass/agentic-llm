'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Validation and coercion helpers for autonomous-ai-platform."
'''

from __future__ import annotations

from typing import Any, Dict

from src.core.errors import ValidationError
from src.utils.safe_utils import _safe_str

## ============================================================
## VALIDATION HELPERS
## ============================================================
def _must_be_dict(value: Any, field: str) -> Dict[str, Any]:
    """
        Ensure payload is a dict

        Args:
            value: Any input
            field: Field name

        Returns:
            Dict
    """

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise ValidationError(
            message="Field must be an object/dict",
            error_code="validation_error",
            details={"field": field, "type": str(type(value))},
            origin="executor",
            cause=None,
            http_status=400,
            is_retryable=False,
        )

    return value

def _require_str(value: Any, field: str) -> str:
    """
        Validate that a value is a non-empty string

        Args:
            value: Input value
            field: Field name

        Returns:
            Validated string
    """

    if value is None:
        raise ValidationError(
            message="Missing required field",
            error_code="validation_error",
            details={"field": field},
            origin="tools",
            cause=None,
            http_status=400,
            is_retryable=False,
        )

    text = str(value).strip()
    if not text:
        raise ValidationError(
            message="Field must be a non-empty string",
            error_code="validation_error",
            details={"field": field},
            origin="tools",
            cause=None,
            http_status=400,
            is_retryable=False,
        )

    return text

def _require_int(
    value: Any,
    field: str,
    *,
    min_value: int = 0,
    max_value: int = 10_000,
) -> int:
    """
        Validate that a value is an integer in a range

        Args:
            value: Input value
            field: Field name
            min_value: Minimum value
            max_value: Maximum value

        Returns:
            Integer
    """

    if value is None:
        raise ValidationError(
            message="Missing required field",
            error_code="validation_error",
            details={"field": field},
            origin="tools",
            cause=None,
            http_status=400,
            is_retryable=False,
        )

    try:
        n = int(value)
    except Exception as exc:
        raise ValidationError(
            message="Field must be an integer",
            error_code="validation_error",
            details={"field": field, "value": _safe_str(value)},
            origin="tools",
            cause=exc,
            http_status=400,
            is_retryable=False,
        ) from exc

    if n < min_value or n > max_value:
        raise ValidationError(
            message="Integer out of range",
            error_code="validation_error",
            details={"field": field, "value": n, "min": min_value, "max": max_value},
            origin="tools",
            cause=None,
            http_status=400,
            is_retryable=False,
        )

    return n

def _optional_bool(value: Any, default: bool = False) -> bool:
    """
        Parse an optional boolean from any value

        Args:
            value: Input value
            default: Default boolean if missing

        Returns:
            Boolean
    """

    if value is None:
        return default

    if isinstance(value, bool):
        return value

    raw = str(value).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}

def _must_be_non_empty(text: str, field: str) -> str:
    """
        Validate non-empty string

        Args:
            text: Input string
            field: Field name

        Returns:
            Clean string
    """

    ## Normalize and strip input
    value = str(text).strip()

    ## Reject empty values
    if not value:
        raise ValidationError(
            message="Field must be non-empty",
            error_code="validation_error",
            details={"field": field},
            origin="reasoning",
            cause=None,
            http_status=400,
            is_retryable=False,
        )

    return value

def _clamp_float(value: Any, default: float = 0.5) -> float:
    """
        Clamp float to range 0..1

        Args:
            value: Any input
            default: Fallback value

        Returns:
            Float in [0,1]
    """

    try:
        f = float(value)
    except Exception:
        return default

    ## Clamp boundaries
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f

def _safe_int(value: Any, default: int = 0) -> int:
    """
        Safe int conversion

        Args:
            value: Any input
            default: Default integer

        Returns:
            Integer
    """

    try:
        return int(value)
    except Exception:
        return default

def _enforce_budget(max_steps: int, planned_steps: int) -> None:
    """
        Enforce execution budget limits

        Args:
            max_steps: Max steps allowed
            planned_steps: Steps requested

        Returns:
            None
    """

    if planned_steps > max_steps:
        raise ValidationError(
            message="Plan exceeds max execution steps",
            error_code="validation_error",
            details={"planned_steps": planned_steps, "max_steps": max_steps},
            origin="executor",
            cause=None,
            http_status=400,
            is_retryable=False,
        )