'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized custom exceptions and structured error handling helpers for autonomous-ai-platform."
'''

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type

from fastapi import Request
from fastapi.responses import JSONResponse

from src.utils.logging_utils import get_logger
from src.utils.request_utils import _get_request_id
from src.utils.safe_utils import _safe_str, _safe_json

## ============================================================
## LOGGER
## ============================================================
logger = get_logger(__name__)

## ============================================================
## ERROR CODES
## ============================================================
ERROR_CODE_BAD_REQUEST = "bad_request"
ERROR_CODE_UNAUTHORIZED = "unauthorized"
ERROR_CODE_FORBIDDEN = "forbidden"
ERROR_CODE_NOT_FOUND = "not_found"
ERROR_CODE_TIMEOUT = "timeout"
ERROR_CODE_RATE_LIMIT = "rate_limit"
ERROR_CODE_DEPENDENCY = "dependency_error"
ERROR_CODE_VALIDATION = "validation_error"
ERROR_CODE_CONFIGURATION = "configuration_error"
ERROR_CODE_LLM_PROVIDER = "llm_provider_error"
ERROR_CODE_TOOL = "tool_execution_error"
ERROR_CODE_RETRIEVAL = "retrieval_error"
ERROR_CODE_SQL = "sql_execution_error"
ERROR_CODE_ORCHESTRATION = "orchestration_error"
ERROR_CODE_EVALUATION = "evaluation_error"
ERROR_CODE_MONITORING = "monitoring_error"
ERROR_CODE_STORAGE = "storage_error"
ERROR_CODE_INTERNAL = "internal_error"

## ============================================================
## STRUCTURED ERROR PAYLOAD
## ============================================================
@dataclass(frozen=True)
class ErrorPayload:
    """
        Structured error payload

        Args:
            error_code: Normalized error code
            message: Human-readable message
            details: Diagnostic details dict
            origin: Optional origin component
            cause_type: Optional cause exception type
    """

    error_code: str
    message: str
    details: Dict[str, Any]
    origin: str
    cause_type: str

## ============================================================
## BASE ERROR
## ============================================================
class AutonomousAIPlatformError(Exception):
    """
        Base exception for autonomous-ai-platform

        Args:
            message: Human-readable error message
            error_code: Normalized error code
            details: Optional diagnostic details
            origin: Component where the error happened
            cause: Original exception if any
            http_status: HTTP status code hint
            is_retryable: Whether this error is retryable
    """

    def __init__(
        self,
        message: str,
        error_code: str = ERROR_CODE_INTERNAL,
        details: Optional[Dict[str, Any]] = None,
        origin: str = "unknown",
        cause: Optional[Exception] = None,
        http_status: int = 400,
        is_retryable: bool = False,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.origin = origin
        self.cause = cause
        self.http_status = http_status
        self.is_retryable = is_retryable
        super().__init__(message)

    def to_payload(self) -> ErrorPayload:
        """
            Convert exception to structured payload

            Returns:
                ErrorPayload
        """

        cause_type = self.cause.__class__.__name__ if self.cause else ""
        return ErrorPayload(
            error_code=self.error_code,
            message=self.message,
            details=self.details,
            origin=self.origin,
            cause_type=cause_type,
        )

## ============================================================
## DOMAIN ERRORS
## ============================================================
class ConfigurationError(AutonomousAIPlatformError):
    """Configuration and environment errors"""

class ValidationError(AutonomousAIPlatformError):
    """Input validation errors (beyond Pydantic)"""

class UnauthorizedError(AutonomousAIPlatformError):
    """Auth errors"""

class ForbiddenError(AutonomousAIPlatformError):
    """Permission errors"""

class NotFoundError(AutonomousAIPlatformError):
    """Missing resource errors"""

class TimeoutError(AutonomousAIPlatformError):
    """Timeout errors"""

class RateLimitError(AutonomousAIPlatformError):
    """Rate limiting errors"""

class DependencyError(AutonomousAIPlatformError):
    """External dependency errors (network, APIs, services)"""

class StorageError(AutonomousAIPlatformError):
    """Filesystem / object storage errors"""

class LlmProviderError(AutonomousAIPlatformError):
    """Provider client errors (OpenAI/xAI/local runtime)"""

class ToolExecutionError(AutonomousAIPlatformError):
    """Tool execution failures"""

class RetrievalError(AutonomousAIPlatformError):
    """Vector store / retrieval errors"""

class SqlExecutionError(AutonomousAIPlatformError):
    """SQLite execution errors"""

class OrchestrationError(AutonomousAIPlatformError):
    """Agentic orchestration errors"""

class EvaluationError(AutonomousAIPlatformError):
    """Evaluation pipeline errors"""

class MonitoringError(AutonomousAIPlatformError):
    """Prometheus/Grafana exporter errors"""

class PlatformError(AutonomousAIPlatformError):
    """ Generic platform-level error. Used when an error does not belong to a specific domain (configuration, retrieval, evaluation, etc.)."""
    
## ============================================================
## RAISE HELPERS (NO RAW ERRORS)
## ============================================================
def raise_platform_error(
    exc_type: Type[AutonomousAIPlatformError],
    message: str,
    *,
    error_code: str,
    details: Optional[Dict[str, Any]] = None,
    origin: str = "unknown",
    cause: Optional[Exception] = None,
    http_status: int = 400,
    is_retryable: bool = False,
) -> None:
    """
        Raise a structured platform exception

        Args:
            exc_type: Exception class to raise
            message: Human-readable message
            error_code: Normalized error code
            details: Diagnostic details dict
            origin: Component where the error happened
            cause: Original exception
            http_status: HTTP status code hint
            is_retryable: Retryability hint

        Returns:
            None

        Raises:
            AutonomousAIPlatformError
    """

    raise exc_type(
        message=message,
        error_code=error_code,
        details=details,
        origin=origin,
        cause=cause,
        http_status=http_status,
        is_retryable=is_retryable,
    )

def wrap_exception(
    exc: Exception,
    *,
    exc_type: Type[AutonomousAIPlatformError],
    message: str,
    error_code: str,
    origin: str,
    details: Optional[Dict[str, Any]] = None,
    http_status: int = 400,
    is_retryable: bool = False,
) -> AutonomousAIPlatformError:
    """
        Wrap any exception into a structured platform exception

        Args:
            exc: Original exception
            exc_type: Target platform exception class
            message: Human-readable message
            error_code: Normalized error code
            origin: Component where the error happened
            details: Extra diagnostic details
            http_status: HTTP status hint
            is_retryable: Retryability hint

        Returns:
            Wrapped AutonomousAIPlatformError
    """

    merged_details = details or {}
    merged_details["cause_message"] = _safe_str(exc)
    merged_details["cause_type"] = exc.__class__.__name__

    return exc_type(
        message=message,
        error_code=error_code,
        details=merged_details,
        origin=origin,
        cause=exc,
        http_status=http_status,
        is_retryable=is_retryable,
    )

## ============================================================
## LOGGING HELPERS
## ============================================================
def log_structured_error(
    exc: AutonomousAIPlatformError,
    *,
    request: Optional[Request] = None,
    include_traceback: bool = False,
) -> None:
    """
        Log a structured platform error

        Args:
            exc: Platform exception
            request: Optional request object
            include_traceback: Whether to include traceback at DEBUG level

        Returns:
            None
    """

    req_id = _get_request_id(request)

    payload = exc.to_payload()
    logger.error(
        "PlatformError | request_id=%s | code=%s | origin=%s | message=%s | details=%s",
        req_id,
        payload.error_code,
        payload.origin,
        payload.message,
        _safe_json(payload.details),
    )

    ## Log traceback only in debug, never in error line
    if include_traceback and exc.cause is not None:
        logger.debug(
            "PlatformError traceback | request_id=%s | cause_type=%s\n%s",
            req_id,
            exc.cause.__class__.__name__,
            "".join(traceback.format_exception(type(exc.cause), exc.cause, exc.cause.__traceback__)),
        )

def log_unhandled_exception(
    exc: Exception,
    *,
    request: Optional[Request] = None,
) -> None:
    """
        Log an unhandled exception in a structured way

        Args:
            exc: Unexpected exception
            request: Optional request

        Returns:
            None
    """

    req_id = _get_request_id(request)

    logger.error(
        "UnhandledException | request_id=%s | type=%s | message=%s",
        req_id,
        exc.__class__.__name__,
        _safe_str(exc),
    )

    ## Full traceback only in debug
    logger.debug(
        "UnhandledException traceback | request_id=%s\n%s",
        req_id,
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )

## ============================================================
## FASTAPI EXCEPTION HANDLERS
## ============================================================
async def platform_exception_handler(
    request: Request,
    exc: AutonomousAIPlatformError,
) -> JSONResponse:
    """
        Handle AutonomousAIPlatformError exceptions in FastAPI

        Args:
            request: FastAPI request object
            exc: Platform exception

        Returns:
            JSONResponse with standardized error payload
    """

    log_structured_error(exc, request=request, include_traceback=False)

    payload = exc.to_payload()
    return JSONResponse(
        status_code=exc.http_status,
        content={
            "error": payload.error_code,
            "message": payload.message,
            "origin": payload.origin,
            "details": payload.details,
            "request_id": _get_request_id(request),
        },
    )

async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
        Handle unexpected exceptions in FastAPI

        Args:
            request: FastAPI request object
            exc: Unexpected exception

        Returns:
            JSONResponse with generic error payload
    """

    ## Convert to structured internal error for consistent API responses
    log_unhandled_exception(exc, request=request)

    return JSONResponse(
        status_code=500,
        content={
            "error": ERROR_CODE_INTERNAL,
            "message": "An unexpected error occurred",
            "origin": "fastapi",
            "details": {"cause_type": exc.__class__.__name__},
            "request_id": _get_request_id(request),
        },
    )

## ============================================================
## COMMON EXCEPTION MAPPINGS (OPTIONAL)
## ============================================================
def map_to_platform_error(
    exc: Exception,
    *,
    origin: str,
    fallback_message: str = "Unexpected error",
) -> AutonomousAIPlatformError:
    """
        Map common python exceptions to platform errors

        Args:
            exc: Original exception
            origin: Component where error happened
            fallback_message: Default message

        Returns:
            AutonomousAIPlatformError
    """

    ## NOTE
    ## Keep mapping conservative and explicit
    mapping: Tuple[Tuple[Type[Exception], Type[AutonomousAIPlatformError], str, int], ...] = (
        (FileNotFoundError, NotFoundError, ERROR_CODE_NOT_FOUND, 404),
        (PermissionError, ForbiddenError, ERROR_CODE_FORBIDDEN, 403),
        (ValueError, ValidationError, ERROR_CODE_VALIDATION, 400),
        (TimeoutError, TimeoutError, ERROR_CODE_TIMEOUT, 408),
    )

    for exc_cls, target_cls, code, status in mapping:
        if isinstance(exc, exc_cls):
            return wrap_exception(
                exc,
                exc_type=target_cls,
                message=fallback_message,
                error_code=code,
                origin=origin,
                http_status=status,
                is_retryable=code in {ERROR_CODE_TIMEOUT, ERROR_CODE_RATE_LIMIT},
            )

    return wrap_exception(
        exc,
        exc_type=AutonomousAIPlatformError,
        message=fallback_message,
        error_code=ERROR_CODE_INTERNAL,
        origin=origin,
        http_status=500,
        is_retryable=False,
    )