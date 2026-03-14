'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Custom exception hierarchy and structured helpers for the local quantization pipeline."
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Type

from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
LOGGER = get_logger(__name__)

## ============================================================
## ERROR CODES
## ============================================================
ERROR_CODE_CONFIGURATION = "configuration_error"
ERROR_CODE_DATA = "data_error"
ERROR_CODE_BACKEND = "backend_error"
ERROR_CODE_EXPORT = "export_error"
ERROR_CODE_BENCHMARK = "benchmark_error"
ERROR_CODE_PIPELINE = "pipeline_error"
ERROR_CODE_VALIDATION = "validation_error"
ERROR_CODE_RESOURCE_NOT_FOUND = "resource_not_found"
ERROR_CODE_INTERNAL = "internal_error"

## ============================================================
## EXCEPTION HIERARCHY
## ============================================================
class LocalQuantizationError(RuntimeError):
    """
        Base exception for all local quantization errors

        High-level workflow:
            1) Normalize quantization-specific failures
            2) Preserve structured context for debugging
            3) Support consistent wrapping of lower-level exceptions

        Args:
            message: Human-readable error message
            error_code: Normalized application error code
            details: Optional structured context payload
            cause: Original exception if available
            is_retryable: Whether retry may succeed
    """

    def __init__(
        self,
        message: str,
        error_code: str = ERROR_CODE_INTERNAL,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        is_retryable: bool = False,
    ) -> None:
        ## Store normalized error metadata
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        self.is_retryable = is_retryable

        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """
            Convert the exception into a structured dictionary

            Returns:
                A normalized error payload
        """

        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "cause_type": self.cause.__class__.__name__
            if self.cause
            else None,
            "is_retryable": self.is_retryable,
        }

class ConfigurationError(LocalQuantizationError):
    """
        Raised when configuration or environment variables are invalid

        Typical causes:
            - Missing required environment variables
            - Invalid enum values
            - Invalid paths defined by settings
    """

class DataError(LocalQuantizationError):
    """
        Raised when input data or model files are invalid or missing

        Typical causes:
            - Model path does not exist
            - Adapter path missing when required
            - Calibration dataset unavailable
    """

class BackendError(LocalQuantizationError):
    """
        Raised when a quantization backend is unavailable or fails

        Typical causes:
            - Missing dependency
            - Unsupported model architecture
            - Runtime conversion failure
    """

class ExportError(LocalQuantizationError):
    """
        Raised when exporting quantized artifacts fails

        Typical causes:
            - Permission issues while writing artifacts
            - Missing output directory
            - Serialization failures
    """

class BenchmarkError(LocalQuantizationError):
    """
        Raised when benchmarking fails

        Typical causes:
            - Runner unavailable for the target format
            - Invalid prompt suite configuration
            - Runtime inference errors
    """

class PipelineError(LocalQuantizationError):
    """
        Raised when a pipeline step fails

        Typical causes:
            - Unhandled runtime exception inside a pipeline step
            - Invalid step ordering or incompatible settings
    """

class ValidationError(LocalQuantizationError):
    """
        Raised when runtime validation checks fail
    """

class ResourceNotFoundError(LocalQuantizationError):
    """
        Raised when a required file or directory is missing
    """

class UnknownLocalQuantizationError(LocalQuantizationError):
    """
        Raised when an unexpected exception must be normalized
    """

## ============================================================
## GENERIC HELPERS
## ============================================================
def raise_project_error(
    exc_type: Type[LocalQuantizationError],
    message: str,
    *,
    error_code: str,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
    is_retryable: bool = False,
) -> None:
    """
        Log and raise a structured project exception

        High-level workflow:
            1) Build a normalized payload
            2) Attach original cause metadata when available
            3) Log the failure in a consistent format
            4) Raise the normalized exception

        Args:
            exc_type: Exception class to raise
            message: Human-readable error message
            error_code: Normalized application error code
            details: Optional contextual details
            cause: Original exception if available
            is_retryable: Whether retry may succeed

        Raises:
            LocalQuantizationError: Always
    """

    ## Build a normalized payload
    payload = details.copy() if details else {}

    ## Attach original cause metadata when available
    if cause is not None:
        payload["cause_message"] = str(cause)
        payload["cause_type"] = cause.__class__.__name__

    ## Emit a structured error log
    LOGGER.error(
        "Local quantization error | type=%s | code=%s | message=%s | "
        "retryable=%s | details=%s",
        exc_type.__name__,
        error_code,
        message,
        is_retryable,
        payload,
    )

    ## Raise the normalized exception
    raise exc_type(
        message=message,
        error_code=error_code,
        details=payload,
        cause=cause,
        is_retryable=is_retryable,
    )

def wrap_exception(
    exc: Exception,
    *,
    exc_type: Type[LocalQuantizationError],
    message: str,
    error_code: str,
    details: Optional[Dict[str, Any]] = None,
    is_retryable: bool = False,
) -> LocalQuantizationError:
    """
        Wrap a raw exception into a structured project exception

        High-level workflow:
            1) Preserve the original exception
            2) Merge it into the structured payload
            3) Return a normalized project error instance

        Args:
            exc: Original exception
            exc_type: Target structured exception type
            message: Human-readable error message
            error_code: Normalized application error code
            details: Optional contextual details
            is_retryable: Whether retry may succeed

        Returns:
            A structured project exception instance
    """

    ## Start from existing details when provided
    payload = details.copy() if details else {}

    ## Attach original cause metadata
    payload["cause_message"] = str(exc)
    payload["cause_type"] = exc.__class__.__name__

    ## Return a normalized wrapped exception
    return exc_type(
        message=message,
        error_code=error_code,
        details=payload,
        cause=exc,
        is_retryable=is_retryable,
    )

def log_unhandled_exception(
    exc: Exception,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> UnknownLocalQuantizationError:
    """
        Normalize an unexpected exception into a project-specific error

        High-level workflow:
            1) Build a safe execution context
            2) Preserve original exception metadata
            3) Log the unexpected failure
            4) Return a normalized unknown error

        Args:
            exc: Original unexpected exception
            context: Optional execution context

        Returns:
            A normalized unknown project exception
    """

    ## Build a safe payload from optional context
    payload = context.copy() if context else {}

    ## Attach original cause metadata
    payload["cause_message"] = str(exc)
    payload["cause_type"] = exc.__class__.__name__

    ## Log the unexpected failure
    LOGGER.error(
        "Unhandled local-quantization exception | type=%s | details=%s",
        exc.__class__.__name__,
        payload,
    )
    LOGGER.debug("Unhandled traceback", exc_info=True)

    ## Return a normalized unknown project error
    return UnknownLocalQuantizationError(
        message="An unexpected local-quantization error occurred",
        error_code=ERROR_CODE_INTERNAL,
        details=payload,
        cause=exc,
        is_retryable=False,
    )

## ============================================================
## ERROR HELPERS
## ============================================================
def log_and_raise_missing_env(
    env_key: str,
    reason: Optional[str] = None,
) -> None:
    """
        Log and raise a configuration error for a missing environment variable

        Args:
            env_key: Name of the missing environment variable
            reason: Optional explanation

        Returns:
            None
    """

    ## Build a consistent message
    message = f"Missing required environment variable: {env_key}"
    if reason:
        message = f"{message} | {reason}"

    ## Emit the error log
    LOGGER.error(message)

    ## Raise the configuration error
    raise ConfigurationError(
        message=message,
        error_code=ERROR_CODE_CONFIGURATION,
        details={"env_key": env_key, "reason": reason},
        is_retryable=False,
    )

def log_and_raise_missing_path(
    path: Path,
    context: Optional[str] = None,
) -> None:
    """
        Log and raise a data error for a missing file or directory path

        Args:
            path: Missing path
            context: Optional context description

        Returns:
            None
    """

    ## Build a consistent message
    message = f"Required path does not exist: {path}"
    if context:
        message = f"{message} | context={context}"

    ## Emit the error log
    LOGGER.error(message)

    ## Raise the data error
    raise DataError(
        message=message,
        error_code=ERROR_CODE_DATA,
        details={"path": str(path), "context": context},
        is_retryable=False,
    )

def log_and_raise_backend_unavailable(
    backend_name: str,
    reason: Optional[str] = None,
) -> None:
    """
        Log and raise a backend error when a backend is unavailable

        Args:
            backend_name: Backend identifier
            reason: Optional explanation

        Returns:
            None
    """

    ## Build a consistent message
    message = f"Quantization backend unavailable: {backend_name}"
    if reason:
        message = f"{message} | {reason}"

    ## Emit the error log
    LOGGER.error(message)

    ## Raise the backend error
    raise BackendError(
        message=message,
        error_code=ERROR_CODE_BACKEND,
        details={"backend_name": backend_name, "reason": reason},
        is_retryable=False,
    )

def log_and_raise_pipeline_error(
    step: str,
    reason: Optional[str] = None,
) -> None:
    """
        Log and raise a pipeline error for a failed pipeline step

        Args:
            step: Pipeline step name
            reason: Optional explanation

        Returns:
            None
    """

    ## Build a consistent message
    message = f"Pipeline step failed: {step}"
    if reason:
        message = f"{message} | {reason}"

    ## Emit the error log
    LOGGER.error(message)

    ## Raise the pipeline error
    raise PipelineError(
        message=message,
        error_code=ERROR_CODE_PIPELINE,
        details={"step": step, "reason": reason},
        is_retryable=False,
    )

def log_and_raise_export_error(
    artifact_name: str,
    reason: Optional[str] = None,
) -> None:
    """
        Log and raise an export error for a failed export operation

        Args:
            artifact_name: Name of the artifact being exported
            reason: Optional explanation

        Returns:
            None
    """

    ## Build a consistent message
    message = f"Export failed for artifact: {artifact_name}"
    if reason:
        message = f"{message} | {reason}"

    ## Emit the error log
    LOGGER.error(message)

    ## Raise the export error
    raise ExportError(
        message=message,
        error_code=ERROR_CODE_EXPORT,
        details={"artifact_name": artifact_name, "reason": reason},
        is_retryable=False,
    )

def log_and_raise_benchmark_error(
    benchmark_name: str,
    reason: Optional[str] = None,
) -> None:
    """
        Log and raise a benchmark error for a failed benchmarking step

        Args:
            benchmark_name: Benchmark identifier
            reason: Optional explanation

        Returns:
            None
    """

    ## Build a consistent message
    message = f"Benchmark failed: {benchmark_name}"
    if reason:
        message = f"{message} | {reason}"

    ## Emit the error log
    LOGGER.error(message)

    ## Raise the benchmark error
    raise BenchmarkError(
        message=message,
        error_code=ERROR_CODE_BENCHMARK,
        details={"benchmark_name": benchmark_name, "reason": reason},
        is_retryable=False,
    )