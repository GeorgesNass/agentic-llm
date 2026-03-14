'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized custom exceptions and structured helpers for the RAG Drive GCP pipeline."
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("errors")

## ============================================================
## ERROR CODES
## ============================================================
ERROR_CODE_CONFIGURATION = "configuration_error"
ERROR_CODE_VALIDATION = "validation_error"
ERROR_CODE_DATA = "data_error"
ERROR_CODE_RETRIEVAL = "retrieval_error"
ERROR_CODE_EMBEDDING = "embedding_error"
ERROR_CODE_STORAGE = "storage_error"
ERROR_CODE_RESOURCE_NOT_FOUND = "resource_not_found"
ERROR_CODE_EXTERNAL_SERVICE = "external_service_error"
ERROR_CODE_PIPELINE = "pipeline_error"
ERROR_CODE_INTERNAL = "internal_error"

## ============================================================
## BASE EXCEPTION
## ============================================================
class RagDriveGcpError(RuntimeError):
    """
        Base exception for the RAG Drive GCP pipeline

        High-level workflow:
            1) Normalize all project-specific failures
            2) Store structured context for debugging
            3) Support clean wrapping of lower-level exceptions

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

## ============================================================
## CUSTOM EXCEPTIONS
## ============================================================
class ConfigurationError(RagDriveGcpError):
    """
        Raised when application configuration is invalid
    """

class ValidationError(RagDriveGcpError):
    """
        Raised when an input payload or parameter is invalid
    """

class DataError(RagDriveGcpError):
    """
        Raised when data loading or parsing fails
    """

class RetrievalError(RagDriveGcpError):
    """
        Raised when retrieval or search fails
    """

class EmbeddingError(RagDriveGcpError):
    """
        Raised when embedding generation fails
    """

class StorageError(RagDriveGcpError):
    """
        Raised when local or remote storage operations fail
    """

class ResourceNotFoundError(RagDriveGcpError):
    """
        Raised when a required file, folder or artifact is missing
    """

class ExternalServiceError(RagDriveGcpError):
    """
        Raised when an external provider or remote service fails
    """

class PipelineError(RagDriveGcpError):
    """
        Raised when pipeline orchestration fails
    """

class UnknownRagDriveGcpError(RagDriveGcpError):
    """
        Raised when an unexpected exception must be normalized
    """

## ============================================================
## GENERIC HELPERS
## ============================================================
def raise_project_error(
    exc_type: Type[RagDriveGcpError],
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
            2) Add original cause metadata when available
            3) Log the error in a consistent format
            4) Raise the target project exception

        Args:
            exc_type: Exception class to raise
            message: Human-readable error message
            error_code: Normalized application error code
            details: Optional contextual details
            cause: Original exception if available
            is_retryable: Whether retry may succeed

        Raises:
            RagDriveGcpError: Always
    """

    ## Build a normalized payload
    payload = details.copy() if details else {}

    ## Attach original cause metadata when available
    if cause is not None:
        payload["cause_message"] = str(cause)
        payload["cause_type"] = cause.__class__.__name__

    ## Emit a structured error log
    logger.error(
        "RAG Drive GCP error | type=%s | code=%s | message=%s | "
        "retryable=%s | details=%s",
        exc_type.__name__,
        error_code,
        message,
        is_retryable,
        payload,
    )

    ## Raise the target project exception
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
    exc_type: Type[RagDriveGcpError],
    message: str,
    error_code: str,
    details: Optional[Dict[str, Any]] = None,
    is_retryable: bool = False,
) -> RagDriveGcpError:
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
) -> UnknownRagDriveGcpError:
    """
        Normalize an unexpected exception into a project-specific error

        High-level workflow:
            1) Build a safe execution context
            2) Preserve the original exception metadata
            3) Log the unhandled failure
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
    logger.error(
        "Unhandled rag-drive-gcp exception | type=%s | details=%s",
        exc.__class__.__name__,
        payload,
    )
    logger.debug("Unhandled traceback", exc_info=True)

    ## Return a normalized unknown project error
    return UnknownRagDriveGcpError(
        message="An unexpected rag-drive-gcp error occurred",
        error_code=ERROR_CODE_INTERNAL,
        details=payload,
        cause=exc,
        is_retryable=False,
    )

## ============================================================
## SPECIALIZED HELPERS
## ============================================================
def log_and_raise_missing_env(vars_missing: List[str]) -> None:
    """
        Log and raise a configuration error for missing environment
        variables

        Args:
            vars_missing: List of missing environment variable names

        Raises:
            ConfigurationError: Always
    """

    ## Keep the original explicit message style
    message = (
        "Missing environment variables (placeholders detected): "
        + ", ".join(vars_missing)
    )

    ## Emit a direct configuration log
    logger.error(message)

    ## Raise the configuration error
    raise ConfigurationError(
        message=message,
        error_code=ERROR_CODE_CONFIGURATION,
        details={"missing_variables": vars_missing},
        is_retryable=False,
    )

def log_and_raise_missing_path(
    path: str | Path,
    *,
    resource_name: str = "Required resource",
) -> None:
    """
        Log and raise a missing resource error

        Args:
            path: Missing filesystem path
            resource_name: Human-readable resource label

        Raises:
            ResourceNotFoundError: Always
    """

    ## Normalize the path for logs and payloads
    normalized_path = str(Path(path))

    ## Raise a structured missing resource error
    raise_project_error(
        exc_type=ResourceNotFoundError,
        message=f"{resource_name} not found",
        error_code=ERROR_CODE_RESOURCE_NOT_FOUND,
        details={"path": normalized_path},
        is_retryable=False,
    )

def log_and_raise_validation_error(
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
        Log and raise a validation error

        Args:
            message: Human-readable validation error message
            details: Optional validation context

        Raises:
            ValidationError: Always
    """

    ## Raise a structured validation error
    raise_project_error(
        exc_type=ValidationError,
        message=message,
        error_code=ERROR_CODE_VALIDATION,
        details=details,
        is_retryable=False,
    )

def log_and_raise_data_error(
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise a data error

        Args:
            message: Human-readable data error message
            details: Optional data context
            cause: Original exception if available

        Raises:
            DataError: Always
    """

    ## Raise a structured data error
    raise_project_error(
        exc_type=DataError,
        message=message,
        error_code=ERROR_CODE_DATA,
        details=details,
        cause=cause,
        is_retryable=False,
    )

def log_and_raise_retrieval_error(
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise a retrieval error

        Args:
            message: Human-readable retrieval error message
            details: Optional retrieval context
            cause: Original exception if available

        Raises:
            RetrievalError: Always
    """

    ## Raise a structured retrieval error
    raise_project_error(
        exc_type=RetrievalError,
        message=message,
        error_code=ERROR_CODE_RETRIEVAL,
        details=details,
        cause=cause,
        is_retryable=True,
    )

def log_and_raise_embedding_error(
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise an embedding error

        Args:
            message: Human-readable embedding error message
            details: Optional embedding context
            cause: Original exception if available

        Raises:
            EmbeddingError: Always
    """

    ## Raise a structured embedding error
    raise_project_error(
        exc_type=EmbeddingError,
        message=message,
        error_code=ERROR_CODE_EMBEDDING,
        details=details,
        cause=cause,
        is_retryable=True,
    )

def log_and_raise_storage_error(
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise a storage error

        Args:
            message: Human-readable storage error message
            details: Optional storage context
            cause: Original exception if available

        Raises:
            StorageError: Always
    """

    ## Raise a structured storage error
    raise_project_error(
        exc_type=StorageError,
        message=message,
        error_code=ERROR_CODE_STORAGE,
        details=details,
        cause=cause,
        is_retryable=True,
    )

def log_and_raise_pipeline_error(
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise a pipeline error

        Args:
            message: Human-readable pipeline error message
            details: Optional pipeline context
            cause: Original exception if available

        Raises:
            PipelineError: Always
    """

    ## Raise a structured pipeline error
    raise_project_error(
        exc_type=PipelineError,
        message=message,
        error_code=ERROR_CODE_PIPELINE,
        details=details,
        cause=cause,
        is_retryable=False,
    )