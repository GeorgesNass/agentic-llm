'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized custom exceptions and structured helpers for the local finetuning pipeline."
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
ERROR_CODE_RESOURCE_NOT_FOUND = "resource_not_found"
ERROR_CODE_DATA = "data_error"
ERROR_CODE_DEPENDENCY = "dependency_error"
ERROR_CODE_MODEL = "model_error"
ERROR_CODE_TRAINING = "training_error"
ERROR_CODE_GPU_RESOURCE = "gpu_resource_error"
ERROR_CODE_EXTERNAL_SERVICE = "external_service_error"
ERROR_CODE_PIPELINE = "pipeline_error"
ERROR_CODE_INTERNAL = "internal_error"

## ============================================================
## BASE EXCEPTION
## ============================================================
class LocalFinetuningError(RuntimeError):
    """
        Base exception for the local finetuning pipeline

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
class ConfigurationError(LocalFinetuningError):
    """
        Raised when application configuration is invalid
    """

class DataError(LocalFinetuningError):
    """
        Raised when dataset files are missing or invalid
    """

class ValidationError(LocalFinetuningError):
    """
        Raised when an input payload or parameter is invalid
    """

class ResourceNotFoundError(LocalFinetuningError):
    """
        Raised when a required file, folder or artifact is missing
    """

class DependencyError(LocalFinetuningError):
    """
        Raised when a required package or runtime dependency is missing
    """

class ModelError(LocalFinetuningError):
    """
        Raised when model or tokenizer loading or saving fails
    """

class TrainingError(LocalFinetuningError):
    """
        Raised when the finetuning process fails
    """

class GPUResourceError(LocalFinetuningError):
    """
        Raised when GPU or memory resources are insufficient
    """
    
class ExternalServiceError(LocalFinetuningError):
    """
        Raised when an external provider or remote service fails
    """

class PipelineError(LocalFinetuningError):
    """
        Raised when pipeline orchestration fails
    """

class UnknownLocalFinetuningError(LocalFinetuningError):
    """
        Raised when an unexpected exception must be normalized
    """

## ============================================================
## GENERIC HELPERS
## ============================================================
def raise_project_error(
    exc_type: Type[LocalFinetuningError],
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
            LocalFinetuningError: Always
    """

    ## Build a normalized payload
    payload = details.copy() if details else {}

    ## Attach original cause metadata when available
    if cause is not None:
        payload["cause_message"] = str(cause)
        payload["cause_type"] = cause.__class__.__name__

    ## Emit a structured error log
    logger.error(
        "Local finetuning error | type=%s | code=%s | message=%s | "
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
    exc_type: Type[LocalFinetuningError],
    message: str,
    error_code: str,
    details: Optional[Dict[str, Any]] = None,
    is_retryable: bool = False,
) -> LocalFinetuningError:
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
) -> UnknownLocalFinetuningError:
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
        "Unhandled local-finetuning exception | type=%s | details=%s",
        exc.__class__.__name__,
        payload,
    )
    logger.debug("Unhandled traceback", exc_info=True)

    ## Return a normalized unknown project error
    return UnknownLocalFinetuningError(
        message="An unexpected local-finetuning error occurred",
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

def log_and_raise_missing_raw_data(raw_dir: Path) -> None:
    """
        Log and raise a data error when no raw dataset file is found

        High-level workflow:
            1) Build the original explanatory message
            2) Log the missing dataset directory
            3) Raise a normalized data error

        Args:
            raw_dir: Directory where raw dataset files are expected

        Raises:
            DataError: Always
    """

    ## Keep the original explicit message style
    message = (
        "No raw data file found in: "
        f"{raw_dir}. "
        "Expected one file with extension: .csv, .jsonl, or .json. "
        "Fix: put your dataset into data/raw/ (or set RAW_DATA_DIR in .env)."
    )

    ## Emit a direct data log
    logger.error(message)

    ## Raise the data error
    raise DataError(
        message=message,
        error_code=ERROR_CODE_DATA,
        details={"raw_dir": str(raw_dir)},
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

def log_and_raise_dependency_error(
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise a dependency error

        Args:
            message: Human-readable dependency error message
            details: Optional dependency context
            cause: Original exception if available

        Raises:
            DependencyError: Always
    """

    ## Raise a structured dependency error
    raise_project_error(
        exc_type=DependencyError,
        message=message,
        error_code=ERROR_CODE_DEPENDENCY,
        details=details,
        cause=cause,
        is_retryable=False,
    )

def log_and_raise_model_error(
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise a model error

        Args:
            message: Human-readable model error message
            details: Optional model context
            cause: Original exception if available

        Raises:
            ModelError: Always
    """

    ## Raise a structured model error
    raise_project_error(
        exc_type=ModelError,
        message=message,
        error_code=ERROR_CODE_MODEL,
        details=details,
        cause=cause,
        is_retryable=False,
    )

def log_and_raise_training_error(
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise a training error

        Args:
            message: Human-readable training error message
            details: Optional training context
            cause: Original exception if available

        Raises:
            TrainingError: Always
    """

    ## Raise a structured training error
    raise_project_error(
        exc_type=TrainingError,
        message=message,
        error_code=ERROR_CODE_TRAINING,
        details=details,
        cause=cause,
        is_retryable=False,
    )

def log_and_raise_gpu_resource_error(
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise a GPU resource error

        Args:
            message: Human-readable GPU resource error message
            details: Optional hardware context
            cause: Original exception if available

        Raises:
            GPUResourceError: Always
    """

    ## Raise a structured GPU resource error
    raise_project_error(
        exc_type=GPUResourceError,
        message=message,
        error_code=ERROR_CODE_GPU_RESOURCE,
        details=details,
        cause=cause,
        is_retryable=True,
    )

def log_and_raise_external_service_error(
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise an external service error

        Args:
            message: Human-readable external service error message
            details: Optional service context
            cause: Original exception if available

        Raises:
            ExternalServiceError: Always
    """

    ## Raise a structured external service error
    raise_project_error(
        exc_type=ExternalServiceError,
        message=message,
        error_code=ERROR_CODE_EXTERNAL_SERVICE,
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