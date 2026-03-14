'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Custom exception hierarchy and structured error helpers for the LLM proxy gateway."
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

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
ERROR_CODE_PROVIDER = "provider_error"
ERROR_CODE_VALIDATION = "validation_error"
ERROR_CODE_PIPELINE = "pipeline_error"
ERROR_CODE_AUTHENTICATION = "authentication_error"
ERROR_CODE_RATE_LIMIT = "rate_limit_error"
ERROR_CODE_TIMEOUT = "timeout_error"
ERROR_CODE_CONNECTION = "connection_error"
ERROR_CODE_RESPONSE_PARSING = "response_parsing_error"
ERROR_CODE_RESOURCE_NOT_FOUND = "resource_not_found"
ERROR_CODE_INTERNAL = "internal_error"

## ============================================================
## EXCEPTION HIERARCHY
## ============================================================
class LLMProxyGatewayError(RuntimeError):
    """
        Base exception for all LLM proxy gateway errors

        High-level workflow:
            1) Normalize gateway-specific failures
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

class ConfigurationError(LLMProxyGatewayError):
    """
        Raised when configuration or environment variables are invalid

        Args:
            message: Human-readable error message
    """

class DataError(LLMProxyGatewayError):
    """
        Raised when input data is missing or invalid

        Args:
            message: Human-readable error message
    """

class ProviderError(LLMProxyGatewayError):
    """
        Raised when a remote provider call fails or is unavailable

        Args:
            message: Human-readable error message
    """

class ValidationError(LLMProxyGatewayError):
    """
        Raised when request validation fails at runtime

        Args:
            message: Human-readable error message
    """

class PipelineError(LLMProxyGatewayError):
    """
        Raised when a pipeline step fails

        Args:
            message: Human-readable error message
    """

class AuthenticationError(LLMProxyGatewayError):
    """
        Raised when upstream authentication fails
    """

class RateLimitError(LLMProxyGatewayError):
    """
        Raised when upstream provider rate limits the request
    """

class UpstreamTimeoutError(LLMProxyGatewayError):
    """
        Raised when an upstream provider request times out
    """

class UpstreamConnectionError(LLMProxyGatewayError):
    """
        Raised when connection to an upstream provider fails
    """

class ResponseParsingError(LLMProxyGatewayError):
    """
        Raised when the provider response cannot be parsed safely
    """

class ResourceNotFoundError(LLMProxyGatewayError):
    """
        Raised when a required file or directory is missing
    """

class UnknownLLMProxyGatewayError(LLMProxyGatewayError):
    """
        Raised when an unexpected exception must be normalized
    """

## ============================================================
## GENERIC HELPERS
## ============================================================
def raise_project_error(
    exc_type: Type[LLMProxyGatewayError],
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
            LLMProxyGatewayError: Always
    """

    ## Build a normalized payload
    payload = details.copy() if details else {}

    ## Attach original cause metadata when available
    if cause is not None:
        payload["cause_message"] = str(cause)
        payload["cause_type"] = cause.__class__.__name__

    ## Emit a structured error log
    LOGGER.error(
        "LLM proxy gateway error | type=%s | code=%s | message=%s | "
        "retryable=%s | details=%s",
        exc_type.__name__,
        error_code,
        message,
        is_retryable,
        payload,
    )

    ## Raise the normalized project exception
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
    exc_type: Type[LLMProxyGatewayError],
    message: str,
    error_code: str,
    details: Optional[Dict[str, Any]] = None,
    is_retryable: bool = False,
) -> LLMProxyGatewayError:
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
) -> UnknownLLMProxyGatewayError:
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
        "Unhandled llm-proxy-gateway exception | type=%s | details=%s",
        exc.__class__.__name__,
        payload,
    )
    LOGGER.debug("Unhandled traceback", exc_info=True)

    ## Return a normalized unknown project error
    return UnknownLLMProxyGatewayError(
        message="An unexpected llm-proxy-gateway error occurred",
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
        Log and raise a ConfigurationError for a missing environment variable

        Args:
            env_key: Name of the missing environment variable
            reason: Optional explanation

        Returns:
            None
    """

    ## Build consistent error message
    message = f"Missing required environment variable: {env_key}"
    if reason:
        message = f"{message} | {reason}"

    LOGGER.error(message)
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
        Log and raise a DataError for a missing file or directory path

        Args:
            path: Missing path
            context: Optional context description

        Returns:
            None
    """

    ## Build consistent error message
    message = f"Required path does not exist: {path}"
    if context:
        message = f"{message} | context={context}"

    LOGGER.error(message)
    raise DataError(
        message=message,
        error_code=ERROR_CODE_DATA,
        details={"path": str(path), "context": context},
        is_retryable=False,
    )

def log_and_raise_missing_file(
    path: Union[str, Path],
    context: Optional[str] = None,
) -> None:
    """
        Log and raise a DataError specifically for missing file paths

        Args:
            path: Missing file path (str or Path)
            context: Optional context description

        Returns:
            None
    """

    ## Normalize input path
    p = Path(path).expanduser().resolve()

    ## Reuse the same formatting as missing_path with explicit wording
    message = f"Required file does not exist: {p}"
    if context:
        message = f"{message} | context={context}"

    LOGGER.error(message)
    raise DataError(
        message=message,
        error_code=ERROR_CODE_DATA,
        details={"path": str(p), "context": context},
        is_retryable=False,
    )

def log_and_raise_provider_error(
    provider: str,
    reason: Optional[str] = None,
) -> None:
    """
        Log and raise a ProviderError when a provider call fails

        Args:
            provider: Provider identifier (openai, google, xai, etc.)
            reason: Optional explanation

        Returns:
            None
    """

    ## Build consistent error message
    message = f"Provider call failed: provider={provider}"
    if reason:
        message = f"{message} | {reason}"

    LOGGER.error(message)
    raise ProviderError(
        message=message,
        error_code=ERROR_CODE_PROVIDER,
        details={"provider": provider, "reason": reason},
        is_retryable=True,
    )

def log_and_raise_validation_error(
    reason: str,
    context: Optional[str] = None,
) -> None:
    """
        Log and raise a ValidationError for invalid runtime input

        Args:
            reason: Validation error reason
            context: Optional context label

        Returns:
            None
    """

    ## Build consistent error message
    message = f"Validation error: {reason}"
    if context:
        message = f"{message} | context={context}"

    LOGGER.error(message)
    raise ValidationError(
        message=message,
        error_code=ERROR_CODE_VALIDATION,
        details={"reason": reason, "context": context},
        is_retryable=False,
    )

def log_and_raise_pipeline_error(
    step: str,
    reason: Optional[str] = None,
) -> None:
    """
        Log and raise a PipelineError for a failed pipeline step

        Args:
            step: Pipeline step name
            reason: Optional explanation

        Returns:
            None
    """

    ## Build consistent error message
    message = f"Pipeline step failed: {step}"
    if reason:
        message = f"{message} | {reason}"

    LOGGER.error(message)
    raise PipelineError(
        message=message,
        error_code=ERROR_CODE_PIPELINE,
        details={"step": step, "reason": reason},
        is_retryable=False,
    )

def log_and_raise_os_error_as_data_error(
    exc: Exception,
    context: Optional[str] = None,
) -> None:
    """
        Convert OS-level and Python IO errors into a clean DataError

        Args:
            exc: Original exception
            context: Optional context label

        Returns:
            None
    """

    ## Build a normalized IO error message
    message = f"Data IO error: {type(exc).__name__} | {str(exc)}"
    if context:
        message = f"{message} | context={context}"

    LOGGER.error(message)
    raise DataError(
        message=message,
        error_code=ERROR_CODE_DATA,
        details={
            "context": context,
            "cause_type": type(exc).__name__,
            "cause_message": str(exc),
        },
        cause=exc,
        is_retryable=False,
    )

def log_and_raise_authentication_error(
    provider: str,
    reason: Optional[str] = None,
) -> None:
    """
        Log and raise an authentication error

        Args:
            provider: Provider identifier
            reason: Optional explanation
    """

    ## Build a normalized authentication error message
    message = f"Provider authentication failed: provider={provider}"
    if reason:
        message = f"{message} | {reason}"

    LOGGER.error(message)
    raise AuthenticationError(
        message=message,
        error_code=ERROR_CODE_AUTHENTICATION,
        details={"provider": provider, "reason": reason},
        is_retryable=False,
    )

def log_and_raise_rate_limit_error(
    provider: str,
    reason: Optional[str] = None,
) -> None:
    """
        Log and raise a rate limit error

        Args:
            provider: Provider identifier
            reason: Optional explanation
    """

    ## Build a normalized rate limit error message
    message = f"Provider rate limit reached: provider={provider}"
    if reason:
        message = f"{message} | {reason}"

    LOGGER.error(message)
    raise RateLimitError(
        message=message,
        error_code=ERROR_CODE_RATE_LIMIT,
        details={"provider": provider, "reason": reason},
        is_retryable=True,
    )

def log_and_raise_timeout_error(
    provider: str,
    reason: Optional[str] = None,
) -> None:
    """
        Log and raise an upstream timeout error

        Args:
            provider: Provider identifier
            reason: Optional explanation
    """

    ## Build a normalized timeout error message
    message = f"Upstream timeout: provider={provider}"
    if reason:
        message = f"{message} | {reason}"

    LOGGER.error(message)
    raise UpstreamTimeoutError(
        message=message,
        error_code=ERROR_CODE_TIMEOUT,
        details={"provider": provider, "reason": reason},
        is_retryable=True,
    )

def log_and_raise_connection_error(
    provider: str,
    reason: Optional[str] = None,
) -> None:
    """
        Log and raise an upstream connection error

        Args:
            provider: Provider identifier
            reason: Optional explanation
    """

    ## Build a normalized connection error message
    message = f"Upstream connection failed: provider={provider}"
    if reason:
        message = f"{message} | {reason}"

    LOGGER.error(message)
    raise UpstreamConnectionError(
        message=message,
        error_code=ERROR_CODE_CONNECTION,
        details={"provider": provider, "reason": reason},
        is_retryable=True,
    )

def log_and_raise_response_parsing_error(
    provider: str,
    reason: Optional[str] = None,
) -> None:
    """
        Log and raise a response parsing error

        Args:
            provider: Provider identifier
            reason: Optional explanation
    """

    ## Build a normalized response parsing error message
    message = f"Provider response parsing failed: provider={provider}"
    if reason:
        message = f"{message} | {reason}"

    LOGGER.error(message)
    raise ResponseParsingError(
        message=message,
        error_code=ERROR_CODE_RESPONSE_PARSING,
        details={"provider": provider, "reason": reason},
        is_retryable=False,
    )