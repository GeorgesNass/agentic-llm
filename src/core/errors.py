'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Custom exception hierarchy and error helpers for LLM proxy gateway."
'''

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from src.utils.logging_utils import get_logger

## Initialize module-level logger
LOGGER = get_logger(__name__)

## ============================================================
## EXCEPTION HIERARCHY
## ============================================================
class LLMProxyGatewayError(RuntimeError):
    """
        Base exception for all LLM proxy gateway errors

        Args:
            message: Human-readable error message
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)

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
    raise ConfigurationError(message)

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
    raise DataError(message)

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

    p = Path(path).expanduser().resolve()

    ## Reuse the same formatting as missing_path, but explicit wording
    message = f"Required file does not exist: {p}"
    if context:
        message = f"{message} | context={context}"

    LOGGER.error(message)
    raise DataError(message)

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
    raise ProviderError(message)

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
    raise ValidationError(message)

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
    raise PipelineError(message)

def log_and_raise_os_error_as_data_error(
    exc: Exception,
    context: Optional[str] = None,
) -> None:
    """
        Convert OS-level / Python IO errors (FileNotFoundError, PermissionError, etc.)
        into a clean DataError with consistent logging.

        Args:
            exc: Original exception
            context: Optional context label

        Returns:
            None
    """

    message = f"Data IO error: {type(exc).__name__} | {str(exc)}"
    if context:
        message = f"{message} | context={context}"

    LOGGER.error(message)
    raise DataError(message)