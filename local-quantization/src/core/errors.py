'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Custom exception hierarchy and helpers for local-quantization pipeline."
'''

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.utils.logging_utils import get_logger

## Initialize module-level logger
LOGGER = get_logger(__name__)


class LocalQuantizationError(RuntimeError):
    """
        Base exception for all local-quantization errors

        This exception is used as the root of the project exception tree
        to ensure consistent catching and reporting at pipeline boundaries

        Args:
            message: Human-readable error message
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ConfigurationError(LocalQuantizationError):
    """
        Raised when configuration or environment variables are invalid

        Typical causes:
            - Missing required environment variables
            - Invalid enum values (backend name, quantization mode)
            - Invalid paths defined by settings

        Args:
            message: Human-readable error message
    """


class DataError(LocalQuantizationError):
    """
        Raised when input data or model files are invalid or missing

        Typical causes:
            - Model path does not exist
            - Adapter path missing when required
            - Calibration dataset unavailable

        Args:
            message: Human-readable error message
    """


class BackendError(LocalQuantizationError):
    """
        Raised when a quantization backend is unavailable or fails

        Typical causes:
            - Missing dependency (llama.cpp tools, autoawq, auto-gptq)
            - Unsupported model architecture for a backend
            - Runtime conversion failure

        Args:
            message: Human-readable error message
    """


class ExportError(LocalQuantizationError):
    """
        Raised when exporting quantized artifacts fails

        Typical causes:
            - Permission issues while writing under artifacts/exports
            - Missing output directory
            - Serialization failures

        Args:
            message: Human-readable error message
    """


class BenchmarkError(LocalQuantizationError):
    """
        Raised when benchmarking fails

        Typical causes:
            - Runner unavailable for the target format
            - Invalid prompt suite configuration
            - Runtime inference errors

        Args:
            message: Human-readable error message
    """


class PipelineError(LocalQuantizationError):
    """
        Raised when a pipeline step fails

        Typical causes:
            - Unhandled runtime exception inside a pipeline step
            - Invalid step ordering or incompatible settings

        Args:
            message: Human-readable error message
    """


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
    message = f"Required path does not exist: {path}"
    if context:
        message = f"{message} | context={context}"

    LOGGER.error(message)
    raise DataError(message)


def log_and_raise_backend_unavailable(
    backend_name: str,
    reason: Optional[str] = None,
) -> None:
    """
        Log and raise a BackendError when a backend is unavailable

        Args:
            backend_name: Backend identifier
            reason: Optional explanation

        Returns:
            None
    """
    message = f"Quantization backend unavailable: {backend_name}"
    if reason:
        message = f"{message} | {reason}"

    LOGGER.error(message)
    raise BackendError(message)


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
    message = f"Pipeline step failed: {step}"
    if reason:
        message = f"{message} | {reason}"

    LOGGER.error(message)
    raise PipelineError(message)


def log_and_raise_export_error(
    artifact_name: str,
    reason: Optional[str] = None,
) -> None:
    """
        Log and raise an ExportError for a failed export operation

        Args:
            artifact_name: Name of the artifact being exported
            reason: Optional explanation

        Returns:
            None
    """
    message = f"Export failed for artifact: {artifact_name}"
    if reason:
        message = f"{message} | {reason}"

    LOGGER.error(message)
    raise ExportError(message)
