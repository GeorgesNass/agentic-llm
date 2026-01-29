'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized custom exceptions and helpers for clean, user-friendly pipeline errors."
'''

from __future__ import annotations

from pathlib import Path
from typing import List

from src.utils.logging_utils import get_logger

logger = get_logger("errors")

## ============================================================
## CUSTOM EXCEPTIONS
## ============================================================
class ConfigurationError(RuntimeError):
    """
		Raised when application configuration is invalid.
    """

class DataError(RuntimeError):
    """
		Raised when dataset/files required for the pipeline are missing or invalid.
    """

## ============================================================
## HELPERS
## ============================================================
def log_and_raise_missing_env(vars_missing: List[str]) -> None:
    """
		Log and raise a configuration error for missing environment variables.

		Args:
			vars_missing: List of missing env variable names.

		Raises:
			ConfigurationError: Always raised after logging.
    """
    
    message = (
        "Missing environment variables (placeholders detected): "
        + ", ".join(vars_missing)
    )
    logger.error(message)
    raise ConfigurationError(message)

def log_and_raise_missing_raw_data(raw_dir: Path) -> None:
    """
		Log and raise a data error when no raw dataset file is found.

		Args:
			raw_dir: Directory where raw dataset files are expected.

		Raises:
			DataError: Always raised after logging.
    """
    
    message = (
        "No raw data file found in: "
        f"{raw_dir}. "
        "Expected one file with extension: .csv, .jsonl, or .json. "
        "Fix: put your dataset into data/raw/ (or set RAW_DATA_DIR in .env)."
    )
    logger.error(message)
    raise DataError(message)