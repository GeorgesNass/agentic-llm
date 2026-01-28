'''
__author__ = "Georges Nassopoulos"
__version__ = "1.0.0"
__status__ = "Dev"
__desc__ = "Centralized error handling utilities for rag-drive-gcp."
'''

from typing import List

from src.utils.logging_utils import get_logger

logger = get_logger("errors")


## ============================================================
## CUSTOM EXCEPTIONS
## ============================================================
class ConfigurationError(RuntimeError):
    """Raised when application configuration is invalid."""


## ============================================================
## HELPERS
## ============================================================
def log_and_raise_missing_env(vars_missing: List[str]) -> None:
    """
        Log and raise a configuration error for missing environment variables

        Args:
            vars_missing (List[str]): Missing or placeholder variables

        Raises:
            ConfigurationError: Always
    """

    message = (
        "Missing environment variables (placeholders detected): "
        + ", ".join(vars_missing)
    )

    ## Log as ERROR (visible in console + file)
    logger.error(message)

    ## Fail fast (config error = non-recoverable)
    raise ConfigurationError(message)
