'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized logging configuration with structured formatting and execution time decorator."
'''

from __future__ import annotations

import logging
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Any

## ============================================================
## LOGGER CONFIGURATION
## ============================================================
LOG_DIR = Path("logs") ## Ensure logs directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "llm_proxy_gateway.log" ## Define main log file

def get_logger(name: str) -> logging.Logger:
    """
        Create and configure a logger instance

        High-level workflow:
            1) Create logger
            2) Attach console handler
            3) Attach file handler
            4) Apply unified formatter

        Args:
            name: Logger name (usually __name__)

        Returns:
            Configured logging.Logger instance
    """

    ## Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ## Prevent duplicate handlers
    if logger.handlers:
        return logger

    ## Unified formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ## Console handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    ## File handler (DEBUG level for full trace)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger

## ============================================================
## EXECUTION TIME DECORATOR
## ============================================================
def log_execution_time_and_path(func: Callable) -> Callable:
    """
        Decorator to log execution time and file path on error

        High-level workflow:
            1) Capture start time
            2) Execute function
            3) Log execution duration
            4) Log absolute path on exception

        Args:
            func: Function to wrap

        Returns:
            Wrapped function with logging
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_logger(func.__module__)

        ## Capture start time
        start_time = time.time()

        try:
            result = func(*args, **kwargs)

            ## Compute execution duration
            duration = round(time.time() - start_time, 4)

            logger.info(
                "Function '%s' executed in %s seconds",
                func.__name__,
                duration,
            )

            return result

        except Exception as exc:
            duration = round(time.time() - start_time, 4)

            ## Capture absolute execution path
            abs_path = Path(os.getcwd()).resolve()

            logger.error(
                "Error in function '%s' after %s seconds",
                func.__name__,
                duration,
            )

            logger.error("Execution path: %s", abs_path)

            ## Console clean error message
            logger.error("%s", str(exc))

            ## Full traceback only stored in DEBUG (file handler)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Traceback:", exc_info=True)

            raise

    return wrapper