'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Generic logging utilities with sync/async execution-time decorator."
'''

from __future__ import annotations

import asyncio
import functools
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

## ============================================================
## LOG DIRECTORY MANAGEMENT
## ============================================================
def _ensure_log_dir(
    log_dir: str | Path | None = None,

    ## Dummy parameter for backward compatibility
    logs_dir: str | Path | None = None,
) -> Path:
    """
        Ensure the log directory exists

        Args:
            log_dir: Optional custom log directory

        Returns:
            Path to the log directory
    """

    ## Resolve compatibility alias for legacy parameter
    if logs_dir and not log_dir:
        log_dir = logs_dir

    ## Resolve log directory from argument or environment
    resolved_dir = Path(log_dir or os.getenv("LOG_DIR", "logs"))

    ## Create directory if it does not exist
    resolved_dir.mkdir(parents=True, exist_ok=True)

    return resolved_dir

## ============================================================
## LOGGER FACTORY
## ============================================================
def get_logger(
    name: str = "app",
    log_file: str | None = None,
    log_dir: str | Path | None = None,

    ## Dummy parameters for backward compatibility (absorbing legacy kwargs)
    level: str | None = None,
    logs_dir: str | Path | None = None,
    log_filename: str | None = None,
    filename: str | None = None,
    enable_file: bool | None = None,
    enable_console: bool | None = None,
    propagate: bool | None = None,
) -> logging.Logger:
    """
        Build and configure a logger

        Behavior:
            - console logging
            - file logging
            - environment driven configuration

        Args:
            name: Logger name
            log_file: Optional log filename
            log_dir: Optional log directory

        Returns:
            Configured logger
    """

    ## Resolve compatibility aliases for legacy parameters
    if logs_dir and not log_dir:
        log_dir = logs_dir

    if log_filename and not log_file:
        log_file = log_filename

    if filename and not log_file:
        log_file = filename

    ## Resolve logging level from environment
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    ## Create or retrieve logger instance
    logger = logging.getLogger(name)
    logger.setLevel(level)

    ## Prevent logs from propagating to root logger
    logger.propagate = False

    ## Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    ## Define standard log format including function name
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"
    )

    ## Create console handler
    console_handler = logging.StreamHandler(sys.stdout)

    ## Apply log level to console
    console_handler.setLevel(level)

    ## Apply formatter
    console_handler.setFormatter(formatter)

    ## Attach console handler
    logger.addHandler(console_handler)

    ## Enable file logging depending on environment variable
    if os.getenv("LOG_TO_FILE", "true").lower() == "true":

        ## Resolve log directory
        log_dir_path = _ensure_log_dir(log_dir)

        ## Determine log filename
        if log_file:
            filename = log_file
        else:
            safe_name = name.replace(".", "_")
            filename = f"{safe_name}.log"

        ## Create file handler
        file_handler = logging.FileHandler(
            log_dir_path / filename,
            encoding="utf-8",
        )

        ## Apply log level
        file_handler.setLevel(level)

        ## Apply formatter
        file_handler.setFormatter(formatter)

        ## Attach file handler
        logger.addHandler(file_handler)

    return logger
    
## ============================================================
## UTILITY FUNCTION
## ===========================================================
def get_absolute_path(path_like: str | Path | None = None) -> str:
    """
        Return absolute path

        Args:
            path_like: Optional path

        Returns:
            Absolute path string
    """

    ## Resolve target path
    target = Path(path_like) if path_like else Path.cwd()

    ## Convert to absolute path
    return str(target.resolve())

## ============================================================
## EXECUTION TIME DECORATOR
## ============================================================
def log_execution_time_and_path(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    """
        Log execution time of sync or async functions

        Args:
            func: Function to decorate

        Returns:
            Wrapped function
    """

    ## Create logger based on module name
    logger = get_logger(func.__module__)

    ## Detect async functions
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:

            ## Record start time
            start_time = time.perf_counter()

            try:

                ## Execute async function
                result = await func(*args, **kwargs)

                ## Compute execution time
                elapsed = time.perf_counter() - start_time

                ## Log execution information
                logger.info(
                    "Function '%s' executed in %.4fs | path=%s",
                    func.__name__,
                    elapsed,
                    get_absolute_path(),
                )

                return result

            except Exception as error:

                ## Log error
                logger.error(
                    "Function '%s' failed: %s",
                    func.__name__,
                    error,
                )

                ## Print traceback if debug enabled
                if os.getenv("DEBUG", "false").lower() == "true":
                    logger.debug(traceback.format_exc())

                raise

        return async_wrapper

    ## Sync function wrapper
    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:

        ## Record start time
        start_time = time.perf_counter()

        try:

            ## Execute function
            result = func(*args, **kwargs)

            ## Compute execution time
            elapsed = time.perf_counter() - start_time

            ## Log execution information
            logger.info(
                "Function '%s' executed in %.4fs | path=%s",
                func.__name__,
                elapsed,
                get_absolute_path(),
            )

            return result

        except Exception as error:

            ## Log error
            logger.error(
                "Function '%s' failed: %s",
                func.__name__,
                error,
            )

            ## Print traceback if debug enabled
            if os.getenv("DEBUG", "false").lower() == "true":
                logger.debug(traceback.format_exc())

            raise

    return sync_wrapper

## ============================================================
## BACKWARD COMPATIBILITY
## ============================================================
def log_execution_time(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    """
        Alias for execution time decorator
    """

    ## Reuse main decorator
    return log_execution_time_and_path(func)