'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized logging configuration and execution time decorator for autonomous-ai-platform."
'''

from __future__ import annotations

import logging
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

T = TypeVar("T")

## ============================================================
## LOGGER FACTORY
## ============================================================
def get_logger(name: str) -> logging.Logger:
    """
        Create or retrieve a configured logger instance

        High-level workflow:
            1) Configure root logger if not already configured
            2) Attach stream handler
            3) Return named logger

        Args:
            name: Logger name

        Returns:
            Configured logging.Logger
    """

    ## Configure root logger only once
    if not logging.getLogger().handlers:
        _configure_root_logger()

    return logging.getLogger(name)

def _configure_root_logger() -> None:
    """
        Configure root logger with standard format

        Returns:
            None
    """

    log_level = logging.INFO

    ## Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ## Create stream handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    ## Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(stream_handler)

## ============================================================
## EXECUTION TIME DECORATOR
## ============================================================
def log_execution_time(func: Callable[..., T]) -> Callable[..., T]:
    """
        Decorator to log execution time of a function

        High-level workflow:
            1) Record start time
            2) Execute wrapped function
            3) Compute duration
            4) Log duration

        Args:
            func: Function to decorate

        Returns:
            Wrapped function
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        logger = get_logger(func.__module__)

        ## Record start time
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            return result

        finally:
            ## Compute duration even if exception is raised
            end_time = time.perf_counter()
            duration = end_time - start_time

            logger.info(
                "Function executed | name=%s | duration=%.4f seconds",
                func.__name__,
                duration,
            )

    return cast(Callable[..., T], wrapper)

## ============================================================
## SAFE FILE LOGGER INITIALIZATION
## ============================================================
def attach_file_handler(log_file: Path) -> None:
    """
        Attach a file handler to the root logger

        High-level workflow:
            1) Ensure parent directory exists
            2) Create file handler
            3) Attach to root logger

        Args:
            log_file: Path to log file

        Returns:
            None
    """

    ## Ensure parent directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logging.getLogger().addHandler(file_handler)