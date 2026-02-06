'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized logging utilities (console + file, no duplicate handlers)."
'''

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.utils.utils import (
    ensure_parent_dir,
    get_env_bool,
    get_env_int,
    get_env_str,
)


## Module-level cache to prevent duplicate logger configuration
_LOGGER_CACHE: dict[str, logging.Logger] = {}


def _resolve_level(level_str: str) -> int:
    """
        Resolve a log level string into a logging level int

        Args:
            level_str: Level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Returns:
            A logging level integer
    """
    normalized = str(level_str).strip().upper()
    return getattr(logging, normalized, logging.INFO)


def _build_formatter() -> logging.Formatter:
    """
        Build the standard formatter used across the project

        Args:
            None

        Returns:
            A configured logging formatter
    """
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(
    name: str = "local_quantization",
    log_dir: str | Path = "logs",
    log_filename: str = "local_quantization.log",
    level: Optional[str] = None,
) -> logging.Logger:
    """
        Build or retrieve a configured logger for the project

        Notes:
            - Prevents duplicate handlers when called multiple times
            - Supports console + rotating file handler
            - Controlled via environment variables

        Environment variables:
            - LOCAL_QUANTIZATION_LOG_LEVEL: Default log level (INFO)
            - LOCAL_QUANTIZATION_LOG_TO_FILE: Enable file logging (true)
            - LOCAL_QUANTIZATION_LOG_MAX_BYTES: Rotating max bytes (10485760)
            - LOCAL_QUANTIZATION_LOG_BACKUP_COUNT: Rotating backup count (5)

        Args:
            name: Logger name
            log_dir: Folder where log files are written
            log_filename: Log filename (within log_dir)
            level: Override log level as string

        Returns:
            A configured logger instance
    """
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    ## Resolve log level (override > env > default)
    env_level = get_env_str("LOCAL_QUANTIZATION_LOG_LEVEL", "INFO")
    effective_level = _resolve_level(level if level is not None else env_level)

    logger = logging.getLogger(name)
    logger.setLevel(effective_level)
    logger.propagate = False

    ## Avoid duplicate handlers if logger was configured elsewhere
    if logger.handlers:
        _LOGGER_CACHE[name] = logger
        return logger

    formatter = _build_formatter()

    ## Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(effective_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    ## Optional file handler
    log_to_file = get_env_bool("LOCAL_QUANTIZATION_LOG_TO_FILE", True)
    if log_to_file:
        max_bytes = get_env_int(
            "LOCAL_QUANTIZATION_LOG_MAX_BYTES",
            10 * 1024 * 1024,
        )
        backup_count = get_env_int(
            "LOCAL_QUANTIZATION_LOG_BACKUP_COUNT",
            5,
        )

        log_file_path = Path(log_dir) / log_filename
        ensure_parent_dir(log_file_path)

        file_handler = RotatingFileHandler(
            filename=str(log_file_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(effective_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGER_CACHE[name] = logger
    return logger
