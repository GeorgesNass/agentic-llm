"""
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Central logging utilities: consistent formatting, console/file handlers, and get_logger helper."
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

## --------------------------------------------------------------------------------------
## Logger configuration (single place to control formatting and handlers)
## --------------------------------------------------------------------------------------
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def _normalize_log_level(level: Optional[str]) -> int:
    """
		Normalize a string log level to a logging numeric level

		Args:
			level: Log level string (e.g., "INFO", "DEBUG")

		Returns:
			Logging level numeric value
    """
    
    if level is None or not level.strip():
        level = _DEFAULT_LOG_LEVEL

    normalized = level.strip().upper()
    return getattr(logging, normalized, logging.INFO)

def _ensure_dir(path: Path) -> None:
    """
		Ensure a directory exists

		Args:
			path: Directory path to create if needed
    """
    
    path.mkdir(parents=True, exist_ok=True)

def _build_log_file_path(
    logs_dir: Path,
    filename: str = "app.log",
) -> Path:
    """
		Build the log file path

		Args:
			logs_dir: Base logs directory
			filename: Log filename

		Returns:
			Full path to the log file
    """
    
    _ensure_dir(logs_dir)
    return logs_dir / filename

def _create_console_handler(level: int) -> logging.Handler:
    """
		Create a console (stdout) logging handler

		Args:
			level: Logging level numeric value

		Returns:
			Configured console handler
    """
    
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT, datefmt=_DEFAULT_DATE_FORMAT))
    return handler

def _create_file_handler(
    log_file_path: Path,
    level: int,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> logging.Handler:
    """
		Create a rotating file logging handler

		Args:
			log_file_path: Target log file path
			level: Logging level numeric value
			max_bytes: Max file size before rotation
			backup_count: Number of rotated files to keep

		Returns:
			Configured file handler
    """
    
    handler = RotatingFileHandler(
        filename=str(log_file_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT, datefmt=_DEFAULT_DATE_FORMAT))
    return handler

def get_logger(
    name: str,
    logs_dir: Optional[Path] = None,
    level: Optional[str] = None,
    log_filename: str = "app.log",
) -> logging.Logger:
    """
		Get a configured logger with consistent handlers and formatting

		Args:
			name: Logger name (usually __name__)
			logs_dir: Directory where log files are written. If None, logs are console-only
			level: Log level string (e.g., "INFO", "DEBUG"). If None, defaults to INFO
			log_filename: Log filename used when logs_dir is provided

		Returns:
			Configured logger instance
    """
    
    numeric_level = _normalize_log_level(level)
    logger = logging.getLogger(name)

    ## Prevent duplicate handlers when get_logger is called multiple times
    if getattr(logger, "_is_configured", False):
        return logger

    logger.setLevel(numeric_level)
    logger.propagate = False

    ## Always attach a console handler
    logger.addHandler(_create_console_handler(numeric_level))

    ## Attach a rotating file handler when logs_dir is provided
    if logs_dir is not None:
        log_file_path = _build_log_file_path(logs_dir=logs_dir, filename=log_filename)
        logger.addHandler(_create_file_handler(log_file_path=log_file_path, level=numeric_level))

    ## Mark as configured to avoid re-attaching handlers
    setattr(logger, "_is_configured", True)

    return logger