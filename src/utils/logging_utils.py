'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized logging utilities for rag-drive-gcp (console + rotating file handlers)."
'''

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

## ============================================================
## LOGGER FACTORY
## ============================================================
def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
        Create or retrieve a configured logger instance

        The logger:
            - Logs to console
            - Logs to rotating file in logs/
            - Uses log level from environment variables

        Args:
            name (str): Logger name (usually __name__ or module name)
            log_file (Optional[str]): Optional custom log file name

        Returns:
            logging.Logger: Configured logger instance
    """

    logger = logging.getLogger(name)

    ## Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    ## ========================================================
    ## LOG LEVEL (FROM ENV ONLY – NO SETTINGS IMPORT)
    ## ========================================================

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(log_level)

    ## ========================================================
    ## FORMATTER
    ## ========================================================

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ## ========================================================
    ## CONSOLE HANDLER
    ## ========================================================

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    ## ========================================================
    ## FILE HANDLER (ROTATING)
    ## ========================================================

    project_root = Path(__file__).resolve().parents[2]
    logs_dir: Path = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    file_name = log_file if log_file else f"{name}.log"
    file_path = logs_dir / file_name

    file_handler = RotatingFileHandler(
        filename=file_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    ## ========================================================
    ## REGISTER HANDLERS
    ## ========================================================

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger
