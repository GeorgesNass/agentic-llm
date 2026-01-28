'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Generic utilities for rag-drive-gcp: path helpers, dotenv loader, and directory management."
'''

import os
from pathlib import Path
from typing import Optional

from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("utils")

## ============================================================
## PATH HELPERS
## ============================================================
def get_project_root() -> Path:
    """
        Resolve the project root directory

        Returns:
            Path: Absolute path to the project root
    """
    
    return Path(__file__).resolve().parents[2]

def to_bool(value: Optional[str], default: bool = False) -> bool:
    """
        Convert an environment variable string into a boolean

        Args:
            value (Optional[str]): Input value, typically from environment variables
            default (bool): Fallback if value is None or invalid

        Returns:
            bool: Parsed boolean value
    """
    
    if value is None:
        return default

    val = value.strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False

    return default

## ============================================================
## ENV LOADING (LIGHTWEIGHT)
## ============================================================
def load_dotenv_if_present(env_path: Optional[Path] = None) -> None:
    """
        Load environment variables from a .env file if present

        This implementation:
            - Parses KEY=VALUE lines
            - Ignores comments and blank lines
            - Does not override existing OS environment variables

        Args:
            env_path (Optional[Path]): Path to the .env file. If None, defaults to <project_root>/.env
    """
    
    if env_path is None:
        env_path = get_project_root() / ".env"

    if not env_path.exists():
        logger.debug("No .env file found. Skipping dotenv loading.")
        return

    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()

        ## Skip blanks and comments
        if not stripped or stripped.startswith("#"):
            continue

        if "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        ## Do not override existing environment variables
        if key and key not in os.environ:
            os.environ[key] = value

def ensure_directories(*dirs: Path) -> None:
    """
        Ensure that all given directories exist.

        Args:
            *dirs (Path): Any number of directory paths to create.
    """
    
    for directory in dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.exception(f"Failed to create directory: {directory} | {exc}")
            raise