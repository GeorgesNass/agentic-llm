'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Generic utilities: env parsing helpers and filesystem helpers for local-quantization."
'''

from __future__ import annotations

import os
from pathlib import Path


def ensure_parent_dir(file_path: Path) -> None:
    """
        Ensure parent directory exists for a given file path

        Args:
            file_path: Target file path

        Returns:
            None
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)


def get_env_str(key: str, default: str) -> str:
    """
        Read an environment variable as a string

        Args:
            key: Environment variable name
            default: Default value if missing

        Returns:
            The resolved string value
    """
    value = os.getenv(key)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip()


def get_env_int(key: str, default: int) -> int:
    """
        Read an environment variable as an integer

        Args:
            key: Environment variable name
            default: Default value if missing or invalid

        Returns:
            The resolved integer value
    """
    raw = os.getenv(key)
    if raw is None:
        return default

    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def get_env_bool(key: str, default: bool) -> bool:
    """
        Read an environment variable as a boolean

        Args:
            key: Environment variable name
            default: Default value if missing or invalid

        Returns:
            The resolved boolean value
    """
    raw = os.getenv(key)
    if raw is None:
        return default

    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default
