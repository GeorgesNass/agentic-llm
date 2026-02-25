'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Environment configuration loader and path resolver for LLM proxy gateway."
'''

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

## ============================================================
## ENVIRONMENT HELPERS
## ============================================================
def get_env_str(key: str, default: str = "") -> str:
    """
        Retrieve a string environment variable

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Resolved string value
    """
    
    return os.getenv(key, default)

def get_env_bool(key: str, default: bool = False) -> bool:
    """
        Retrieve a boolean environment variable

        Args:
            key: Environment variable name
            default: Default boolean value

        Returns:
            Boolean value
    """
    
    raw = os.getenv(key)
    if raw is None:
        return default

    return raw.strip().lower() in {"1", "true", "yes", "y"}

def get_env_int(key: str, default: int = 0) -> int:
    """
        Retrieve an integer environment variable

        Args:
            key: Environment variable name
            default: Default integer value

        Returns:
            Integer value
    """
    
    raw = os.getenv(key)
    if raw is None:
        return default

    try:
        return int(raw)
    except ValueError:
        return default

## ============================================================
## PATH RESOLUTION
## ============================================================
def resolve_path(path_str: str) -> Path:
    """
        Resolve a string into an absolute Path

        Args:
            path_str: Raw path string

        Returns:
            Absolute resolved Path
    """
    
    return Path(path_str).expanduser().resolve()

def ensure_directory(path: Path) -> None:
    """
        Ensure a directory exists

        Args:
            path: Directory path

        Returns:
            None
    """
    
    path.mkdir(parents=True, exist_ok=True)

## ============================================================
## SETTINGS DATACLASS
## ============================================================
@dataclass(frozen=True)
class AppConfig:
    """
        Global application configuration for LLM proxy gateway

        Args:
            app_name: Application name
            app_version: Version string
            debug: Debug mode flag
            log_level: Logging level string
            data_raw_dir: Path to raw txt files
            data_processed_dir: Path to processed csv files
            exports_dir: Path to exports directory
            models_catalog_path: Path to models catalog JSON
            pricing_catalog_path: Path to pricing catalog JSON
    """

    app_name: str
    app_version: str
    debug: bool
    log_level: str
    data_raw_dir: Path
    data_processed_dir: Path
    exports_dir: Path
    models_catalog_path: Path
    pricing_catalog_path: Path

## ============================================================
## CONFIG FACTORY
## ============================================================
def build_app_config() -> AppConfig:
    """
        Build AppConfig from environment variables

        Environment variables:
            - APP_NAME
            - APP_VERSION
            - APP_DEBUG
            - APP_LOG_LEVEL
            - DATA_RAW_DIR
            - DATA_PROCESSED_DIR
            - EXPORTS_DIR
            - MODELS_CATALOG_PATH
            - PRICING_CATALOG_PATH

        Returns:
            AppConfig instance
    """

    app_name = get_env_str("APP_NAME", "llm-proxy-gateway")
    app_version = get_env_str("APP_VERSION", "1.0.0")
    debug = get_env_bool("APP_DEBUG", False)
    log_level = get_env_str("APP_LOG_LEVEL", "INFO")

    data_raw_dir = resolve_path(get_env_str("DATA_RAW_DIR", "data/raw"))
    data_processed_dir = resolve_path(
        get_env_str("DATA_PROCESSED_DIR", "data/processed")
    )
    exports_dir = resolve_path(get_env_str("EXPORTS_DIR", "artifacts/exports"))

    models_catalog_path = resolve_path(
        get_env_str(
            "MODELS_CATALOG_PATH",
            "artifacts/resources/models_catalog.json",
        )
    )

    pricing_catalog_path = resolve_path(
        get_env_str(
            "PRICING_CATALOG_PATH",
            "artifacts/resources/pricing_catalog.json",
        )
    )

    ## Ensure core directories exist
    ensure_directory(data_raw_dir)
    ensure_directory(data_processed_dir)
    ensure_directory(exports_dir)

    return AppConfig(
        app_name=app_name,
        app_version=app_version,
        debug=debug,
        log_level=log_level,
        data_raw_dir=data_raw_dir,
        data_processed_dir=data_processed_dir,
        exports_dir=exports_dir,
        models_catalog_path=models_catalog_path,
        pricing_catalog_path=pricing_catalog_path,
    )
    
settings = build_app_config()