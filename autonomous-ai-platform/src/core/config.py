'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Environment configuration loader and structured AppConfig builder for autonomous-ai-platform."
'''

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.utils.logging_utils import get_logger
from src.utils.env_utils import _get_env_str, _get_env_int, _get_env_bool

logger = get_logger(__name__)

## ============================================================
## CONFIG MODELS
## ============================================================
@dataclass(frozen=True)
class RuntimeConfig:
    """
        Runtime configuration

        Args:
            use_gpu: Whether GPU backends are enabled
            provider_default: Default LLM provider (auto/local/openai/...)
            local_model_path: Optional path to local quantized model
    """

    use_gpu: bool
    provider_default: str
    local_model_path: Optional[Path]

@dataclass(frozen=True)
class RagConfig:
    """
        Retrieval configuration

        Args:
            top_k: Default retrieval depth
            chunk_size: Chunk size for text splitting
            chunk_overlap: Overlap between chunks
            vector_backend: Vector backend identifier
    """

    top_k: int
    chunk_size: int
    chunk_overlap: int
    vector_backend: str

@dataclass(frozen=True)
class PathsConfig:
    """
        Filesystem paths configuration

        Args:
            root_dir: Project root
            logs_dir: Logs directory
            data_raw_dir: Raw ingestion directory
            data_processed_dir: Processed chunks directory
            data_sqlite_dir: SQLite DB directory
            artifacts_models_dir: Local models directory
            artifacts_vector_store_dir: Vector store directory
            artifacts_reports_dir: Reports directory
            artifacts_evaluations_dir: Evaluation outputs directory
    """

    root_dir: Path
    logs_dir: Path
    data_raw_dir: Path
    data_processed_dir: Path
    data_sqlite_dir: Path
    artifacts_models_dir: Path
    artifacts_vector_store_dir: Path
    artifacts_reports_dir: Path
    artifacts_evaluations_dir: Path

@dataclass(frozen=True)
class ApiKeysConfig:
    """
        API keys configuration

        Args:
            openai_api_key: OpenAI API key
            xai_api_key: xAI API key
    """

    openai_api_key: str
    xai_api_key: str

@dataclass(frozen=True)
class AppConfig:
    """
        Global application configuration

        Args:
            app_name: Application name
            app_version: Version string
            environment: Environment name (dev/prod)
            log_level: Logging level
            runtime: Runtime configuration
            rag: Retrieval configuration
            paths: Filesystem paths configuration
            api_keys: API keys configuration
    """

    app_name: str
    app_version: str
    environment: str
    log_level: str
    runtime: RuntimeConfig
    rag: RagConfig
    paths: PathsConfig
    api_keys: ApiKeysConfig

## ============================================================
## CONFIG FACTORY
## ============================================================
def build_config() -> AppConfig:
    """
        Build the global AppConfig from environment variables

        High-level workflow:
            1) Read metadata
            2) Resolve runtime configuration
            3) Resolve retrieval configuration
            4) Resolve filesystem paths
            5) Build structured AppConfig

        Returns:
            AppConfig instance
    """

    ## Metadata
    app_name = _get_env_str("APP_NAME", "autonomous-ai-platform")
    app_version = _get_env_str("APP_VERSION", "1.0.0")
    environment = _get_env_str("ENVIRONMENT", "dev")
    log_level = _get_env_str("LOG_LEVEL", "INFO")

    ## Runtime
    use_gpu = _get_env_bool("USE_GPU", False)
    provider_default = _get_env_str("PROVIDER_DEFAULT", "auto")
    local_model_path_raw = _get_env_str("LOCAL_MODEL_PATH", "").strip()
    local_model_path = (
        Path(local_model_path_raw).expanduser().resolve()
        if local_model_path_raw
        else None
    )

    runtime = RuntimeConfig(
        use_gpu=use_gpu,
        provider_default=provider_default,
        local_model_path=local_model_path,
    )

    ## Retrieval
    top_k = _get_env_int("TOP_K", 8)
    chunk_size = _get_env_int("CHUNK_SIZE", 1024)
    chunk_overlap = _get_env_int("CHUNK_OVERLAP", 128)
    vector_backend = _get_env_str("VECTOR_BACKEND", "faiss")

    rag = RagConfig(
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        vector_backend=vector_backend,
    )

    ## Paths
    root_dir = Path(".").resolve()
    logs_dir = Path("logs").resolve()
    data_raw_dir = Path("data/raw").resolve()
    data_processed_dir = Path("data/processed").resolve()
    data_sqlite_dir = Path("data/sqlite").resolve()
    artifacts_models_dir = Path("artifacts/models").resolve()
    artifacts_vector_dir = Path("artifacts/vector_store").resolve()
    artifacts_reports_dir = Path("artifacts/reports").resolve()
    artifacts_evaluations_dir = Path("artifacts/evaluations").resolve()

    ## Ensure directories exist
    for directory in [
        logs_dir,
        data_raw_dir,
        data_processed_dir,
        data_sqlite_dir,
        artifacts_models_dir,
        artifacts_vector_dir,
        artifacts_reports_dir,
        artifacts_evaluations_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    paths = PathsConfig(
        root_dir=root_dir,
        logs_dir=logs_dir,
        data_raw_dir=data_raw_dir,
        data_processed_dir=data_processed_dir,
        data_sqlite_dir=data_sqlite_dir,
        artifacts_models_dir=artifacts_models_dir,
        artifacts_vector_store_dir=artifacts_vector_dir,
        artifacts_reports_dir=artifacts_reports_dir,
        artifacts_evaluations_dir=artifacts_evaluations_dir,
    )

    ## API Keys
    api_keys = ApiKeysConfig(
        openai_api_key=_get_env_str("OPENAI_API_KEY", ""),
        xai_api_key=_get_env_str("XAI_API_KEY", ""),
    )

    return AppConfig(
        app_name=app_name,
        app_version=app_version,
        environment=environment,
        log_level=log_level,
        runtime=runtime,
        rag=rag,
        paths=paths,
        api_keys=api_keys,
    )


## Global config instance
config = build_config()