'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unified configuration loader for autonomous-ai-platform: dotenv, env parsing, paths, profiles, secrets, runtime metadata and structured app config."
'''

from __future__ import annotations

import json
import os
import platform
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## PLACEHOLDER TOKENS
## ============================================================
PLACEHOLDER_PREFIXES: Tuple[str, ...] = ("<YOUR_", "YOUR_", "CHANGE_ME", "REPLACE_ME", "TODO")

## ============================================================
## FAILSAFE DEFAULT VALUES (NON-SENSITIVE)
## ============================================================

local_model_path = _get_env_str("LOCAL_MODEL_PATH", "path/to/local/model.gguf")
hf_model_id = _get_env_str("HF_MODEL_ID", "repo/model-name")
hf_model_filename = _get_env_str("HF_MODEL_FILENAME", "model.gguf")

vllm_model = _get_env_str("VLLM_MODEL", "local-model")
vllm_api_key = _get_env_str("VLLM_API_KEY", "")

generic_base_url = _get_env_str("GENERIC_OAI_BASE_URL", "http://localhost:8000/v1")
generic_model = _get_env_str("GENERIC_OAI_MODEL", "local-model")
generic_embedding_model = _get_env_str("GENERIC_OAI_EMBEDDING_MODEL", "local-embedding")

## ============================================================
## OS / SYSTEM CONSTANTS
## ============================================================
SYSTEM_NAME = platform.system().lower()
IS_WINDOWS = SYSTEM_NAME == "windows"
IS_LINUX = SYSTEM_NAME == "linux"
IS_MACOS = SYSTEM_NAME == "darwin"
DEFAULT_ENCODING = "utf-8"
DEFAULT_SHELL = "cmd" if IS_WINDOWS else "bash"
DEFAULT_LIBREOFFICE_BINARY = "soffice.exe" if IS_WINDOWS else "soffice"
DEFAULT_SQLITE_FILENAME = "app.db"

## ============================================================
## STABLE DOMAIN CONSTANTS
## ============================================================
DEFAULT_APP_NAME = "autonomous-ai-platform"
DEFAULT_APP_VERSION = "1.0.0"
DEFAULT_ENVIRONMENT = "dev"
DEFAULT_PROFILE = "cpu"

DEFAULT_DATA_DIR = "data"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_SECRETS_DIR = "secrets"

DEFAULT_DATA_RAW_DIR = "data/raw"
DEFAULT_DATA_PROCESSED_DIR = "data/processed"
DEFAULT_DATA_SQLITE_DIR = "data/sqlite"

DEFAULT_ARTIFACTS_MODELS_DIR = "artifacts/models"
DEFAULT_ARTIFACTS_VECTOR_STORE_DIR = "artifacts/vector_store"
DEFAULT_ARTIFACTS_REPORTS_DIR = "artifacts/reports"
DEFAULT_ARTIFACTS_EVALUATIONS_DIR = "artifacts/evaluations"

SUPPORTED_INPUT_EXTENSIONS = (".txt", ".md", ".pdf", ".docx", ".csv", ".json")
SUPPORTED_VECTOR_BACKENDS = ("faiss", "chroma")
SUPPORTED_PROVIDERS = ("auto", "openai", "xai", "google", "local", "vllm")

## ============================================================
## CONFIG MODELS
## ============================================================
@dataclass(frozen=True)
class ExecutionMetadata:
    """
        Execution metadata configuration

        Args:
            run_id: Unique execution identifier
            started_at_utc: UTC timestamp of configuration build
            hostname: Current host name
            platform_name: Current operating system name
            profile: Active runtime profile
            environment: Active environment
    """

    run_id: str
    started_at_utc: str
    hostname: str
    platform_name: str
    profile: str
    environment: str

@dataclass(frozen=True)
class RuntimeConfig:
    """
        Runtime configuration

        Args:
            environment: Environment name
            profile: Runtime profile name
            use_gpu_mode: Raw GPU mode
            use_gpu: Whether GPU execution is enabled
            debug: Whether debug mode is enabled
            log_level: Logging level
            provider_default: Default LLM provider
            request_timeout_seconds: Global timeout in seconds
            max_concurrency: Maximum concurrent tasks
            temperature: Default generation temperature
            max_tokens: Default generation max tokens
            allowed_origins: Optional list of allowed origins
    """

    environment: str
    profile: str
    use_gpu_mode: str
    use_gpu: bool
    debug: bool
    log_level: str
    provider_default: str
    request_timeout_seconds: int
    max_concurrency: int
    temperature: float
    max_tokens: int
    allowed_origins: list[str]

@dataclass(frozen=True)
class RagConfig:
    """
        Retrieval configuration

        Args:
            vector_backend: Vector backend identifier
            embedding_provider: Embedding provider name
            top_k: Default retrieval depth
            chunk_size: Chunk size for splitting
            chunk_overlap: Chunk overlap for splitting
            score_threshold: Optional retrieval score threshold
            recursive_ingestion: Whether recursive ingestion is enabled
            supported_extensions: Supported input extensions
    """

    vector_backend: str
    embedding_provider: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    score_threshold: float
    recursive_ingestion: bool
    supported_extensions: list[str]

@dataclass(frozen=True)
class PathsConfig:
    """
        Filesystem paths configuration

        Args:
            root_dir: Project root directory
            src_dir: Source directory
            data_dir: Data root directory
            data_raw_dir: Raw data directory
            data_processed_dir: Processed data directory
            data_sqlite_dir: SQLite storage directory
            logs_dir: Logs directory
            artifacts_dir: Artifacts root directory
            artifacts_models_dir: Models artifact directory
            artifacts_vector_store_dir: Vector store artifact directory
            artifacts_reports_dir: Reports artifact directory
            artifacts_evaluations_dir: Evaluation artifact directory
            secrets_dir: Secrets directory
            sqlite_db_path: SQLite database path
            libreoffice_binary: LibreOffice binary path or command
    """

    root_dir: Path
    src_dir: Path
    data_dir: Path
    data_raw_dir: Path
    data_processed_dir: Path
    data_sqlite_dir: Path
    logs_dir: Path
    artifacts_dir: Path
    artifacts_models_dir: Path
    artifacts_vector_store_dir: Path
    artifacts_reports_dir: Path
    artifacts_evaluations_dir: Path
    secrets_dir: Path
    sqlite_db_path: Path
    libreoffice_binary: str

@dataclass(frozen=True)
class ApiKeysConfig:
    """
        API keys configuration

        Args:
            openai_api_key: OpenAI API key
            xai_api_key: xAI API key
            google_api_key: Google API key
            anthropic_api_key: Anthropic API key
    """

    openai_api_key: str
    xai_api_key: str
    google_api_key: str
    anthropic_api_key: str

@dataclass(frozen=True)
class AppConfig:
    """
        Global application configuration

        Args:
            app_name: Application name
            app_version: Application version
            execution: Execution metadata
            runtime: Runtime configuration
            rag: Retrieval configuration
            paths: Filesystem paths configuration
            api_keys: API keys configuration
    """

    app_name: str
    app_version: str
    execution: ExecutionMetadata
    runtime: RuntimeConfig
    rag: RagConfig
    paths: PathsConfig
    api_keys: ApiKeysConfig

## ============================================================
## DOTENV / ENV HELPERS
## ============================================================
def _find_project_root() -> Path:
    """
        Resolve the project root directory

        High-level workflow:
            1) Prefer PROJECT_ROOT when explicitly provided
            2) Otherwise derive the root from the current file location

        Returns:
            Resolved project root path
    """

    ## Prefer explicit override
    project_root_raw = os.getenv("PROJECT_ROOT", "").strip()
    if project_root_raw:
        return Path(project_root_raw).expanduser().resolve()

    ## Fallback to relative resolution
    return Path(__file__).resolve().parents[2]

def _load_dotenv_if_present() -> None:
    """
        Load environment variables from a local .env file if available

        Design choice:
            - Dotenv loading stays optional
            - Missing python-dotenv never breaks runtime

        Returns:
            None
    """

    ## Import dotenv lazily
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.debug("python-dotenv not installed. Skipping .env loading.")
        return

    ## Load project-level .env when present
    env_path = _find_project_root() / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        logger.debug("Loaded .env from %s", env_path)

def _is_placeholder(value: str) -> bool:
    """
        Detect whether a raw environment value looks like a placeholder

        Args:
            value: Raw environment value

        Returns:
            True if the value looks like a placeholder
    """

    ## Normalize before checks
    normalized = value.strip().upper()
    return any(token in normalized for token in PLACEHOLDER_PREFIXES)

def _get_env_str(key: str, default: str = "", *, strip: bool = True) -> str:
    """
        Read a string environment variable

        Args:
            key: Environment variable name
            default: Default fallback value
            strip: Whether to strip whitespace

        Returns:
            Normalized string value
    """

    ## Read from process environment
    value = os.getenv(key, default)

    ## Normalize potential None values
    if value is None:
        value = default

    ## Strip when requested
    return value.strip() if strip else value

def _get_env_bool(key: str, default: bool = False) -> bool:
    """
        Read a boolean environment variable

        Args:
            key: Environment variable name
            default: Default fallback value

        Returns:
            Parsed boolean value

        Raises:
            ValueError: If the raw value is invalid
    """

    ## Read raw value
    raw = _get_env_str(key, str(default)).lower()

    ## Parse common truthy values
    if raw in {"1", "true", "yes", "y", "on"}:
        return True

    ## Parse common falsy values
    if raw in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(f"Invalid boolean value for {key}: {raw}")

def _get_env_int(key: str, default: int) -> int:
    """
        Read an integer environment variable

        Args:
            key: Environment variable name
            default: Default fallback value

        Returns:
            Parsed integer value

        Raises:
            ValueError: If the raw value is invalid
    """

    ## Read raw value
    raw = _get_env_str(key, str(default))

    ## Parse integer strictly
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer. Got: {raw}") from exc

def _get_env_float(key: str, default: float) -> float:
    """
        Read a float environment variable

        Args:
            key: Environment variable name
            default: Default fallback value

        Returns:
            Parsed float value

        Raises:
            ValueError: If the raw value is invalid
    """

    ## Read raw value
    raw = _get_env_str(key, str(default))

    ## Parse float strictly
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a float. Got: {raw}") from exc

def _get_env_list(key: str, default: Optional[list[str]] = None, *, separator: str = ",") -> list[str]:
    """
        Read a list-like environment variable

        Args:
            key: Environment variable name
            default: Default fallback list
            separator: Element separator in the raw string

        Returns:
            Parsed list of strings
    """

    ## Read raw value
    raw = _get_env_str(key, "")

    ## Fallback to default when empty
    if not raw:
        return list(default or [])

    ## Split and normalize elements
    return [item.strip() for item in raw.split(separator) if item.strip()]

def _get_env_path(key: str, default: str, root_dir: Path) -> Path:
    """
        Read and resolve a path environment variable

        Args:
            key: Environment variable name
            default: Default path value
            root_dir: Project root directory

        Returns:
            Resolved path
    """

    ## Read raw path value
    raw = _get_env_str(key, default)

    ## Expand user home and env vars
    expanded = os.path.expandvars(raw)
    path_obj = Path(expanded).expanduser()

    ## Resolve relative paths from project root
    if path_obj.is_absolute():
        return path_obj.resolve()

    return (root_dir / path_obj).resolve()

def _read_secret_value(direct_key: str, file_key: str, *, root_dir: Path, default: str = "") -> str:
    """
        Read a secret either directly from env or from a referenced file

        High-level workflow:
            1) Prefer direct environment value
            2) Otherwise try a file path from env
            3) Return default when nothing is available

        Args:
            direct_key: Direct env key containing the secret value
            file_key: Env key containing the secret file path
            root_dir: Project root directory
            default: Default fallback value

        Returns:
            Resolved secret value
    """

    ## Prefer direct secret value
    direct_value = _get_env_str(direct_key, default)
    if direct_value and not _is_placeholder(direct_value):
        return direct_value

    ## Fallback to secret file path
    secret_path_raw = _get_env_str(file_key, "")
    if not secret_path_raw:
        return default

    ## Resolve file path safely
    secret_path = _get_env_path(file_key, secret_path_raw, root_dir)

    ## Read secret content if the file exists
    if secret_path.exists() and secret_path.is_file():
        return secret_path.read_text(encoding=DEFAULT_ENCODING).strip()

    logger.warning("Secret file not found for %s at %s", file_key, secret_path)
    return default

## ============================================================
## PROFILE HELPERS
## ============================================================
def _get_profiled_env_str(base_key: str, default: str, *, profile: str) -> str:
    """
        Read a string env value with optional profile override

        Args:
            base_key: Base environment key
            default: Default fallback value
            profile: Active profile name

        Returns:
            Resolved string value
    """

    ## Try profile-specific override first
    override_key = f"{profile.upper()}_{base_key}"
    if os.getenv(override_key) is not None:
        return _get_env_str(override_key, default)

    ## Fallback to base key
    return _get_env_str(base_key, default)

def _get_profiled_env_bool(base_key: str, default: bool, *, profile: str) -> bool:
    """
        Read a boolean env value with optional profile override

        Args:
            base_key: Base environment key
            default: Default fallback value
            profile: Active profile name

        Returns:
            Resolved boolean value
    """

    ## Try profile-specific override first
    override_key = f"{profile.upper()}_{base_key}"
    if os.getenv(override_key) is not None:
        return _get_env_bool(override_key, default)

    ## Fallback to base key
    return _get_env_bool(base_key, default)

def _get_profiled_env_int(base_key: str, default: int, *, profile: str) -> int:
    """
        Read an integer env value with optional profile override

        Args:
            base_key: Base environment key
            default: Default fallback value
            profile: Active profile name

        Returns:
            Resolved integer value
    """

    ## Try profile-specific override first
    override_key = f"{profile.upper()}_{base_key}"
    if os.getenv(override_key) is not None:
        return _get_env_int(override_key, default)

    ## Fallback to base key
    return _get_env_int(base_key, default)

def _get_profiled_env_float(base_key: str, default: float, *, profile: str) -> float:
    """
        Read a float env value with optional profile override

        Args:
            base_key: Base environment key
            default: Default fallback value
            profile: Active profile name

        Returns:
            Resolved float value
    """

    ## Try profile-specific override first
    override_key = f"{profile.upper()}_{base_key}"
    if os.getenv(override_key) is not None:
        return _get_env_float(override_key, default)

    ## Fallback to base key
    return _get_env_float(base_key, default)

## ============================================================
## VALIDATION HELPERS
## ============================================================
def _validate_required_keys(keys: list[str]) -> None:
    """
        Validate required environment keys against placeholders

        Args:
            keys: Required environment key names

        Returns:
            None

        Raises:
            ValueError: If one or more values are placeholders
    """

    ## Collect invalid required keys
    invalid_keys: list[str] = []

    for key in keys:
        value = _get_env_str(key, "")
        if value and _is_placeholder(value):
            invalid_keys.append(key)

    ## Fail on placeholder values
    if invalid_keys:
        raise ValueError("Placeholder values detected for: " + ", ".join(invalid_keys))

def _validate_positive_int(value: int, field_name: str) -> None:
    """
        Validate that an integer is strictly positive

        Args:
            value: Integer value
            field_name: Human-readable field name

        Returns:
            None
        """

    ## Reject non-positive values
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0. Got: {value}")

def _validate_non_negative_int(value: int, field_name: str) -> None:
    """
        Validate that an integer is non-negative

        Args:
            value: Integer value
            field_name: Human-readable field name

        Returns:
            None
        """

    ## Reject negative values
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0. Got: {value}")

def _validate_probability(value: float, field_name: str) -> None:
    """
        Validate that a float is between 0 and 1

        Args:
            value: Float value
            field_name: Human-readable field name

        Returns:
            None
        """

    ## Reject invalid probability values
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]. Got: {value}")

def _validate_provider(value: str, field_name: str) -> None:
    """
        Validate provider values

        Args:
            value: Provider value
            field_name: Human-readable field name

        Returns:
            None
        """

    ## Validate supported providers
    if value not in SUPPORTED_PROVIDERS:
        raise ValueError(f"{field_name} must be one of: {', '.join(SUPPORTED_PROVIDERS)}. Got: {value}")

def _validate_vector_backend(value: str) -> None:
    """
        Validate vector backend values

        Args:
            value: Vector backend value

        Returns:
            None
        """

    ## Validate supported vector backends
    if value not in SUPPORTED_VECTOR_BACKENDS:
        raise ValueError(f"VECTOR_BACKEND must be one of: {', '.join(SUPPORTED_VECTOR_BACKENDS)}. Got: {value}")

def _ensure_directories(paths: list[Path]) -> None:
    """
        Ensure runtime directories exist

        Args:
            paths: Directories to create

        Returns:
            None
        """

    ## Create directories safely
    for directory in paths:
        directory.mkdir(parents=True, exist_ok=True)

def _validate_config(config_obj: AppConfig) -> None:
    """
        Validate the final structured configuration

        Args:
            config_obj: Built application configuration

        Returns:
            None
        """

    ## Validate runtime integers
    _validate_positive_int(config_obj.runtime.request_timeout_seconds, "REQUEST_TIMEOUT_SECONDS")
    _validate_positive_int(config_obj.runtime.max_concurrency, "MAX_CONCURRENCY")
    _validate_positive_int(config_obj.runtime.max_tokens, "MAX_TOKENS")

    ## Validate runtime floats
    _validate_probability(config_obj.runtime.temperature, "TEMPERATURE")

    ## Validate runtime provider
    _validate_provider(config_obj.runtime.provider_default, "PROVIDER_DEFAULT")

    ## Validate retrieval integers
    _validate_positive_int(config_obj.rag.top_k, "TOP_K")
    _validate_positive_int(config_obj.rag.chunk_size, "CHUNK_SIZE")
    _validate_non_negative_int(config_obj.rag.chunk_overlap, "CHUNK_OVERLAP")

    ## Validate retrieval floats
    _validate_probability(config_obj.rag.score_threshold, "SCORE_THRESHOLD")

    ## Validate retrieval provider/backend
    _validate_provider(config_obj.rag.embedding_provider, "EMBEDDING_PROVIDER")
    _validate_vector_backend(config_obj.rag.vector_backend)

    ## Validate supported extensions
    if not config_obj.rag.supported_extensions:
        raise ValueError("SUPPORTED_EXTENSIONS cannot be empty")

    ## Validate cross-field consistency
    if config_obj.rag.chunk_overlap >= config_obj.rag.chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE. "
            f"Got overlap={config_obj.rag.chunk_overlap}, "
            f"chunk_size={config_obj.rag.chunk_size}"
        )

## ============================================================
## EXPORT HELPERS
## ============================================================
def config_to_dict(config_obj: AppConfig) -> dict[str, Any]:
    """
        Convert AppConfig into a serializable dictionary

        Args:
            config_obj: Structured configuration object

        Returns:
            Dictionary representation
        """

    ## Convert dataclass tree
    payload = asdict(config_obj)

    ## Normalize Path recursively
    def _normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: _normalize(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_normalize(item) for item in value]
        return value

    return _normalize(payload)

def config_to_json(config_obj: AppConfig) -> str:
    """
        Convert AppConfig into a JSON string

        Args:
            config_obj: Structured configuration object

        Returns:
            JSON string
        """

    ## Serialize normalized dictionary
    return json.dumps(config_to_dict(config_obj), indent=2, ensure_ascii=False)

## ============================================================
## CONFIG FACTORY
## ============================================================
@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
        Build the global AppConfig from environment variables

        High-level workflow:
            1) Load .env if available
            2) Resolve project root and runtime profile
            3) Read profiled env values
            4) Resolve paths and secrets
            5) Build and validate AppConfig
            6) Ensure runtime directories exist

        Returns:
            Structured AppConfig
        """

    ## Load optional .env
    _load_dotenv_if_present()

    ## Resolve root directory
    root_dir = _find_project_root()

    ## Resolve global metadata
    app_name = _get_env_str("APP_NAME", DEFAULT_APP_NAME)
    app_version = _get_env_str("APP_VERSION", DEFAULT_APP_VERSION)
    environment = _get_env_str("ENVIRONMENT", DEFAULT_ENVIRONMENT).lower()
    profile = _get_env_str("PROFILE", "gpu" if _get_env_bool("USE_GPU", False) else DEFAULT_PROFILE).lower()

    ## Validate placeholder-only env values
    _validate_required_keys(["APP_NAME", "ENVIRONMENT", "PROFILE", "OPENAI_API_KEY", "XAI_API_KEY", "GOOGLE_API_KEY"])

    ## Build execution metadata
    execution = ExecutionMetadata(
        run_id=_get_env_str("RUN_ID", str(uuid.uuid4())),
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        hostname=platform.node(),
        platform_name=SYSTEM_NAME,
        profile=profile,
        environment=environment,
    )

    ## Read runtime settings with profile overrides
    runtime = RuntimeConfig(
        environment=environment,
        profile=profile,
        use_gpu_mode=_get_profiled_env_str("USE_GPU", "true" if profile == "gpu" else "false", profile=profile),
        use_gpu=_get_profiled_env_bool("USE_GPU", profile == "gpu", profile=profile),
        debug=_get_profiled_env_bool("DEBUG", environment == "dev", profile=profile),
        log_level=_get_profiled_env_str("LOG_LEVEL", "INFO", profile=profile),
        provider_default=_get_profiled_env_str("PROVIDER_DEFAULT", "auto", profile=profile).lower(),
        request_timeout_seconds=_get_profiled_env_int("REQUEST_TIMEOUT_SECONDS", 120, profile=profile),
        max_concurrency=_get_profiled_env_int("MAX_CONCURRENCY", 4, profile=profile),
        temperature=_get_profiled_env_float("TEMPERATURE", 0.2, profile=profile),
        max_tokens=_get_profiled_env_int("MAX_TOKENS", 1024, profile=profile),
        allowed_origins=_get_env_list("ALLOWED_ORIGINS", ["*"]),
    )

    ## Read retrieval settings with profile overrides
    rag = RagConfig(
        vector_backend=_get_profiled_env_str("VECTOR_BACKEND", "faiss", profile=profile).lower(),
        embedding_provider=_get_profiled_env_str("EMBEDDING_PROVIDER", "openai", profile=profile).lower(),
        top_k=_get_profiled_env_int("TOP_K", 8, profile=profile),
        chunk_size=_get_profiled_env_int("CHUNK_SIZE", 1024, profile=profile),
        chunk_overlap=_get_profiled_env_int("CHUNK_OVERLAP", 128, profile=profile),
        score_threshold=_get_profiled_env_float("SCORE_THRESHOLD", 0.0, profile=profile),
        recursive_ingestion=_get_profiled_env_bool("RECURSIVE_INGESTION", True, profile=profile),
        supported_extensions=_get_env_list("SUPPORTED_EXTENSIONS", list(SUPPORTED_INPUT_EXTENSIONS)),
    )

    ## Resolve runtime paths
    paths = PathsConfig(
        root_dir=root_dir,
        src_dir=(root_dir / "src").resolve(),
        data_dir=_get_env_path("DATA_DIR", DEFAULT_DATA_DIR, root_dir),
        data_raw_dir=_get_env_path("DATA_RAW_DIR", DEFAULT_DATA_RAW_DIR, root_dir),
        data_processed_dir=_get_env_path("DATA_PROCESSED_DIR", DEFAULT_DATA_PROCESSED_DIR, root_dir),
        data_sqlite_dir=_get_env_path("DATA_SQLITE_DIR", DEFAULT_DATA_SQLITE_DIR, root_dir),
        logs_dir=_get_env_path("LOGS_DIR", DEFAULT_LOGS_DIR, root_dir),
        artifacts_dir=_get_env_path("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR, root_dir),
        artifacts_models_dir=_get_env_path("ARTIFACTS_MODELS_DIR", DEFAULT_ARTIFACTS_MODELS_DIR, root_dir),
        artifacts_vector_store_dir=_get_env_path("ARTIFACTS_VECTOR_STORE_DIR", DEFAULT_ARTIFACTS_VECTOR_STORE_DIR, root_dir),
        artifacts_reports_dir=_get_env_path("ARTIFACTS_REPORTS_DIR", DEFAULT_ARTIFACTS_REPORTS_DIR, root_dir),
        artifacts_evaluations_dir=_get_env_path("ARTIFACTS_EVALUATIONS_DIR", DEFAULT_ARTIFACTS_EVALUATIONS_DIR, root_dir),
        secrets_dir=_get_env_path("SECRETS_DIR", DEFAULT_SECRETS_DIR, root_dir),
        sqlite_db_path=_get_env_path("SQLITE_DB_PATH", f"{DEFAULT_DATA_SQLITE_DIR}/{DEFAULT_SQLITE_FILENAME}", root_dir),
        libreoffice_binary=_get_env_str("LIBREOFFICE_BINARY", DEFAULT_LIBREOFFICE_BINARY),
    )

    ## Ensure runtime directories exist
    _ensure_directories([
        paths.data_dir,
        paths.data_raw_dir,
        paths.data_processed_dir,
        paths.data_sqlite_dir,
        paths.logs_dir,
        paths.artifacts_dir,
        paths.artifacts_models_dir,
        paths.artifacts_vector_store_dir,
        paths.artifacts_reports_dir,
        paths.artifacts_evaluations_dir,
        paths.secrets_dir,
    ])

    ## Read direct keys or file-based secrets
    secrets_path = _get_env_path("LLM_SECRETS_FILE", "", root_dir)

    llm_json = {}
    if secrets_path and secrets_path.exists():
        llm_json = json.loads(secrets_path.read_text(encoding=DEFAULT_ENCODING))

    api_keys = ApiKeysConfig(
        openai_api_key=llm_json.get("openai_api_key", ""),
        xai_api_key=llm_json.get("xai_api_key", ""),
        google_api_key=llm_json.get("google_api_key", ""),
        anthropic_api_key="",
    )

    ## Optional direct access if needed elsewhere
    google_generative_ai_api_key = llm_json.get("google_generative_ai_api_key", "")
    x_api_key = llm_json.get("x_api_key", "")

    ## Build final structured config
    config_obj = AppConfig(
        app_name=app_name,
        app_version=app_version,
        execution=execution,
        runtime=runtime,
        rag=rag,
        paths=paths,
        api_keys=api_keys,
    )

    ## Validate final configuration
    _validate_config(config_obj)

    ## Log concise configuration summary
    logger.info(
        "Configuration loaded | app=%s | env=%s | profile=%s | gpu=%s | provider=%s | run_id=%s",
        config_obj.app_name,
        config_obj.runtime.environment,
        config_obj.runtime.profile,
        config_obj.runtime.use_gpu,
        config_obj.runtime.provider_default,
        config_obj.execution.run_id,
    )
    return config_obj

def load_config() -> AppConfig:
    """
        Backward-compatible alias for configuration loading

        Returns:
            Structured AppConfig
        """

    ## Keep a second public entrypoint for compatibility
    return get_config()

def build_config() -> AppConfig:
    """
        Backward-compatible config builder

        Returns:
            Structured AppConfig
        """

    ## Preserve an additional public entrypoint
    return get_config()

## ============================================================
## GLOBAL CONFIG INSTANCE
## ============================================================
CONFIG: AppConfig = get_config()
config = CONFIG