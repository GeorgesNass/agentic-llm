'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unified configuration loader for rag-drive-gcp: dotenv, env parsing, Google Drive, OCR, GCP, GCS, retrieval, paths, secrets and runtime metadata."
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
from typing import Any, Literal, Optional, Tuple

from src.core.errors import ConfigurationError, log_and_raise_missing_env
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## TYPES
## ============================================================
ProfileName = Literal["local", "gcp"]
OcrMode = Literal["local_docker", "remote_service"]

## ============================================================
## PLACEHOLDER TOKENS
## ============================================================
PLACEHOLDER_PREFIXES: Tuple[str, ...] = ("<YOUR_", "YOUR_", "CHANGE_ME", "REPLACE_ME", "TODO")

## ============================================================
## OS / SYSTEM CONSTANTS
## ============================================================
SYSTEM_NAME = platform.system().lower()
IS_WINDOWS = SYSTEM_NAME == "windows"
IS_LINUX = SYSTEM_NAME == "linux"
IS_MACOS = SYSTEM_NAME == "darwin"
DEFAULT_ENCODING = "utf-8"

## ============================================================
## STABLE DOMAIN CONSTANTS
## ============================================================
DEFAULT_APP_NAME = "rag-drive-gcp"
DEFAULT_APP_VERSION = "1.0.0"
DEFAULT_ENVIRONMENT = "dev"
DEFAULT_PROFILE = "local"

DEFAULT_DATA_DIR = "data"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_SECRETS_DIR = "secrets"

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_TMP_DIR = "data/tmp"
DEFAULT_TRACES_DIR = "data/traces"
DEFAULT_EXPORTS_DIR = "artifacts/exports"
DEFAULT_REPORTS_DIR = "artifacts/reports"

DEFAULT_OCR_MODE = "local_docker"
DEFAULT_OCR_DOCKER_IMAGE = "ocr-universal:latest"
DEFAULT_GCP_REGION = "europe-west1"
DEFAULT_VERTEX_LLM_MODEL = "gemini-1.5-pro"
DEFAULT_VERTEX_EMBED_MODEL = "text-embedding-004"

DEFAULT_GCS_PREFIX_TEXT = "texts/"
DEFAULT_GCS_PREFIX_EMBEDDINGS = "embeddings/"

DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 128
DEFAULT_TOP_K = 8

SUPPORTED_INPUT_EXTENSIONS = (".pdf", ".docx", ".txt", ".md", ".csv", ".json")
SUPPORTED_OCR_MODES = ("local_docker", "remote_service")

## ============================================================
## CONFIG MODELS
## ============================================================
@dataclass(frozen=True)
class ExecutionMetadata:
    """
        Execution metadata

        Args:
            run_id: Unique runtime identifier
            started_at_utc: UTC timestamp when config was built
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
class PathsConfig:
    """
        Filesystem paths configuration

        Args:
            project_root: Project root directory
            src_dir: Source directory
            data_dir: Main data directory
            raw_dir: Raw downloaded files directory
            tmp_dir: Temporary processing directory
            traces_dir: Local pipeline traces directory
            artifacts_dir: Artifacts root directory
            exports_dir: Exports directory
            reports_dir: Reports directory
            logs_dir: Logs directory
            secrets_dir: Secrets directory
    """

    project_root: Path
    src_dir: Path
    data_dir: Path
    raw_dir: Path
    tmp_dir: Path
    traces_dir: Path
    artifacts_dir: Path
    exports_dir: Path
    reports_dir: Path
    logs_dir: Path
    secrets_dir: Path

@dataclass(frozen=True)
class RuntimeConfig:
    """
        Runtime configuration

        Args:
            environment: Environment name
            profile: Active runtime profile
            debug: Whether debug mode is enabled
            log_level: Logging level
            keep_local: Whether local traces are preserved
            request_timeout_seconds: External request timeout
            max_workers: Maximum worker count
            batch_sleep_seconds: Sleep delay between batches
            allowed_origins: Allowed origins for future API usage
    """

    environment: str
    profile: ProfileName
    debug: bool
    log_level: str
    keep_local: bool
    request_timeout_seconds: int
    max_workers: int
    batch_sleep_seconds: float
    allowed_origins: list[str]

@dataclass(frozen=True)
class DriveConfig:
    """
        Google Drive configuration

        Args:
            drive_folder_id: Default Drive folder ID
    """

    drive_folder_id: Optional[str]

@dataclass(frozen=True)
class OcrConfig:
    """
        OCR configuration

        Args:
            ocr_mode: OCR mode
            ocr_docker_image: OCR docker image for local mode
            ocr_service_url: OCR service URL for remote mode
    """

    ocr_mode: OcrMode
    ocr_docker_image: Optional[str]
    ocr_service_url: Optional[str]

@dataclass(frozen=True)
class GcpConfig:
    """
        GCP / Vertex / GCS configuration

        Args:
            gcp_project_id: GCP project ID
            gcp_region: GCP region
            vertex_llm_model: Vertex AI LLM model name
            vertex_embed_model: Vertex AI embedding model name
            gcs_bucket_text: GCS bucket for text artifacts
            gcs_prefix_text: GCS prefix for text artifacts
            gcs_bucket_embeddings: GCS bucket for embeddings artifacts
            gcs_prefix_embeddings: GCS prefix for embeddings artifacts
    """

    gcp_project_id: Optional[str]
    gcp_region: str
    vertex_llm_model: Optional[str]
    vertex_embed_model: Optional[str]
    gcs_bucket_text: Optional[str]
    gcs_prefix_text: str
    gcs_bucket_embeddings: Optional[str]
    gcs_prefix_embeddings: str

@dataclass(frozen=True)
class RetrievalConfig:
    """
        Chunking and retrieval configuration

        Args:
            chunk_size: Chunk size for splitting
            chunk_overlap: Overlap size for splitting
            top_k: Top-K chunks to retrieve
    """

    chunk_size: int
    chunk_overlap: int
    top_k: int

@dataclass(frozen=True)
class SecretsConfig:
    """
        Secret values resolved from env or files

        Args:
            google_application_credentials: Optional service account path content
            api_key: Optional generic API key
    """

    google_application_credentials: str
    api_key: str

@dataclass(frozen=True)
class AppConfig:
    """
        Unified application configuration

        Args:
            app_name: Application name
            app_version: Application version
            execution: Execution metadata
            paths: Filesystem paths configuration
            runtime: Runtime configuration
            drive: Google Drive configuration
            ocr: OCR configuration
            gcp: GCP / Vertex / GCS configuration
            retrieval: Retrieval configuration
            secrets: Secret values
    """

    app_name: str
    app_version: str
    execution: ExecutionMetadata
    paths: PathsConfig
    runtime: RuntimeConfig
    drive: DriveConfig
    ocr: OcrConfig
    gcp: GcpConfig
    retrieval: RetrievalConfig
    secrets: SecretsConfig

## ============================================================
## DOTENV / ENV HELPERS
## ============================================================
def _resolve_project_root() -> Path:
    """
        Resolve the project root directory

        High-level workflow:
            1) Prefer PROJECT_ROOT when explicitly provided
            2) Otherwise derive the root from this file location

        Returns:
            Absolute project root path
    """

    ## Prefer explicit project root override when available
    project_root_raw = os.getenv("PROJECT_ROOT", "").strip()
    return Path(project_root_raw).expanduser().resolve() if project_root_raw else Path(__file__).resolve().parents[2]

def load_dotenv_if_present(env_path: Optional[Path] = None) -> None:
    """
        Load a local .env file if available

        Args:
            env_path: Optional .env path override

        Returns:
            None
    """

    ## Import dotenv lazily to avoid hard dependency issues
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    ## Load project-level .env when present
    dot_env_path = env_path or (_resolve_project_root() / ".env")
    if dot_env_path.exists():
        load_dotenv(dotenv_path=dot_env_path, override=False)

def _is_placeholder(value: str) -> bool:
    """
        Detect placeholder-like values

        Args:
            value: Raw environment value

        Returns:
            True if the value looks like a placeholder
    """

    ## Normalize before inspection
    normalized = value.strip().upper()
    return any(token in normalized for token in PLACEHOLDER_PREFIXES)

def _get_env(name: str, default: Optional[str] = None) -> str:
    """
        Read environment variable safely

        Args:
            name: Environment variable name
            default: Optional default value

        Returns:
            Normalized environment value

        Raises:
            ConfigurationError: If missing and no default provided
    """

    ## Read raw value from process environment
    value = os.getenv(name)
    if value is None:
        if default is None:
            log_and_raise_missing_env([name])
        return default
    return value.strip()

def _get_env_bool(name: str, default: bool) -> bool:
    """
        Parse a boolean environment variable

        Args:
            name: Environment variable name
            default: Default fallback value

        Returns:
            Parsed boolean value
    """

    ## Parse normalized boolean values
    raw = _get_env(name, str(default)).lower()
    if raw in {"true", "1", "yes", "y", "on"}:
        return True
    if raw in {"false", "0", "no", "n", "off"}:
        return False
    raise ConfigurationError(f"Invalid boolean value for {name}: {raw}")

def _get_env_int(name: str, default: int) -> int:
    """
        Parse an integer environment variable

        Args:
            name: Environment variable name
            default: Default fallback value

        Returns:
            Parsed integer value
    """

    ## Parse integer strictly
    try:
        return int(_get_env(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc

def _get_env_float(name: str, default: float) -> float:
    """
        Parse a float environment variable

        Args:
            name: Environment variable name
            default: Default fallback value

        Returns:
            Parsed float value
    """

    ## Parse float strictly
    try:
        return float(_get_env(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a float") from exc

def _get_env_list(name: str, default: Optional[list[str]] = None, *, separator: str = ",") -> list[str]:
    """
        Parse a list-like environment variable

        Args:
            name: Environment variable name
            default: Default fallback list
            separator: Value separator

        Returns:
            Parsed list of strings
    """

    ## Read raw list-like value
    raw = _get_env(name, "")
    if not raw:
        return list(default or [])
    return [item.strip() for item in raw.split(separator) if item.strip()]

def _expand_env_vars(value: str) -> str:
    """
        Expand shell variables and user home in a string

        Args:
            value: Raw string value

        Returns:
            Expanded string
    """

    ## Expand shell variables and user home
    return os.path.expandvars(value)

def _resolve_path(path_value: str, project_root: Path) -> Path:
    """
        Resolve a path against the project root

        Args:
            path_value: Raw path value
            project_root: Project root directory

        Returns:
            Resolved absolute path
    """

    ## Expand shell variables and user home
    path_obj = Path(_expand_env_vars(path_value)).expanduser()
    return path_obj.resolve() if path_obj.is_absolute() else (project_root / path_obj).resolve()

def _read_secret_value(direct_key: str, file_key: str, *, project_root: Path, default: str = "") -> str:
    """
        Read a secret from env directly or from a file path

        Args:
            direct_key: Environment variable containing the secret
            file_key: Environment variable containing the secret file path
            project_root: Project root directory
            default: Default fallback value

        Returns:
            Secret value or default
    """

    ## Prefer direct env value first
    direct_value = _get_env(direct_key, default)
    if direct_value and not _is_placeholder(direct_value):
        return direct_value

    ## Fallback to file-based secret
    secret_file_raw = _get_env(file_key, "")
    if not secret_file_raw:
        return default

    ## Resolve and read secret file when available
    secret_file = _resolve_path(secret_file_raw, project_root)
    if secret_file.exists() and secret_file.is_file():
        return secret_file.read_text(encoding=DEFAULT_ENCODING).strip()
    return default

## ============================================================
## PROFILE HELPERS
## ============================================================
def _get_profiled_env(name: str, default: str, profile: ProfileName) -> str:
    """
        Read an env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active runtime profile

        Returns:
            Resolved string value
    """

    ## Prefer LOCAL_/GCP_ override when present
    override_key = f"{profile.upper()}_{name}"
    return _get_env(override_key, default) if os.getenv(override_key) is not None else _get_env(name, default)

def _get_profiled_env_bool(name: str, default: bool, profile: ProfileName) -> bool:
    """
        Read a boolean env value with optional profile override
    """

    ## Prefer LOCAL_/GCP_ override when present
    override_key = f"{profile.upper()}_{name}"
    return _get_env_bool(override_key, default) if os.getenv(override_key) is not None else _get_env_bool(name, default)

def _get_profiled_env_int(name: str, default: int, profile: ProfileName) -> int:
    """
        Read an integer env value with optional profile override
    """

    ## Prefer LOCAL_/GCP_ override when present
    override_key = f"{profile.upper()}_{name}"
    return _get_env_int(override_key, default) if os.getenv(override_key) is not None else _get_env_int(name, default)

def _get_profiled_env_float(name: str, default: float, profile: ProfileName) -> float:
    """
        Read a float env value with optional profile override
    """

    ## Prefer LOCAL_/GCP_ override when present
    override_key = f"{profile.upper()}_{name}"
    return _get_env_float(override_key, default) if os.getenv(override_key) is not None else _get_env_float(name, default)

## ============================================================
## VALIDATION / BUILD HELPERS
## ============================================================
def _validate_required_placeholders(keys: list[str]) -> None:
    """
        Validate that required env keys are not unresolved placeholders

        Args:
            keys: Environment keys to inspect

        Returns:
            None
    """

    ## Collect invalid placeholder values
    invalid_keys = [key for key in keys if (value := _get_env(key, "")) and _is_placeholder(value)]
    if invalid_keys:
        log_and_raise_missing_env(invalid_keys)

def _validate_positive_int(value: int, field_name: str) -> None:
    """
        Validate that an integer is strictly positive

        Args:
            value: Integer value
            field_name: Human-readable field name

        Returns:
            None
    """

    ## Reject non-positive integers
    if value <= 0:
        raise ConfigurationError(f"{field_name} must be > 0. Got: {value}")

def _validate_non_negative_int(value: int, field_name: str) -> None:
    """
        Validate that an integer is non-negative

        Args:
            value: Integer value
            field_name: Human-readable field name

        Returns:
            None
    """

    ## Reject negative integers
    if value < 0:
        raise ConfigurationError(f"{field_name} must be >= 0. Got: {value}")

def _ensure_directories_exist(paths: list[Path]) -> None:
    """
        Ensure runtime directories exist

        Args:
            paths: Directories to create if missing

        Returns:
            None
    """

    ## Create runtime directories safely
    for directory in paths:
        directory.mkdir(parents=True, exist_ok=True)

def _validate_ocr_mode(value: str) -> OcrMode:
    """
        Validate OCR mode

        Args:
            value: Raw OCR mode

        Returns:
            Validated OCR mode
    """

    ## Restrict OCR mode to supported values
    if value not in SUPPORTED_OCR_MODES:
        raise ConfigurationError(f"OCR_MODE must be one of: {', '.join(SUPPORTED_OCR_MODES)}")
    return value  # type: ignore[return-value]

def _validate_config(config: AppConfig) -> None:
    """
        Validate the final structured configuration

        High-level workflow:
            1) Validate runtime numeric values
            2) Validate retrieval parameters
            3) Validate OCR consistency
            4) Validate GCP/GCS constraints for gcp profile

        Args:
            config: Structured configuration object

        Returns:
            None
    """

    ## Validate runtime numeric parameters
    _validate_positive_int(config.runtime.request_timeout_seconds, "REQUEST_TIMEOUT_SECONDS")
    _validate_positive_int(config.runtime.max_workers, "MAX_WORKERS")
    if config.runtime.batch_sleep_seconds < 0:
        raise ConfigurationError(f"BATCH_SLEEP_SECONDS must be >= 0. Got: {config.runtime.batch_sleep_seconds}")

    ## Validate retrieval parameters
    _validate_positive_int(config.retrieval.chunk_size, "CHUNK_SIZE")
    _validate_non_negative_int(config.retrieval.chunk_overlap, "CHUNK_OVERLAP")
    _validate_positive_int(config.retrieval.top_k, "TOP_K")
    if config.retrieval.chunk_overlap >= config.retrieval.chunk_size:
        raise ConfigurationError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    ## Validate OCR consistency
    if config.ocr.ocr_mode == "local_docker" and not config.ocr.ocr_docker_image:
        raise ConfigurationError("OCR_DOCKER_IMAGE is required when OCR_MODE=local_docker")
    if config.ocr.ocr_mode == "remote_service" and not config.ocr.ocr_service_url:
        raise ConfigurationError("OCR_SERVICE_URL is required when OCR_MODE=remote_service")

    ## Validate GCP profile constraints
    if config.runtime.profile == "gcp":
        required_fields = [
            config.gcp.gcp_project_id, config.gcp.gcp_region, config.gcp.vertex_llm_model,
            config.gcp.vertex_embed_model, config.gcp.gcs_bucket_text, config.gcp.gcs_bucket_embeddings,
        ]
        if not all(required_fields):
            raise ConfigurationError("Missing required GCP / Vertex / GCS configuration for PROFILE=gcp")

## ============================================================
## EXPORT HELPERS
## ============================================================
def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """
        Convert AppConfig into a serializable dictionary

        Args:
            config: Structured configuration object

        Returns:
            Dictionary representation
    """

    ## Convert dataclass tree into a plain dictionary
    payload = asdict(config)

    ## Normalize Path objects recursively
    def _normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: _normalize(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_normalize(item) for item in value]
        return value

    return _normalize(payload)

def config_to_json(config: AppConfig) -> str:
    """
        Convert AppConfig into a JSON string

        Args:
            config: Structured configuration object

        Returns:
            JSON string
        """

    ## Serialize normalized configuration
    return json.dumps(config_to_dict(config), indent=2, ensure_ascii=False)

## ============================================================
## APP FACTORY
## ============================================================
@lru_cache(maxsize=1)
def get_config(env_path: Optional[Path] = None) -> AppConfig:
    """
        Build full application configuration from environment variables

        High-level workflow:
            1) Load optional local .env
            2) Resolve root paths and active profile
            3) Build runtime, Drive, OCR, GCP and retrieval sections
            4) Resolve optional secrets
            5) Validate and cache final configuration

        Args:
            env_path: Optional .env path override

        Returns:
            AppConfig instance
    """

    ## Load optional local .env file first
    load_dotenv_if_present(env_path)

    ## Resolve project root and runtime profile
    project_root = _resolve_project_root()
    environment = _get_env("ENVIRONMENT", DEFAULT_ENVIRONMENT).lower()
    profile_raw = _get_env("PROFILE", DEFAULT_PROFILE).lower()
    profile: ProfileName = "gcp" if profile_raw == "gcp" else "local"

    ## Validate placeholder values where relevant
    _validate_required_placeholders(["ENVIRONMENT", "PROFILE", "API_KEY"])

    ## Build execution metadata
    execution = ExecutionMetadata(
        run_id=_get_env("RUN_ID", str(uuid.uuid4())), started_at_utc=datetime.now(timezone.utc).isoformat(),
        hostname=platform.node(), platform_name=SYSTEM_NAME, profile=profile, environment=environment,
    )

    ## Resolve filesystem paths
    paths = PathsConfig(
        project_root=project_root, src_dir=(project_root / "src").resolve(),
        data_dir=_resolve_path(_get_env("DATA_DIR", DEFAULT_DATA_DIR), project_root),
        raw_dir=_resolve_path(_get_env("RAW_DIR", DEFAULT_RAW_DIR), project_root),
        tmp_dir=_resolve_path(_get_env("TMP_DIR", DEFAULT_TMP_DIR), project_root),
        traces_dir=_resolve_path(_get_env("TRACES_DIR", DEFAULT_TRACES_DIR), project_root),
        artifacts_dir=_resolve_path(_get_env("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR), project_root),
        exports_dir=_resolve_path(_get_env("EXPORTS_DIR", DEFAULT_EXPORTS_DIR), project_root),
        reports_dir=_resolve_path(_get_env("REPORTS_DIR", DEFAULT_REPORTS_DIR), project_root),
        logs_dir=_resolve_path(_get_env("LOGS_DIR", DEFAULT_LOGS_DIR), project_root),
        secrets_dir=_resolve_path(_get_env("SECRETS_DIR", DEFAULT_SECRETS_DIR), project_root),
    )

    ## Ensure runtime directories exist
    _ensure_directories_exist([
        paths.data_dir, paths.raw_dir, paths.tmp_dir, paths.traces_dir, paths.artifacts_dir,
        paths.exports_dir, paths.reports_dir, paths.logs_dir, paths.secrets_dir,
    ])

    ## Build runtime section
    runtime = RuntimeConfig(
        environment=environment, profile=profile,
        debug=_get_profiled_env_bool("DEBUG", environment == "dev", profile),
        log_level=_get_profiled_env("LOG_LEVEL", "INFO", profile),
        keep_local=_get_profiled_env_bool("KEEP_LOCAL", False, profile),
        request_timeout_seconds=_get_profiled_env_int("REQUEST_TIMEOUT_SECONDS", 120, profile),
        max_workers=_get_profiled_env_int("MAX_WORKERS", 4, profile),
        batch_sleep_seconds=_get_profiled_env_float("BATCH_SLEEP_SECONDS", 0.0, profile),
        allowed_origins=_get_env_list("ALLOWED_ORIGINS", ["*"]),
    )

    ## Build Drive section
    drive = DriveConfig(
        drive_folder_id=gcp_json.get("drive_folder_id", "")
    )
    
    ## Build OCR section
    ocr = OcrConfig(
        ocr_mode=_validate_ocr_mode(_get_env("OCR_MODE", DEFAULT_OCR_MODE)),
        ocr_docker_image=_get_env("OCR_DOCKER_IMAGE", DEFAULT_OCR_DOCKER_IMAGE) or None,
        ocr_service_url=_get_env("OCR_SERVICE_URL", "") or None,
    )

    ## Load GCP config JSON
    gcp_config_path = _resolve_path(_get_env("GCP_CONFIG_FILE", ""), project_root)

    gcp_json = {}
    if gcp_config_path.exists():
        gcp_json = json.loads(gcp_config_path.read_text(encoding=DEFAULT_ENCODING))
        
    ## Build GCP section
    gcp = GcpConfig(
        gcp_project_id=gcp_json.get("project_id", ""),
        gcp_region=gcp_json.get("region", DEFAULT_GCP_REGION),
        vertex_llm_model=_get_env("VERTEX_LLM_MODEL", DEFAULT_VERTEX_LLM_MODEL) or None,
        vertex_embed_model=_get_env("VERTEX_EMBED_MODEL", DEFAULT_VERTEX_EMBED_MODEL) or None,
        gcs_bucket_text=gcp_json.get("gcs_bucket_text", ""),
        gcs_prefix_text=_get_env("GCS_PREFIX_TEXT", DEFAULT_GCS_PREFIX_TEXT),
        gcs_bucket_embeddings=gcp_json.get("gcs_bucket_embeddings", ""),
        gcs_prefix_embeddings=_get_env("GCS_PREFIX_EMBEDDINGS", DEFAULT_GCS_PREFIX_EMBEDDINGS),
    )
    
    ## Build retrieval section
    retrieval = RetrievalConfig(
        chunk_size=_get_profiled_env_int("CHUNK_SIZE", DEFAULT_CHUNK_SIZE, profile),
        chunk_overlap=_get_profiled_env_int("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP, profile),
        top_k=_get_profiled_env_int("TOP_K", DEFAULT_TOP_K, profile),
    )

    ## Resolve optional secrets
    secrets = SecretsConfig(
        google_application_credentials=_read_secret_value(
            "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_APPLICATION_CREDENTIALS_FILE", project_root=project_root,
        ),
        api_key=_read_secret_value("API_KEY", "API_KEY_FILE", project_root=project_root),
    )

    ## Build final config
    config = AppConfig(
        app_name=_get_env("APP_NAME", DEFAULT_APP_NAME), app_version=_get_env("APP_VERSION", DEFAULT_APP_VERSION),
        execution=execution, paths=paths, runtime=runtime, drive=drive, ocr=ocr, gcp=gcp,
        retrieval=retrieval, secrets=secrets,
    )

    ## Validate final configuration
    _validate_config(config)

    ## Log concise configuration summary
    logger.info(
        "Configuration loaded | app=%s | env=%s | profile=%s | ocr=%s | gcp_project=%s | top_k=%s | run_id=%s",
        config.app_name, config.runtime.environment, config.runtime.profile, config.ocr.ocr_mode,
        config.gcp.gcp_project_id or "none", config.retrieval.top_k, config.execution.run_id,
    )
    
    return config

def get_settings(env_path: Optional[Path] = None) -> AppConfig:
    """
        Backward-compatible settings factory

        Args:
            env_path: Optional .env path override

        Returns:
            AppConfig instance
    """

    ## Keep compatibility with previous imports
    return get_config(env_path)

def load_config(env_path: Optional[Path] = None) -> AppConfig:
    """
        Backward-compatible alias for configuration loading

        Args:
            env_path: Optional .env path override

        Returns:
            AppConfig instance
    """

    ## Keep compatibility with existing imports
    return get_config(env_path)

def build_config(env_path: Optional[Path] = None) -> AppConfig:
    """
        Backward-compatible config builder

        Args:
            env_path: Optional .env path override

        Returns:
            AppConfig instance
    """

    ## Preserve the original public entrypoint
    return get_config(env_path)

## ============================================================
## PUBLIC SINGLETON CONFIG
## ============================================================
CONFIG: AppConfig = get_config()
config = CONFIG