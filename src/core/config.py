'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unified configuration loader for llm-proxy-gateway: dotenv, env parsing, paths, profiles, providers, catalogs, secrets and runtime metadata."
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

from src.core.errors import ConfigurationError
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## TYPES
## ============================================================
ProviderName = Literal["openai", "gemini", "xai", "auto"]
UseGpuMode = Literal["auto", "true", "false"]

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
CSV_SEPARATOR = ";"

## ============================================================
## STABLE DOMAIN CONSTANTS
## ============================================================
DEFAULT_APP_NAME = "llm-proxy-gateway"
DEFAULT_APP_VERSION = "1.0.0"
DEFAULT_ENVIRONMENT = "dev"
DEFAULT_PROFILE = "cpu"

DEFAULT_DATA_DIR = "data"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_SECRETS_DIR = "secrets"

DEFAULT_DATA_RAW_DIR = "data/raw"
DEFAULT_DATA_PROCESSED_DIR = "data/processed"
DEFAULT_EXPORTS_DIR = "artifacts/exports"
DEFAULT_RESOURCES_DIR = "artifacts/resources"
DEFAULT_CONFIG_DIR = "artifacts/config"

DEFAULT_MODELS_CATALOG_PATH = "artifacts/resources/models_catalog.json"
DEFAULT_PRICING_CATALOG_PATH = "artifacts/resources/pricing_catalog.json"
DEFAULT_SWAGGER_PATH = "artifacts/config/swagger.yaml"

SUPPORTED_COMPLETION_PROVIDERS: Tuple[str, ...] = ("openai", "gemini", "xai")
SUPPORTED_EMBEDDING_PROVIDERS: Tuple[str, ...] = ("openai", "gemini", "xai")
SUPPORTED_EXPORT_FORMATS: Tuple[str, ...] = ("json", "csv")
SUPPORTED_INPUT_EXTENSIONS: Tuple[str, ...] = (".txt", ".json", ".csv", ".md")

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
class RuntimeConfig:
    """
        Runtime configuration

        Args:
            environment: Environment name
            profile: Active runtime profile
            debug: Whether debug mode is enabled
            log_level: Logging level
            use_gpu_mode: Raw GPU mode
            use_gpu: Final GPU decision
            request_timeout_seconds: Global request timeout
            max_retries: Default retry count
            max_concurrency: Maximum concurrent tasks
            batch_sleep_seconds: Sleep delay between batches
            allowed_origins: Allowed origins for future API usage
    """

    environment: str
    profile: str
    debug: bool
    log_level: str
    use_gpu_mode: UseGpuMode
    use_gpu: bool
    request_timeout_seconds: int
    max_retries: int
    max_concurrency: int
    batch_sleep_seconds: float
    allowed_origins: list[str]

@dataclass(frozen=True)
class ProvidersConfig:
    """
        Provider routing configuration

        Args:
            default_completion_provider: Default completion provider
            default_embedding_provider: Default embedding provider
            enabled_completion_providers: Enabled completion providers
            enabled_embedding_providers: Enabled embedding providers
            enable_cost_simulation: Whether cost simulation is enabled
            enable_evaluation: Whether evaluation is enabled
    """

    default_completion_provider: ProviderName
    default_embedding_provider: ProviderName
    enabled_completion_providers: list[str]
    enabled_embedding_providers: list[str]
    enable_cost_simulation: bool
    enable_evaluation: bool

@dataclass(frozen=True)
class CatalogsConfig:
    """
        Catalog file configuration

        Args:
            models_catalog_path: Models catalog JSON path
            pricing_catalog_path: Pricing catalog JSON path
            swagger_path: Swagger/OpenAPI YAML path
    """

    models_catalog_path: Path
    pricing_catalog_path: Path
    swagger_path: Path

@dataclass(frozen=True)
class PathsConfig:
    """
        Filesystem paths configuration

        Args:
            project_root: Project root directory
            src_dir: Source directory
            data_dir: Main data directory
            data_raw_dir: Raw input data directory
            data_processed_dir: Processed data directory
            exports_dir: Export output directory
            logs_dir: Logs directory
            secrets_dir: Secrets directory
            artifacts_dir: Artifacts root directory
            resources_dir: Resources directory
            config_dir: Config artifact directory
    """

    project_root: Path
    src_dir: Path
    data_dir: Path
    data_raw_dir: Path
    data_processed_dir: Path
    exports_dir: Path
    logs_dir: Path
    secrets_dir: Path
    artifacts_dir: Path
    resources_dir: Path
    config_dir: Path

@dataclass(frozen=True)
class SecretsConfig:
    """
        Secret values resolved from env or files

        Args:
            openai_api_key: OpenAI API key
            gemini_api_key: Gemini API key
            xai_api_key: xAI API key
            api_key: Internal API key
    """

    openai_api_key: str
    gemini_api_key: str
    xai_api_key: str
    api_key: str

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
            data_processed_dir: Path to processed files
            exports_dir: Path to exports directory
            models_catalog_path: Path to models catalog JSON
            pricing_catalog_path: Path to pricing catalog JSON
            execution: Execution metadata
            runtime: Runtime configuration
            providers: Provider routing configuration
            catalogs: Catalog file configuration
            paths: Filesystem paths configuration
            secrets: Secret values
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
    execution: ExecutionMetadata
    runtime: RuntimeConfig
    providers: ProvidersConfig
    catalogs: CatalogsConfig
    paths: PathsConfig
    secrets: SecretsConfig

## ============================================================
## DOTENV / ENV HELPERS
## ============================================================
def _resolve_project_root() -> Path:
    """
        Resolve the project root path

        Returns:
            Absolute project root path
    """

    ## Prefer explicit project root override when available
    project_root_raw = os.getenv("PROJECT_ROOT", "").strip()
    return Path(project_root_raw).expanduser().resolve() if project_root_raw else Path(__file__).resolve().parents[2]

def _load_dotenv_if_present() -> None:
    """
        Load a local .env file if available

        Returns:
            None
    """

    ## Import dotenv lazily to avoid hard dependency issues
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    ## Load only when a project-level .env file exists
    env_path = _resolve_project_root() / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)

def _is_placeholder(value: str) -> bool:
    """
        Detect placeholder-like values

        Args:
            value: Raw environment value

        Returns:
            True if the value looks like a placeholder
    """

    ## Normalize before checking placeholder tokens
    normalized = value.strip().upper()
    return any(token in normalized for token in PLACEHOLDER_PREFIXES)

def _get_env(name: str, default: str = "") -> str:
    """
        Read an environment variable safely

        Args:
            name: Environment variable name
            default: Default fallback value

        Returns:
            Normalized environment value
    """

    ## Read raw value from process environment
    value = os.getenv(name, default)
    return (value if value is not None else default).strip()

def _get_env_bool(name: str, default: bool) -> bool:
    """
        Parse a boolean environment variable

        Args:
            name: Environment variable name
            default: Default fallback value

        Returns:
            Parsed boolean value

        Raises:
            ConfigurationError: If invalid
    """

    ## Read and normalize raw value
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

        Raises:
            ConfigurationError: If invalid
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

        Raises:
            ConfigurationError: If invalid
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
            separator: Raw value separator

        Returns:
            Parsed list of strings
    """

    ## Read and normalize list-like value
    raw = _get_env(name, "")
    if not raw:
        return list(default or [])
    return [item.strip() for item in raw.split(separator) if item.strip()]

def _expand_env_vars(value: str) -> str:
    """
        Expand shell variables in a string

        Args:
            value: Raw string value

        Returns:
            Expanded string
    """

    ## Expand shell variables such as %USERPROFILE% or $HOME
    return os.path.expandvars(value)

def _resolve_path(path_value: str, project_root: Path) -> Path:
    """
        Resolve a path against the project root

        Args:
            path_value: Raw path value
            project_root: Project root directory

        Returns:
            Absolute resolved Path
    """

    ## Expand shell variables and user home
    path_obj = Path(_expand_env_vars(path_value)).expanduser()
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (project_root / path_obj).resolve()

def _get_env_path(name: str, default: str, project_root: Path) -> Path:
    """
        Read and resolve a path environment variable

        Args:
            name: Environment variable name
            default: Default path value
            project_root: Project root directory

        Returns:
            Resolved path
    """

    ## Resolve env override or default path
    return _resolve_path(_get_env(name, default), project_root)

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

    ## Prefer direct env secret value first
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
def _get_profiled_env(name: str, default: str, profile: str) -> str:
    """
        Read an env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active runtime profile

        Returns:
            Resolved string value
    """

    ## Prefer profile-specific override when present
    override_key = f"{profile.upper()}_{name}"
    return _get_env(override_key, default) if os.getenv(override_key) is not None else _get_env(name, default)

def _get_profiled_env_bool(name: str, default: bool, profile: str) -> bool:
    """
        Read a boolean env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active runtime profile

        Returns:
            Parsed boolean value
    """

    ## Prefer profile-specific override when present
    override_key = f"{profile.upper()}_{name}"
    return _get_env_bool(override_key, default) if os.getenv(override_key) is not None else _get_env_bool(name, default)

def _get_profiled_env_int(name: str, default: int, profile: str) -> int:
    """
        Read an integer env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active runtime profile

        Returns:
            Parsed integer value
    """

    ## Prefer profile-specific override when present
    override_key = f"{profile.upper()}_{name}"
    return _get_env_int(override_key, default) if os.getenv(override_key) is not None else _get_env_int(name, default)

def _get_profiled_env_float(name: str, default: float, profile: str) -> float:
    """
        Read a float env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active runtime profile

        Returns:
            Parsed float value
    """

    ## Prefer profile-specific override when present
    override_key = f"{profile.upper()}_{name}"
    return _get_env_float(override_key, default) if os.getenv(override_key) is not None else _get_env_float(name, default)

## ============================================================
## VALIDATION / BUILD HELPERS
## ============================================================
def _detect_gpu_requested(mode: UseGpuMode) -> bool:
    """
        Determine whether GPU usage is requested and available

        Args:
            mode: Raw GPU mode value

        Returns:
            Final GPU usage decision
    """

    ## Respect explicit override
    if mode == "true":
        return True
    if mode == "false":
        return False

    ## Auto mode falls back to torch detection
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())

def _validate_required_placeholders(keys: list[str]) -> None:
    """
        Validate that required values are not unresolved placeholders

        Args:
            keys: Environment keys to inspect

        Returns:
            None

        Raises:
            ConfigurationError: If placeholders are detected
    """

    ## Collect required keys still using placeholder values
    invalid_keys = [key for key in keys if (value := _get_env(key, "")) and _is_placeholder(value)]
    if invalid_keys:
        raise ConfigurationError("Placeholder values detected for: " + ", ".join(invalid_keys))

def _validate_positive_int(value: int, field_name: str) -> None:
    """
        Validate that an integer is strictly positive

        Args:
            value: Value to validate
            field_name: Human-readable field name

        Returns:
            None

        Raises:
            ConfigurationError: If invalid
    """

    ## Reject non-positive integers
    if value <= 0:
        raise ConfigurationError(f"{field_name} must be > 0. Got: {value}")

def _validate_non_negative_int(value: int, field_name: str) -> None:
    """
        Validate that an integer is non-negative

        Args:
            value: Value to validate
            field_name: Human-readable field name

        Returns:
            None

        Raises:
            ConfigurationError: If invalid
    """

    ## Reject negative integers
    if value < 0:
        raise ConfigurationError(f"{field_name} must be >= 0. Got: {value}")

def _validate_non_negative_float(value: float, field_name: str) -> None:
    """
        Validate that a float is non-negative

        Args:
            value: Value to validate
            field_name: Human-readable field name

        Returns:
            None

        Raises:
            ConfigurationError: If invalid
    """

    ## Reject negative floats
    if value < 0.0:
        raise ConfigurationError(f"{field_name} must be >= 0. Got: {value}")

def _validate_provider(value: str, field_name: str, supported: tuple[str, ...]) -> ProviderName:
    """
        Validate a provider value

        Args:
            value: Raw provider name
            field_name: Human-readable field name
            supported: Supported provider values

        Returns:
            Validated provider name

        Raises:
            ConfigurationError: If unsupported
    """

    ## Restrict providers to supported values
    if value not in supported:
        raise ConfigurationError(f"{field_name} must be one of: {', '.join(supported)}")
    return value  # type: ignore[return-value]

def _ensure_directories_exist(paths: list[Path]) -> None:
    """
        Ensure runtime directories exist

        Args:
            paths: Directories to create if missing

        Returns:
            None
    """

    ## Create all runtime directories safely
    for directory in paths:
        directory.mkdir(parents=True, exist_ok=True)

def _validate_config(config: AppConfig) -> None:
    """
        Validate the final structured configuration

        Args:
            config: Structured application configuration

        Returns:
            None
        """

    ## Validate runtime numeric parameters
    _validate_non_negative_int(config.runtime.random_seed, "RANDOM_SEED")
    _validate_positive_int(config.runtime.max_workers, "MAX_WORKERS")
    _validate_positive_int(config.runtime.batch_size, "BATCH_SIZE")
    _validate_non_negative_float(config.runtime.batch_sleep_seconds, "BATCH_SLEEP_SECONDS")

    ## Validate clustering numeric parameters
    _validate_positive_int(config.clustering.default_n_clusters, "DEFAULT_N_CLUSTERS")
    _validate_positive_int(config.clustering.pca_components, "PCA_COMPONENTS")

    ## Validate MLflow configuration basics
    if config.mlflow.enabled and not config.mlflow.tracking_uri:
        raise ConfigurationError("MLFLOW_TRACKING_URI cannot be empty when MLflow is enabled")

## ============================================================
## EXPORT HELPERS
## ============================================================
def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """
        Convert AppConfig into a serializable dictionary

        Args:
            config: Structured configuration object

        Returns:
            Serializable dictionary
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

    ## Serialize normalized configuration to JSON
    return json.dumps(config_to_dict(config), indent=2, ensure_ascii=False)

## ============================================================
## CONFIG FACTORY
## ============================================================
@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
        Build full application configuration from environment variables

        High-level workflow:
            1) Load optional project-level .env
            2) Resolve project root and active profile
            3) Build execution, paths, mlflow, runtime and clustering sections
            4) Resolve optional secrets
            5) Validate and cache the final AppConfig

        Returns:
            AppConfig instance
    """

    ## Load optional local .env file first
    _load_dotenv_if_present()

    ## Resolve project root and active runtime profile
    project_root = _resolve_project_root()
    environment = _get_env("ENVIRONMENT", DEFAULT_ENVIRONMENT).lower()
    profile = _get_env("PROFILE", "gpu" if _get_env("USE_GPU", "auto").lower() != "false" else DEFAULT_PROFILE).lower()

    ## Validate placeholder values where relevant
    _validate_required_placeholders(["ENVIRONMENT", "PROFILE", "MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"])

    ## Build execution metadata
    execution = ExecutionMetadata(
        run_id=_get_env("RUN_ID", str(uuid.uuid4())),
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        hostname=platform.node(),
        platform_name=SYSTEM_NAME,
        profile=profile,
        environment=environment,
    )

    ## Resolve main root folders
    data_dir = _get_env_path("DATA_DIR", DEFAULT_DATA_DIR, project_root)
    logs_dir = _get_env_path("LOGS_DIR", DEFAULT_LOGS_DIR, project_root)
    artifacts_dir = _get_env_path("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR, project_root)
    secrets_dir = _get_env_path("SECRETS_DIR", DEFAULT_SECRETS_DIR, project_root)

    ## Build nested filesystem paths
    raw_dir = _get_env_path("RAW_DIR", DEFAULT_RAW_DIR, project_root)
    interim_dir = _get_env_path("INTERIM_DIR", DEFAULT_INTERIM_DIR, project_root)
    processed_dir = _get_env_path("PROCESSED_DIR", DEFAULT_PROCESSED_DIR, project_root)

    paths = PathsConfig(
        project_root=project_root,
        src_dir=(project_root / "src").resolve(),
        data_dir=data_dir,
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        interim_structured_dir=_get_env_path("INTERIM_STRUCTURED_DIR", str(interim_dir / DEFAULT_STRUCTURED_DIRNAME), project_root),
        interim_datasets_dir=_get_env_path("INTERIM_DATASETS_DIR", str(interim_dir / DEFAULT_DATASETS_DIRNAME), project_root),
        processed_features_dir=_get_env_path("PROCESSED_FEATURES_DIR", str(processed_dir / DEFAULT_FEATURES_DIRNAME), project_root),
        processed_error_analysis_dir=_get_env_path("PROCESSED_ERROR_ANALYSIS_DIR", str(processed_dir / DEFAULT_ERROR_ANALYSIS_DIRNAME), project_root),
        artifacts_dir=artifacts_dir,
        artifacts_models_dir=_get_env_path("ARTIFACTS_MODELS_DIR", DEFAULT_MODELS_DIR, project_root),
        artifacts_exports_dir=_get_env_path("ARTIFACTS_EXPORTS_DIR", DEFAULT_EXPORTS_DIR, project_root),
        artifacts_resources_dir=_get_env_path("ARTIFACTS_RESOURCES_DIR", DEFAULT_RESOURCES_DIR, project_root),
        artifacts_config_dir=_get_env_path("ARTIFACTS_CONFIG_DIR", DEFAULT_CONFIG_DIR, project_root),
        logs_dir=logs_dir,
        secrets_dir=secrets_dir,
    )

    ## Ensure all runtime directories exist
    _ensure_directories_exist([
        paths.data_dir, paths.raw_dir, paths.interim_dir, paths.processed_dir,
        paths.interim_structured_dir, paths.interim_datasets_dir, paths.processed_features_dir,
        paths.processed_error_analysis_dir, paths.artifacts_dir, paths.artifacts_models_dir,
        paths.artifacts_exports_dir, paths.artifacts_resources_dir, paths.artifacts_config_dir,
        paths.logs_dir, paths.secrets_dir,
    ])

    ## Build MLflow configuration
    default_mlflow_uri = str((project_root / DEFAULT_MLFLOW_DIR).resolve())
    mlflow = MlflowConfig(
        tracking_uri=_get_profiled_env("MLFLOW_TRACKING_URI", default_mlflow_uri, profile),
        experiment_name=_get_profiled_env("MLFLOW_EXPERIMENT_NAME", DEFAULT_MLFLOW_EXPERIMENT, profile),
        enabled=_get_profiled_env_bool("MLFLOW_ENABLED", True, profile),
        artifact_location=_get_profiled_env("MLFLOW_ARTIFACT_LOCATION", str(paths.artifacts_dir), profile),
    )

    ## Resolve runtime configuration
    use_gpu_mode_raw = _get_profiled_env("USE_GPU", "auto", profile).lower()
    if use_gpu_mode_raw not in {"auto", "true", "false"}:
        raise ConfigurationError("USE_GPU must be auto|true|false")
    use_gpu_mode = use_gpu_mode_raw
    use_gpu = _detect_gpu_requested(use_gpu_mode)

    runtime = RuntimeConfig(
        environment=environment,
        profile=profile,
        debug=_get_profiled_env_bool("DEBUG", environment == "dev", profile),
        log_level=_get_profiled_env("LOG_LEVEL", "INFO", profile),
        use_gpu_mode=use_gpu_mode,
        use_gpu=use_gpu,
        random_seed=_get_profiled_env_int("RANDOM_SEED", 42, profile),
        max_workers=_get_profiled_env_int("MAX_WORKERS", 4, profile),
        batch_size=_get_profiled_env_int("BATCH_SIZE", 32, profile),
        batch_sleep_seconds=_get_profiled_env_float("BATCH_SLEEP_SECONDS", 0.0, profile),
        allowed_origins=_get_env_list("ALLOWED_ORIGINS", ["*"]),
    )

    ## Build clustering configuration
    clustering = ClusteringConfig(
        default_algorithm=_get_profiled_env("DEFAULT_ALGORITHM", "kmeans", profile),
        default_n_clusters=_get_profiled_env_int("DEFAULT_N_CLUSTERS", 3, profile),
        use_pca=_get_profiled_env_bool("USE_PCA", True, profile),
        pca_components=_get_profiled_env_int("PCA_COMPONENTS", 2, profile),
        scale_features=_get_profiled_env_bool("SCALE_FEATURES", True, profile),
    )

    ## Resolve optional secrets from direct env or files
    secrets = SecretsConfig(
        mlflow_tracking_username=_read_secret_value("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_USERNAME_FILE", project_root=project_root),
        mlflow_tracking_password=_read_secret_value("MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_PASSWORD_FILE", project_root=project_root),
    )

    ## Build and validate final application config
    config = AppConfig(
        app_name=_get_env("APP_NAME", DEFAULT_APP_NAME),
        app_version=_get_env("APP_VERSION", DEFAULT_APP_VERSION),
        execution=execution,
        paths=paths,
        mlflow=mlflow,
        runtime=runtime,
        clustering=clustering,
        secrets=secrets,
    )

    ## Validate final configuration
    _validate_config(config)

    ## Log a compact configuration summary
    logger.info(
        "Configuration loaded | app=%s | env=%s | profile=%s | gpu=%s | mlflow=%s | experiment=%s | run_id=%s",
        config.app_name,
        config.runtime.environment,
        config.runtime.profile,
        config.runtime.use_gpu,
        config.mlflow.enabled,
        config.mlflow.experiment_name,
        config.execution.run_id,
    )
    return config

def load_config() -> AppConfig:
    """
        Backward-compatible alias for configuration loading

        Returns:
            AppConfig instance
    """

    ## Keep compatibility with existing imports
    return get_config()

def build_config() -> AppConfig:
    """
        Backward-compatible config builder

        Returns:
            AppConfig instance
    """

    ## Preserve an additional public entrypoint
    return get_config()

## ============================================================
## PUBLIC SINGLETON CONFIG
## ============================================================
CONFIG: AppConfig = get_config()
config = CONFIG