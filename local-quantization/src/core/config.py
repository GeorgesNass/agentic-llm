'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unified configuration loader for local-quantization: dotenv, env parsing, paths, profiles, quantization, benchmarking, export and runtime metadata."
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
UseGpuMode = Literal["auto", "true", "false"]
ProfileName = Literal["cpu", "gpu"]
QuantizationBackend = Literal["gguf", "awq", "gptq", "bitsandbytes", "onnx"]

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
DEFAULT_APP_NAME = "local-quantization"
DEFAULT_APP_VERSION = "1.0.0"
DEFAULT_ENVIRONMENT = "dev"
DEFAULT_PROFILE = "gpu"

DEFAULT_DATA_DIR = "data"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_SECRETS_DIR = "secrets"
DEFAULT_MODELS_DIR = "models"

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_CALIBRATION_DIR = "data/calibration"
DEFAULT_INTERIM_DIR = "data/interim"
DEFAULT_PROCESSED_DIR = "data/processed"

DEFAULT_MODELS_BASE_DIR = "models/base"
DEFAULT_MODELS_ADAPTERS_DIR = "models/adapters"

DEFAULT_RUNS_DIR = "artifacts/runs"
DEFAULT_EXPORTS_DIR = "artifacts/exports"
DEFAULT_BENCHMARKS_DIR = "artifacts/benchmarks"
DEFAULT_REPORTS_DIR = "artifacts/reports"

DEFAULT_BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_BACKEND = "gguf"
DEFAULT_CALIBRATION_FILE = "calibration.jsonl"
DEFAULT_OUTPUT_NAME = "quantized_model"

SUPPORTED_BACKENDS: Tuple[str, ...] = ("gguf", "awq", "gptq", "bitsandbytes", "onnx")
SUPPORTED_INPUT_EXTENSIONS: Tuple[str, ...] = (".json", ".jsonl", ".txt", ".csv")
SUPPORTED_EXPORT_EXTENSIONS: Tuple[str, ...] = (".gguf", ".awq", ".gptq", ".bin", ".onnx", ".safetensors")

def _read_json_secret(secret_file: Path) -> dict[str, Any]:
    """
        Read a JSON secret file safely

        Args:
            secret_file (Path): Path to the JSON file

        Returns:
            dict[str, Any]: Parsed JSON content or empty dict
    """

    if not secret_file.exists():
        return {}

    try:
        return json.loads(secret_file.read_text(encoding=DEFAULT_ENCODING))
    except Exception:
        return {}
        
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
            raw_dir: Raw data directory
            calibration_dir: Calibration dataset directory
            interim_dir: Interim data directory
            processed_dir: Processed data directory
            models_dir: Models root directory
            models_base_dir: Base model checkpoints directory
            models_adapters_dir: Optional adapters directory
            artifacts_dir: Artifacts root directory
            runs_dir: Run metadata directory
            exports_dir: Quantized exports directory
            benchmarks_dir: Benchmark outputs directory
            reports_dir: Reports directory
            logs_dir: Logs directory
            secrets_dir: Secrets directory
            calibration_file: Calibration file path
    """

    project_root: Path
    src_dir: Path
    data_dir: Path
    raw_dir: Path
    calibration_dir: Path
    interim_dir: Path
    processed_dir: Path
    models_dir: Path
    models_base_dir: Path
    models_adapters_dir: Path
    artifacts_dir: Path
    runs_dir: Path
    exports_dir: Path
    benchmarks_dir: Path
    reports_dir: Path
    logs_dir: Path
    secrets_dir: Path
    calibration_file: Path

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
            allowed_origins: Allowed HTTP origins for future API usage
    """

    environment: str
    profile: ProfileName
    debug: bool
    log_level: str
    use_gpu_mode: UseGpuMode
    use_gpu: bool
    allowed_origins: list[str]

@dataclass(frozen=True)
class QuantizationConfig:
    """
        Quantization configuration

        Args:
            base_model_name: Base model HF id or local path
            adapter_path: Optional adapter path
            backend: Quantization backend
            quant_bits: Quantization bits
            group_size: Backend group size
            desc_act: Whether activation-desc sorting is enabled
            sym: Whether symmetric quantization is enabled
            use_calibration: Whether calibration is enabled
            output_name: Output artifact base name
            save_tokenizer: Whether tokenizer is exported
    """

    base_model_name: str
    adapter_path: Optional[Path]
    backend: QuantizationBackend
    quant_bits: int
    group_size: int
    desc_act: bool
    sym: bool
    use_calibration: bool
    output_name: str
    save_tokenizer: bool

@dataclass(frozen=True)
class BenchmarkConfig:
    """
        Benchmark configuration

        Args:
            enabled: Whether benchmark stage is enabled
            prompt_tokens: Input prompt token target
            max_new_tokens: Max generated tokens
            num_runs: Number of repeated runs
            warmup_runs: Number of warmup runs
            deterministic: Whether deterministic decoding is enabled
    """

    enabled: bool
    prompt_tokens: int
    max_new_tokens: int
    num_runs: int
    warmup_runs: int
    deterministic: bool

@dataclass(frozen=True)
class SecretsConfig:
    """
        Secret values resolved from env or files

        Args:
            huggingface_token: Optional Hugging Face token
            api_key: Optional generic API key
    """

    huggingface_token: str
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
            quantization: Quantization configuration
            benchmark: Benchmark configuration
            secrets: Secret values
    """

    app_name: str
    app_version: str
    execution: ExecutionMetadata
    paths: PathsConfig
    runtime: RuntimeConfig
    quantization: QuantizationConfig
    benchmark: BenchmarkConfig
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

    ## Load project-level .env when present
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

    ## Read raw value from environment
    value = os.getenv(name)
    if value is None:
        if default is None:
            raise ConfigurationError(f"Missing environment variable: {name}")
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

    ## Read raw list value
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

    ## Expand shell variables
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

        High-level workflow:
            1) Prefer direct env value
            2) Fallback to secret file path
            3) Return default when nothing is available

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

    ## Prefer CPU_/GPU_ override when present
    override_key = f"{profile.upper()}_{name}"
    return _get_env(override_key, default) if os.getenv(override_key) is not None else _get_env(name, default)

def _get_profiled_env_bool(name: str, default: bool, profile: ProfileName) -> bool:
    """
        Read a boolean env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active runtime profile

        Returns:
            Parsed boolean value
    """

    ## Prefer CPU_/GPU_ override when present
    override_key = f"{profile.upper()}_{name}"
    return _get_env_bool(override_key, default) if os.getenv(override_key) is not None else _get_env_bool(name, default)

def _get_profiled_env_int(name: str, default: int, profile: ProfileName) -> int:
    """
        Read an integer env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active runtime profile

        Returns:
            Parsed integer value
    """

    ## Prefer CPU_/GPU_ override when present
    override_key = f"{profile.upper()}_{name}"
    return _get_env_int(override_key, default) if os.getenv(override_key) is not None else _get_env_int(name, default)

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
        Validate that required env keys are not unresolved placeholders

        Args:
            keys: Environment keys to inspect

        Returns:
            None
    """

    ## Collect invalid placeholder values
    invalid_keys = [key for key in keys if (value := _get_env(key, "")) and _is_placeholder(value)]
    if invalid_keys:
        raise ConfigurationError("Placeholder values detected for: " + ", ".join(invalid_keys))

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

def _validate_backend(value: str) -> QuantizationBackend:
    """
        Validate the quantization backend

        Args:
            value: Raw backend name

        Returns:
            Validated backend
    """

    ## Restrict backends to supported values
    if value not in SUPPORTED_BACKENDS:
        raise ConfigurationError(f"BACKEND must be one of: {', '.join(SUPPORTED_BACKENDS)}")
    
    return value  # type: ignore[return-value]

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

def _validate_config(config: AppConfig) -> None:
    """
        Validate the final structured configuration

        High-level workflow:
            1) Validate quantization parameters
            2) Validate benchmark parameters
            3) Validate backend-specific consistency
            4) Validate optional adapter path

        Args:
            config: Structured configuration object

        Returns:
            None
    """

    ## Validate quantization parameters
    _validate_positive_int(config.quantization.quant_bits, "QUANT_BITS")
    _validate_positive_int(config.quantization.group_size, "GROUP_SIZE")
    if config.quantization.quant_bits not in {2, 3, 4, 8, 16}:
        raise ConfigurationError(f"QUANT_BITS must be one of 2, 3, 4, 8, 16. Got: {config.quantization.quant_bits}")

    ## Validate backend-specific constraints
    if config.quantization.backend == "bitsandbytes" and config.quantization.quant_bits not in {4, 8}:
        raise ConfigurationError("bitsandbytes backend only supports QUANT_BITS in {4, 8}")
    if config.quantization.backend == "onnx" and config.quantization.use_calibration and config.quantization.quant_bits not in {8}:
        raise ConfigurationError("ONNX calibration is only supported here with QUANT_BITS=8")

    ## Validate benchmark parameters
    _validate_positive_int(config.benchmark.prompt_tokens, "PROMPT_TOKENS")
    _validate_positive_int(config.benchmark.max_new_tokens, "MAX_NEW_TOKENS")
    _validate_positive_int(config.benchmark.num_runs, "NUM_RUNS")
    _validate_positive_int(config.benchmark.warmup_runs, "WARMUP_RUNS")

    ## Validate optional adapter path
    if config.quantization.adapter_path is not None and not config.quantization.adapter_path.exists():
        raise ConfigurationError(f"ADAPTER_PATH does not exist: {config.quantization.adapter_path}")

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
def get_config() -> AppConfig:
    """
        Build full application configuration from environment variables

        High-level workflow:
            1) Load optional local .env
            2) Resolve root paths and active profile
            3) Build runtime, quantization and benchmark sections
            4) Resolve optional secrets
            5) Validate and cache final configuration

        Args:
            None

        Returns:
            AppConfig instance
    """

    ## Load optional local .env file first
    _load_dotenv_if_present()

    ## Resolve project root and runtime profile
    project_root = _resolve_project_root()
    environment = _get_env("ENVIRONMENT", DEFAULT_ENVIRONMENT).lower()
    profile_raw = _get_env("PROFILE", DEFAULT_PROFILE).lower()
    profile: ProfileName = "cpu" if profile_raw == "cpu" else "gpu"

    ## Validate placeholder values where relevant
    _validate_required_placeholders(["ENVIRONMENT", "PROFILE", "HUGGINGFACE_TOKEN", "API_KEY"])

    ## Build execution metadata
    execution = ExecutionMetadata(
        run_id=_get_env("RUN_ID", str(uuid.uuid4())),
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        hostname=platform.node(),
        platform_name=SYSTEM_NAME,
        profile=profile,
        environment=environment,
    )

    ## Resolve filesystem paths
    paths = PathsConfig(
        project_root=project_root, src_dir=(project_root / "src").resolve(),
        data_dir=_get_env_path("DATA_DIR", DEFAULT_DATA_DIR, project_root),
        raw_dir=_get_env_path("RAW_DIR", DEFAULT_RAW_DIR, project_root),
        calibration_dir=_get_env_path("CALIBRATION_DIR", DEFAULT_CALIBRATION_DIR, project_root),
        interim_dir=_get_env_path("INTERIM_DIR", DEFAULT_INTERIM_DIR, project_root),
        processed_dir=_get_env_path("PROCESSED_DIR", DEFAULT_PROCESSED_DIR, project_root),
        models_dir=_get_env_path("MODELS_DIR", DEFAULT_MODELS_DIR, project_root),
        models_base_dir=_get_env_path("MODELS_BASE_DIR", DEFAULT_MODELS_BASE_DIR, project_root),
        models_adapters_dir=_get_env_path("MODELS_ADAPTERS_DIR", DEFAULT_MODELS_ADAPTERS_DIR, project_root),
        artifacts_dir=_get_env_path("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR, project_root),
        runs_dir=_get_env_path("RUNS_DIR", DEFAULT_RUNS_DIR, project_root),
        exports_dir=_get_env_path("EXPORTS_DIR", DEFAULT_EXPORTS_DIR, project_root),
        benchmarks_dir=_get_env_path("BENCHMARKS_DIR", DEFAULT_BENCHMARKS_DIR, project_root),
        reports_dir=_get_env_path("REPORTS_DIR", DEFAULT_REPORTS_DIR, project_root),
        logs_dir=_get_env_path("LOGS_DIR", DEFAULT_LOGS_DIR, project_root),
        secrets_dir=_get_env_path("SECRETS_DIR", DEFAULT_SECRETS_DIR, project_root),
        calibration_file=_get_env_path("CALIBRATION_FILE", f"{DEFAULT_CALIBRATION_DIR}/{DEFAULT_CALIBRATION_FILE}", project_root),
    )

    ## Ensure runtime directories exist
    _ensure_directories_exist([
        paths.data_dir, paths.raw_dir, paths.calibration_dir, paths.interim_dir, paths.processed_dir,
        paths.models_dir, paths.models_base_dir, paths.models_adapters_dir, paths.artifacts_dir,
        paths.runs_dir, paths.exports_dir, paths.benchmarks_dir, paths.reports_dir,
        paths.logs_dir, paths.secrets_dir,
    ])

    ## Resolve runtime section
    use_gpu_mode_raw = _get_profiled_env("USE_GPU", "true" if profile == "gpu" else "false", profile).lower()
    if use_gpu_mode_raw not in {"auto", "true", "false"}:
        raise ConfigurationError("USE_GPU must be auto|true|false")
    use_gpu_mode: UseGpuMode = use_gpu_mode_raw  # type: ignore[assignment]

    runtime = RuntimeConfig(
        environment=environment, profile=profile,
        debug=_get_profiled_env_bool("DEBUG", environment == "dev", profile),
        log_level=_get_profiled_env("LOG_LEVEL", "INFO", profile),
        use_gpu_mode=use_gpu_mode, use_gpu=_detect_gpu_requested(use_gpu_mode),
        allowed_origins=_get_env_list("ALLOWED_ORIGINS", ["*"]),
    )

    ## Resolve quantization section
    adapter_path_str = _get_env("ADAPTER_PATH", "")
    quantization = QuantizationConfig(
        base_model_name=_get_env("BASE_MODEL_NAME", DEFAULT_BASE_MODEL_NAME),
        adapter_path=_resolve_path(adapter_path_str, project_root) if adapter_path_str else None,
        backend=_validate_backend(_get_env("BACKEND", DEFAULT_BACKEND).lower()),
        quant_bits=_get_profiled_env_int("QUANT_BITS", 4, profile),
        group_size=_get_env_int("GROUP_SIZE", 128),
        desc_act=_get_env_bool("DESC_ACT", False),
        sym=_get_env_bool("SYMMETRIC", True),
        use_calibration=_get_env_bool("USE_CALIBRATION", True),
        output_name=_get_env("OUTPUT_NAME", DEFAULT_OUTPUT_NAME),
        save_tokenizer=_get_env_bool("SAVE_TOKENIZER", True),
    )

    ## Resolve benchmark section
    benchmark = BenchmarkConfig(
        enabled=_get_env_bool("BENCHMARK_ENABLED", True),
        prompt_tokens=_get_env_int("PROMPT_TOKENS", 512),
        max_new_tokens=_get_env_int("MAX_NEW_TOKENS", 128),
        num_runs=_get_env_int("NUM_RUNS", 10),
        warmup_runs=_get_env_int("WARMUP_RUNS", 2),
        deterministic=_get_env_bool("DETERMINISTIC", True),
    )

    ## Resolve optional secrets Load JSON secrets
    secrets_path = _get_env_path("APP_SECRETS_FILE", "", project_root)

    app_json = _read_json_secret(secrets_path) if secrets_path else {}

    secrets = SecretsConfig(
        huggingface_token=app_json.get("huggingface_token", ""),
        api_key=app_json.get("api_key", ""),
    )

    ## Build final config
    config = AppConfig(
        app_name=_get_env("APP_NAME", DEFAULT_APP_NAME), app_version=_get_env("APP_VERSION", DEFAULT_APP_VERSION),
        execution=execution, paths=paths, runtime=runtime, quantization=quantization,
        benchmark=benchmark, secrets=secrets,
    )

    ## Validate final configuration
    _validate_config(config)

    ## Log concise configuration summary
    logger.info(
        "Configuration loaded | app=%s | env=%s | profile=%s | gpu=%s | backend=%s | bits=%s | run_id=%s",
        config.app_name, config.runtime.environment, config.runtime.profile, config.runtime.use_gpu,
        config.quantization.backend, config.quantization.quant_bits, config.execution.run_id,
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

    ## Preserve the original public entrypoint
    return get_config()

## ============================================================
## PUBLIC SINGLETON CONFIG
## ============================================================
CONFIG: AppConfig = get_config()
config = CONFIG