'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unified configuration loader for local-finetuning: dotenv, env parsing, paths, profiles, training, evaluation, secrets and runtime metadata."
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
JSONL_EXTENSION = ".jsonl"

## ============================================================
## STABLE DOMAIN CONSTANTS
## ============================================================
DEFAULT_APP_NAME = "local-finetuning"
DEFAULT_APP_VERSION = "1.0.0"
DEFAULT_ENVIRONMENT = "dev"
DEFAULT_PROFILE = "gpu"

DEFAULT_DATA_DIR = "data"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_SECRETS_DIR = "secrets"

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_INTERIM_DIR = "data/interim"
DEFAULT_PROCESSED_DIR = "data/processed"

DEFAULT_RUNS_DIR = "artifacts/runs"
DEFAULT_EXPORTS_DIR = "artifacts/exports"
DEFAULT_REPORTS_DIR = "artifacts/reports"

DEFAULT_BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_TRAIN_FILE = "train.jsonl"
DEFAULT_VAL_FILE = "val.jsonl"
DEFAULT_TEST_FILE = "test.jsonl"
DEFAULT_REJECT_TOKEN = "UNKNOWN"

SUPPORTED_INPUT_EXTENSIONS = (".json", ".jsonl", ".csv", ".txt")
SUPPORTED_EXPORT_EXTENSIONS = (".json", ".jsonl", ".bin", ".safetensors")

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
            raw_dir: Raw source data directory
            interim_dir: Interim data directory
            processed_dir: Processed dataset directory
            artifacts_dir: Artifacts root directory
            runs_dir: Training runs directory
            exports_dir: Exported adapters directory
            reports_dir: Reports directory
            logs_dir: Logs directory
            secrets_dir: Secrets directory
            label_list_file: Optional label list file
    """

    project_root: Path
    src_dir: Path
    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    runs_dir: Path
    exports_dir: Path
    reports_dir: Path
    logs_dir: Path
    secrets_dir: Path
    label_list_file: Optional[Path]

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
class DataConfig:
    """
        Dataset configuration

        Args:
            train_file: Training filename
            val_file: Validation filename
            test_file: Test filename
            split_seed: Dataset split seed
            train_ratio: Training split ratio
            val_ratio: Validation split ratio
            test_ratio: Test split ratio
    """

    train_file: str
    val_file: str
    test_file: str
    split_seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float

@dataclass(frozen=True)
class TrainingConfig:
    """
        Training configuration for LoRA / QLoRA

        Args:
            base_model_name: Base model HF id or local path
            output_dir: Directory where run artifacts are written
            seed: Global RNG seed
            num_train_epochs: Number of training epochs
            learning_rate: Optimizer learning rate
            batch_size: Per-device batch size
            grad_accum_steps: Gradient accumulation steps
            max_seq_len: Tokenization max sequence length
            lora_r: LoRA rank
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout
            bf16: Whether bf16 is enabled
            fp16: Whether fp16 is enabled
            quantization_4bit: Whether 4-bit quantization is enabled
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
    """

    base_model_name: str
    output_dir: Path
    seed: int
    num_train_epochs: int
    learning_rate: float
    batch_size: int
    grad_accum_steps: int
    max_seq_len: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    bf16: bool
    fp16: bool
    quantization_4bit: bool
    logging_steps: int
    save_steps: int

@dataclass(frozen=True)
class EvaluationConfig:
    """
        Evaluation configuration

        Args:
            top_k: Top-k accuracy target
            enable_reject: Whether reject token is enabled
            reject_token: Canonical reject token
    """

    top_k: int
    enable_reject: bool
    reject_token: str

@dataclass(frozen=True)
class SecretsConfig:
    """
        Secret values resolved from env or files

        Args:
            huggingface_token: Optional Hugging Face token
            wandb_api_key: Optional Weights & Biases API key
            api_key: Optional generic API key
    """

    huggingface_token: str
    wandb_api_key: str
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
            data: Data configuration
            training: Training configuration
            evaluation: Evaluation configuration
            secrets: Secret values
    """

    app_name: str
    app_version: str
    execution: ExecutionMetadata
    paths: PathsConfig
    runtime: RuntimeConfig
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
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

def _validate_probability_open(value: float, field_name: str) -> None:
    """
        Validate that a float is within (0, 1)

        Args:
            value: Float value
            field_name: Human-readable field name

        Returns:
            None
    """

    ## Reject invalid open probabilities
    if value <= 0.0 or value >= 1.0:
        raise ConfigurationError(f"{field_name} must be within (0, 1). Got: {value}")

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
            1) Validate dataset split ratios
            2) Validate training hyperparameters
            3) Validate evaluation parameters
            4) Validate mixed precision consistency

        Args:
            config: Structured configuration object

        Returns:
            None
    """

    ## Validate data split ratios
    _validate_probability_open(config.data.train_ratio, "TRAIN_RATIO")
    _validate_probability_open(config.data.val_ratio, "VAL_RATIO")
    _validate_probability_open(config.data.test_ratio, "TEST_RATIO")
    total = config.data.train_ratio + config.data.val_ratio + config.data.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ConfigurationError(f"TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0. Got: {total}")

    ## Validate training parameters
    _validate_positive_int(config.training.num_train_epochs, "NUM_TRAIN_EPOCHS")
    _validate_positive_int(config.training.batch_size, "BATCH_SIZE")
    _validate_positive_int(config.training.grad_accum_steps, "GRAD_ACCUM_STEPS")
    _validate_positive_int(config.training.max_seq_len, "MAX_SEQ_LEN")
    _validate_positive_int(config.training.lora_r, "LORA_R")
    _validate_positive_int(config.training.lora_alpha, "LORA_ALPHA")
    _validate_positive_int(config.training.logging_steps, "LOGGING_STEPS")
    _validate_positive_int(config.training.save_steps, "SAVE_STEPS")
    if config.training.learning_rate <= 0:
        raise ConfigurationError(f"LEARNING_RATE must be > 0. Got: {config.training.learning_rate}")
    if config.training.lora_dropout < 0.0 or config.training.lora_dropout >= 1.0:
        raise ConfigurationError(f"LORA_DROPOUT must be within [0, 1). Got: {config.training.lora_dropout}")

    ## Validate precision consistency
    if config.training.bf16 and config.training.fp16:
        raise ConfigurationError("BF16 and FP16 cannot both be True")

    ## Validate evaluation parameters
    _validate_positive_int(config.evaluation.top_k, "TOP_K")

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
            3) Build runtime, data, training and evaluation sections
            4) Resolve optional secrets
            5) Validate and cache final configuration

        Args:
            None

        Returns:
            AppConfig instance
    """

    ## Load optional local .env file first
    _load_dotenv_if_present()

    ## Resolve root and runtime profile
    project_root = _resolve_project_root()
    environment = _get_env("ENVIRONMENT", DEFAULT_ENVIRONMENT).lower()
    profile_raw = _get_env("PROFILE", DEFAULT_PROFILE).lower()
    profile: ProfileName = "cpu" if profile_raw == "cpu" else "gpu"

    ## Validate placeholder values where relevant
    _validate_required_placeholders(["ENVIRONMENT", "PROFILE", "HUGGINGFACE_TOKEN", "WANDB_API_KEY", "API_KEY"])

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
    data_dir = _get_env_path("DATA_DIR", DEFAULT_DATA_DIR, project_root)
    artifacts_dir = _get_env_path("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR, project_root)
    logs_dir = _get_env_path("LOGS_DIR", DEFAULT_LOGS_DIR, project_root)
    secrets_dir = _get_env_path("SECRETS_DIR", DEFAULT_SECRETS_DIR, project_root)
    label_list_str = _get_env("LABEL_LIST_FILE", "")

    paths = PathsConfig(
        project_root=project_root, src_dir=(project_root / "src").resolve(), data_dir=data_dir,
        raw_dir=_get_env_path("RAW_DATA_DIR", _get_env("DATA_RAW_DIR", DEFAULT_RAW_DIR), project_root),
        interim_dir=_get_env_path("INTERIM_DATA_DIR", _get_env("DATA_INTERIM_DIR", DEFAULT_INTERIM_DIR), project_root),
        processed_dir=_get_env_path("PROCESSED_DATA_DIR", _get_env("DATA_PROCESSED_DIR", DEFAULT_PROCESSED_DIR), project_root),
        artifacts_dir=artifacts_dir, runs_dir=_get_env_path("RUNS_DIR", DEFAULT_RUNS_DIR, project_root),
        exports_dir=_get_env_path("EXPORTS_DIR", DEFAULT_EXPORTS_DIR, project_root),
        reports_dir=_get_env_path("REPORTS_DIR", DEFAULT_REPORTS_DIR, project_root),
        logs_dir=logs_dir, secrets_dir=secrets_dir,
        label_list_file=_resolve_path(label_list_str, project_root) if label_list_str else None,
    )

    ## Ensure runtime directories exist
    _ensure_directories_exist([
        paths.data_dir, paths.raw_dir, paths.interim_dir, paths.processed_dir,
        paths.artifacts_dir, paths.runs_dir, paths.exports_dir, paths.reports_dir,
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

    ## Build data section
    data = DataConfig(
        train_file=_get_env("TRAIN_FILE", DEFAULT_TRAIN_FILE), val_file=_get_env("VAL_FILE", DEFAULT_VAL_FILE),
        test_file=_get_env("TEST_FILE", DEFAULT_TEST_FILE), split_seed=_get_env_int("SPLIT_SEED", 42),
        train_ratio=_get_env_float("TRAIN_RATIO", 0.8), val_ratio=_get_env_float("VAL_RATIO", 0.1),
        test_ratio=_get_env_float("TEST_RATIO", 0.1),
    )

    ## Build training section
    output_dir = _get_env_path("OUTPUT_DIR", str(paths.runs_dir), project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    training = TrainingConfig(
        base_model_name=_get_env("BASE_MODEL_NAME", DEFAULT_BASE_MODEL_NAME), output_dir=output_dir,
        seed=_get_env_int("SEED", 42), num_train_epochs=_get_profiled_env_int("NUM_TRAIN_EPOCHS", 3, profile),
        learning_rate=_get_env_float("LEARNING_RATE", 2e-4), batch_size=_get_profiled_env_int("BATCH_SIZE", 2, profile),
        grad_accum_steps=_get_profiled_env_int("GRAD_ACCUM_STEPS", 8, profile),
        max_seq_len=_get_profiled_env_int("MAX_SEQ_LEN", 512, profile), lora_r=_get_env_int("LORA_R", 16),
        lora_alpha=_get_env_int("LORA_ALPHA", 32), lora_dropout=_get_env_float("LORA_DROPOUT", 0.05),
        bf16=_get_profiled_env_bool("BF16", True, profile), fp16=_get_profiled_env_bool("FP16", False, profile),
        quantization_4bit=_get_profiled_env_bool("QUANTIZATION_4BIT", False, profile),
        logging_steps=_get_env_int("LOGGING_STEPS", 25), save_steps=_get_env_int("SAVE_STEPS", 200),
    )

    ## Build evaluation section
    evaluation = EvaluationConfig(
        top_k=_get_env_int("TOP_K", 3), enable_reject=_get_env_bool("ENABLE_REJECT", False),
        reject_token=_get_env("REJECT_TOKEN", DEFAULT_REJECT_TOKEN),
    )

    ## Resolve optional secrets
    secrets = SecretsConfig(
        huggingface_token=_read_secret_value("HUGGINGFACE_TOKEN", "HUGGINGFACE_TOKEN_FILE", project_root=project_root),
        wandb_api_key=_read_secret_value("WANDB_API_KEY", "WANDB_API_KEY_FILE", project_root=project_root),
        api_key=_read_secret_value("API_KEY", "API_KEY_FILE", project_root=project_root),
    )

    ## Build final config
    config = AppConfig(
        app_name=_get_env("APP_NAME", DEFAULT_APP_NAME), app_version=_get_env("APP_VERSION", DEFAULT_APP_VERSION),
        execution=execution, paths=paths, runtime=runtime, data=data, training=training,
        evaluation=evaluation, secrets=secrets,
    )

    ## Validate final configuration
    _validate_config(config)

    ## Log concise configuration summary
    logger.info(
        "Configuration loaded | app=%s | env=%s | profile=%s | gpu=%s | base_model=%s | epochs=%s | run_id=%s",
        config.app_name, config.runtime.environment, config.runtime.profile, config.runtime.use_gpu,
        config.training.base_model_name, config.training.num_train_epochs, config.execution.run_id,
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
