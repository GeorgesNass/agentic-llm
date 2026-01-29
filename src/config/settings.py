"""
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Settings loader: .env loading, path expansion, settings builder, and JSON-serializable export."
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from src.config.schemas import (
    AppSettings,
    DataSettings,
    EvalSettings,
    TrainingSettings,
    validate_data_settings,
    validate_training_settings,
)

def _to_bool(value: Optional[str], default: bool = False) -> bool:
    """
		Convert a string env value to boolean

		Args:
			value: Raw environment variable value
			default: Default boolean if value is None or empty

		Returns:
			Parsed boolean
    """
    
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    return default

def _to_int(value: Optional[str], default: int) -> int:
    """
		Convert env value to int safely

		Args:
			value: Raw env value
			default: Default int if conversion fails

		Returns:
			Parsed integer
    """
    
    if value is None or not value.strip():
        return default

    try:
        return int(value.strip())
    except ValueError:
        return default

def _to_float(value: Optional[str], default: float) -> float:
    """
		Convert env value to float safely

		Args:
			value: Raw env value
			default: Default float if conversion fails

		Returns:
			Parsed float
    """
    
    if value is None or not value.strip():
        return default

    try:
        return float(value.strip())
    except ValueError:
        return default

def _expand_path(path_str: str) -> Path:
    """
		Expand user symbols and resolve a path

		Args:
			path_str: Path string potentially containing ~ or env vars

		Returns:
			Resolved Path
    """
    
    expanded = os.path.expandvars(os.path.expanduser(path_str))
    return Path(expanded).resolve()

def _detect_project_root() -> Path:
    """
		Detect project root based on this file location

		Returns:
			Project root directory (resolved)
    """
    
    ## src/config/settings.py -> src/config -> src -> project_root
    return Path(__file__).resolve().parents[2]

def load_settings(env_path: Optional[Path] = None) -> AppSettings:
    """
		Load environment variables and build strongly-typed settings

		Args:
			env_path: Optional .env path. If None, loads from project root ".env" if present

		Returns:
			AppSettings fully validated and ready for pipeline usage

		Raises:
			ValueError: If configuration is invalid
    """
    
    project_root = _detect_project_root()

    ## Load .env with priority: explicit env_path -> project_root/.env -> default discovery
    default_env = project_root / ".env"
    env_file = env_path or (default_env if default_env.exists() else None)

    if env_file is not None:
        load_dotenv(dotenv_path=str(env_file), override=False)
    else:
        load_dotenv(override=False)

    ## -------------------------
    ## Profile overrides
    ## -------------------------
    profile = os.getenv("PROFILE", "gpu").strip().lower()
    if profile not in {"cpu", "gpu"}:
        profile = "gpu"

    prefix = "CPU_" if profile == "cpu" else "GPU_"

    ## -------------------------
    ## Data settings
    ## -------------------------
    raw_dir = _expand_path(
        os.getenv(
            "RAW_DATA_DIR",
            os.getenv("DATA_RAW_DIR", str(project_root / "data" / "raw")),
        )
    )
    interim_dir = _expand_path(
        os.getenv(
            "INTERIM_DATA_DIR",
            os.getenv("DATA_INTERIM_DIR", str(project_root / "data" / "interim")),
        )
    )
    processed_dir = _expand_path(
        os.getenv(
            "PROCESSED_DATA_DIR",
            os.getenv("DATA_PROCESSED_DIR", str(project_root / "data" / "processed")),
        )
    )

    label_list_str = os.getenv("LABEL_LIST_FILE", "").strip()
    label_list_file = _expand_path(label_list_str) if label_list_str else None

    data_settings = DataSettings(
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        train_file=os.getenv("TRAIN_FILE", "train.jsonl"),
        val_file=os.getenv("VAL_FILE", "val.jsonl"),
        test_file=os.getenv("TEST_FILE", "test.jsonl"),
        label_list_file=label_list_file,
        split_seed=_to_int(os.getenv("SPLIT_SEED"), default=42),
        train_ratio=_to_float(os.getenv("TRAIN_RATIO"), default=0.8),
        val_ratio=_to_float(os.getenv("VAL_RATIO"), default=0.1),
        test_ratio=_to_float(os.getenv("TEST_RATIO"), default=0.1),
    )

    ## -------------------------
    ## Training settings
    ## -------------------------
    output_dir = _expand_path(
        os.getenv("OUTPUT_DIR", str(project_root / "artifacts" / "runs"))
    )

    training_settings = TrainingSettings(
        base_model_name=os.getenv(
            "BASE_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3"
        ),
        output_dir=output_dir,
        use_gpu=_to_bool(os.getenv(f"{prefix}USE_GPU"), default=True),
        seed=_to_int(os.getenv("SEED"), default=42),
        num_train_epochs=_to_int(os.getenv(f"{prefix}NUM_TRAIN_EPOCHS"), default=3),
        learning_rate=_to_float(os.getenv("LEARNING_RATE"), default=2e-4),
        batch_size=_to_int(os.getenv(f"{prefix}BATCH_SIZE"), default=2),
        grad_accum_steps=_to_int(os.getenv(f"{prefix}GRAD_ACCUM_STEPS"), default=8),
        max_seq_len=_to_int(os.getenv(f"{prefix}MAX_SEQ_LEN"), default=512),
        lora_r=_to_int(os.getenv("LORA_R"), default=16),
        lora_alpha=_to_int(os.getenv("LORA_ALPHA"), default=32),
        lora_dropout=_to_float(os.getenv("LORA_DROPOUT"), default=0.05),
        bf16=_to_bool(os.getenv(f"{prefix}BF16"), default=True),
        fp16=_to_bool(os.getenv(f"{prefix}FP16"), default=False),
        quantization_4bit=_to_bool(
            os.getenv(f"{prefix}QUANTIZATION_4BIT"),
            default=False,
        ),
        logging_steps=_to_int(os.getenv("LOGGING_STEPS"), default=25),
        save_steps=_to_int(os.getenv("SAVE_STEPS"), default=200),
    )

    ## -------------------------
    ## Evaluation settings
    ## -------------------------
    evaluation_settings = EvalSettings(
        top_k=_to_int(os.getenv("TOP_K"), default=3),
        enable_reject=_to_bool(os.getenv("ENABLE_REJECT"), default=False),
        reject_token=os.getenv("REJECT_TOKEN", "UNKNOWN"),
    )

    ## -------------------------
    ## Validate and return
    ## -------------------------
    validate_data_settings(data_settings)
    validate_training_settings(training_settings)

    return AppSettings(
        project_root=project_root,
        data=data_settings,
        training=training_settings,
        evaluation=evaluation_settings,
    )

def settings_to_dict(settings: AppSettings) -> Dict[str, Any]:
    """
		Convert settings dataclasses to a JSON-serializable dict

		Args:
			settings: AppSettings instance

		Returns:
			Dict representation suitable for logging and saving
    """
    
    ## Dataclasses are not directly JSON-serializable when containing Paths
    return {
        "project_root": str(settings.project_root),
        "data": {
            "raw_dir": str(settings.data.raw_dir),
            "interim_dir": str(settings.data.interim_dir),
            "processed_dir": str(settings.data.processed_dir),
            "train_file": settings.data.train_file,
            "val_file": settings.data.val_file,
            "test_file": settings.data.test_file,
            "label_list_file": str(settings.data.label_list_file) if settings.data.label_list_file else None,
            "split_seed": settings.data.split_seed,
            "train_ratio": settings.data.train_ratio,
            "val_ratio": settings.data.val_ratio,
            "test_ratio": settings.data.test_ratio,
        },
        "training": {
            "base_model_name": settings.training.base_model_name,
            "output_dir": str(settings.training.output_dir),
            "use_gpu": settings.training.use_gpu,
            "seed": settings.training.seed,
            "num_train_epochs": settings.training.num_train_epochs,
            "learning_rate": settings.training.learning_rate,
            "batch_size": settings.training.batch_size,
            "grad_accum_steps": settings.training.grad_accum_steps,
            "max_seq_len": settings.training.max_seq_len,
            "lora_r": settings.training.lora_r,
            "lora_alpha": settings.training.lora_alpha,
            "lora_dropout": settings.training.lora_dropout,
            "bf16": settings.training.bf16,
            "fp16": settings.training.fp16,
            "quantization_4bit": settings.training.quantization_4bit,
            "logging_steps": settings.training.logging_steps,
            "save_steps": settings.training.save_steps,
        },
        "evaluation": {
            "top_k": settings.evaluation.top_k,
            "enable_reject": settings.evaluation.enable_reject,
            "reject_token": settings.evaluation.reject_token,
        },
    }