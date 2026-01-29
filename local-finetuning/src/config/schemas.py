"""
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Settings schemas and validations: dataclasses for data/training/eval and strict validation helpers."
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class TrainingSettings:
    """
		Training configuration for LoRA SFT

		Args:
			base_model_name: HF model id or local path to base model
			output_dir: Directory where run artifacts and exports are written
			use_gpu: Whether to use GPU if available
			seed: Global RNG seed for reproducibility
			num_train_epochs: Number of training epochs
			learning_rate: Optimizer learning rate
			batch_size: Per-device batch size
			grad_accum_steps: Gradient accumulation steps
			max_seq_len: Tokenization max sequence length
			lora_r: LoRA rank
			lora_alpha: LoRA alpha scaling
			lora_dropout: LoRA dropout
			bf16: Use bfloat16 (GPU only, if supported)
			fp16: Use float16 (GPU only)
			quantization_4bit: Enable 4-bit quantization (QLoRA mode)
			logging_steps: Log every N steps
			save_steps: Save checkpoint every N steps
    """

    base_model_name: str
    output_dir: Path
    use_gpu: bool
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
class DataSettings:
    """
		Data configuration for symptom normalization dataset

		Args:
			raw_dir: Directory containing raw source data
			interim_dir: Directory for cleaned / deduplicated intermediate data
			processed_dir: Directory for final train/val/test JSONL
			train_file: Processed training JSONL filename
			val_file: Processed validation JSONL filename
			test_file: Processed test JSONL filename
			label_list_file: Optional label list file (one CISP label per line)
			split_seed: Seed for deterministic splitting
			train_ratio: Train split ratio
			val_ratio: Validation split ratio
			test_ratio: Test split ratio
    """

    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    train_file: str
    val_file: str
    test_file: str
    label_list_file: Optional[Path]
    split_seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float

@dataclass(frozen=True)
class EvalSettings:
    """
		Evaluation configuration

		Args:
			top_k: Top-k accuracy for assisted workflows
			enable_reject: Whether model can output a reject token
			reject_token: The canonical reject token (e.g., "UNKNOWN")
    """

    top_k: int
    enable_reject: bool
    reject_token: str

@dataclass(frozen=True)
class AppSettings:
    """
		Global application settings

		Args:
			project_root: Root folder of the project
			data: Data settings
			training: Training settings
			evaluation: Evaluation settings
    """

    project_root: Path
    data: DataSettings
    training: TrainingSettings
    evaluation: EvalSettings

def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """
		Validate split ratios are sane and sum to 1

		Args:
			train_ratio: Train ratio
			val_ratio: Validation ratio
			test_ratio: Test ratio

		Raises:
			ValueError: If ratios are invalid
    """
    
    ## Basic range checks
    for name, ratio in [
        ("train_ratio", train_ratio),
        ("val_ratio", val_ratio),
        ("test_ratio", test_ratio),
    ]:
        if ratio <= 0.0 or ratio >= 1.0:
            raise ValueError(f"{name} must be within (0, 1), got {ratio}")

    ## Sum check with a small tolerance
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

def validate_training_settings(training: TrainingSettings) -> None:
    """
		Validate training settings for safety and reproducibility

		Args:
			training: TrainingSettings instance

		Raises:
			ValueError: If settings are invalid
    """
    
    ## Check numerical constraints
    if training.num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be > 0")
    if training.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if training.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if training.grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be > 0")
    if training.max_seq_len <= 0:
        raise ValueError("max_seq_len must be > 0")

    ## LoRA constraints
    if training.lora_r <= 0:
        raise ValueError("lora_r must be > 0")
    if training.lora_alpha <= 0:
        raise ValueError("lora_alpha must be > 0")
    if training.lora_dropout < 0.0 or training.lora_dropout >= 1.0:
        raise ValueError("lora_dropout must be within [0, 1)")

    ## Logging constraints
    if training.logging_steps <= 0:
        raise ValueError("logging_steps must be > 0")
    if training.save_steps <= 0:
        raise ValueError("save_steps must be > 0")

    ## Mixed precision options should not be both enabled
    if training.bf16 and training.fp16:
        raise ValueError("bf16 and fp16 cannot both be True")

def validate_data_settings(data: DataSettings) -> None:
    """
		Validate data settings

		Args:
			data: DataSettings instance

		Raises:
			ValueError: If settings are invalid
    """
    
    validate_split_ratios(data.train_ratio, data.val_ratio, data.test_ratio)

    ## Ensure filenames are non-empty
    for name, filename in [
        ("train_file", data.train_file),
        ("val_file", data.val_file),
        ("test_file", data.test_file),
    ]:
        if not filename or not filename.strip():
            raise ValueError(f"{name} must be a non-empty filename")

    ## label list file is optional; validate path object if present
    if data.label_list_file is not None and not isinstance(data.label_list_file, Path):
        raise ValueError("label_list_file must be a Path or None")