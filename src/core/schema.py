"""
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Settings schemas and validations: dataclasses for data/training/eval and strict validation helpers."
"""

from __future__ import annotations

## STANDARD IMPORTS
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover
    BaseSettings = BaseModel  # type: ignore[misc, assignment]
    SettingsConfigDict = dict  # type: ignore[misc, assignment]

## ============================================================
## COMMON TYPES
## ============================================================
PrecisionName = Literal["fp32", "fp16", "bf16"]
OptimizerName = Literal["adamw", "paged_adamw", "lion"]
SchedulerName = Literal["linear", "cosine", "constant"]

JobStatusName = Literal["pending", "running", "success", "failed", "cancelled"]
TaskTypeName = Literal[
    "prepare_dataset",
    "train",
    "evaluate",
    "export_adapter",
    "benchmark",
]
LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

## ============================================================
## BASE PYDANTIC SCHEMAS
## ============================================================
class BaseSchema(BaseModel):
    """
        Base schema with shared validation and serialization helpers

        Returns:
            A reusable Pydantic base model
    """

    model_config = {
        "extra": "forbid",
        "populate_by_name": True,
        "str_strip_whitespace": True,
    }

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the model to a Python dictionary

            Returns:
                Serialized model as dictionary
        """

        return self.model_dump()

    def to_json(self) -> str:
        """
            Convert the model to a JSON string

            Returns:
                Serialized model as JSON
        """

        return self.model_dump_json()

    def to_record(self) -> dict[str, Any]:
        """
            Convert the model to a row-oriented dictionary

            Returns:
                Flat dictionary representation
        """

        return self.model_dump(mode="json")

    def to_pandas(self) -> Any:
        """
            Convert the model to a one-row pandas DataFrame

            Returns:
                A pandas DataFrame with one row
        """

        import pandas as pd

        return pd.DataFrame([self.to_record()])

class WarningMixin(BaseSchema):
    """
        Mixin exposing warnings in response payloads

        Args:
            warnings: Warning messages list
    """

    warnings: list[str] = Field(default_factory=list)

## ============================================================
## ENV SETTINGS
## ============================================================
class EnvSettings(BaseSettings):
    """
        Runtime settings for local-finetuning

        Args:
            app_name: Application name
            environment: Runtime environment
            default_precision: Default training precision
            default_optimizer: Default optimizer name
            default_scheduler: Default scheduler name
            default_learning_rate: Default learning rate
            default_batch_size: Default batch size
            default_top_k: Default evaluation top-k
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        env_prefix="LOCAL_FINETUNING_",
        case_sensitive=False,
    )

    app_name: str = "local-finetuning"
    environment: str = "dev"
    default_precision: PrecisionName = "bf16"
    default_optimizer: OptimizerName = "adamw"
    default_scheduler: SchedulerName = "linear"
    default_learning_rate: float = Field(default=2e-4, gt=0.0)
    default_batch_size: int = Field(default=4, ge=1)
    default_top_k: int = Field(default=5, ge=1, le=100)

## ============================================================
## EXISTING DATACLASS SETTINGS
## ============================================================
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

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

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

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

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

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

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

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

## ============================================================
## EXISTING VALIDATION HELPERS
## ============================================================
def validate_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
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

def validate_eval_settings(evaluation: EvalSettings) -> None:
    """
        Validate evaluation settings

        Args:
            evaluation: EvalSettings instance

        Raises:
            ValueError: If settings are invalid
    """

    ## Validate top-k and reject token configuration
    if evaluation.top_k <= 0:
        raise ValueError("top_k must be > 0")
    if evaluation.enable_reject and not evaluation.reject_token.strip():
        raise ValueError("reject_token must be non-empty when enable_reject is True")

def validate_app_settings(app_settings: AppSettings) -> None:
    """
        Validate full application settings

        Args:
            app_settings: AppSettings instance

        Raises:
            ValueError: If one nested configuration is invalid
    """

    ## Validate nested settings
    validate_data_settings(app_settings.data)
    validate_training_settings(app_settings.training)
    validate_eval_settings(app_settings.evaluation)

## ============================================================
## PYDANTIC CONFIG SCHEMAS
## ============================================================
class TrainingSettingsSchema(BaseSchema):
    """
        Pydantic training configuration schema

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
            bf16: Use bfloat16
            fp16: Use float16
            quantization_4bit: Enable 4-bit quantization
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
    """

    base_model_name: str = Field(..., min_length=1)
    output_dir: str = Field(..., min_length=1)
    use_gpu: bool
    seed: int
    num_train_epochs: int = Field(..., ge=1)
    learning_rate: float = Field(..., gt=0.0)
    batch_size: int = Field(..., ge=1)
    grad_accum_steps: int = Field(..., ge=1)
    max_seq_len: int = Field(..., ge=1)
    lora_r: int = Field(..., ge=1)
    lora_alpha: int = Field(..., ge=1)
    lora_dropout: float = Field(..., ge=0.0, lt=1.0)
    bf16: bool
    fp16: bool
    quantization_4bit: bool
    logging_steps: int = Field(..., ge=1)
    save_steps: int = Field(..., ge=1)

    @model_validator(mode="after")
    def validate_precision_flags(self) -> "TrainingSettingsSchema":
        """
            Validate precision flag consistency

            Returns:
                The validated training settings schema

            Raises:
                ValueError: If incompatible precision flags are enabled
        """

        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 cannot both be True")
        return self

class DataSettingsSchema(BaseSchema):
    """
        Pydantic data configuration schema

        Args:
            raw_dir: Directory containing raw source data
            interim_dir: Directory for cleaned intermediate data
            processed_dir: Directory for final train/val/test JSONL
            train_file: Processed training JSONL filename
            val_file: Processed validation JSONL filename
            test_file: Processed test JSONL filename
            label_list_file: Optional label list file
            split_seed: Seed for deterministic splitting
            train_ratio: Train split ratio
            val_ratio: Validation split ratio
            test_ratio: Test split ratio
    """

    raw_dir: str = Field(..., min_length=1)
    interim_dir: str = Field(..., min_length=1)
    processed_dir: str = Field(..., min_length=1)
    train_file: str = Field(..., min_length=1)
    val_file: str = Field(..., min_length=1)
    test_file: str = Field(..., min_length=1)
    label_list_file: str | None = None
    split_seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float

    @model_validator(mode="after")
    def validate_ratios(self) -> "DataSettingsSchema":
        """
            Validate split ratio consistency

            Returns:
                The validated data settings schema
        """

        validate_split_ratios(
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
        )
        return self

class EvalSettingsSchema(BaseSchema):
    """
        Pydantic evaluation configuration schema

        Args:
            top_k: Top-k accuracy for assisted workflows
            enable_reject: Whether model can output a reject token
            reject_token: Canonical reject token
    """

    top_k: int = Field(..., ge=1)
    enable_reject: bool
    reject_token: str = Field(..., min_length=1)

class AppSettingsSchema(BaseSchema):
    """
        Pydantic full application settings schema

        Args:
            project_root: Root folder of the project
            data: Data settings
            training: Training settings
            evaluation: Evaluation settings
    """

    project_root: str = Field(..., min_length=1)
    data: DataSettingsSchema
    training: TrainingSettingsSchema
    evaluation: EvalSettingsSchema

## ============================================================
## REQUEST AND RESPONSE SCHEMAS
## ============================================================
class TrainRequest(BaseSchema):
    """
        Fine-tuning request payload

        Args:
            config: Full fine-tuning configuration
            run_name: Optional run name
    """

    config: AppSettingsSchema
    run_name: str | None = None

class TrainResponse(WarningMixin):
    """
        Fine-tuning response payload

        Args:
            status: Final execution status
            run_dir: Run directory path
            adapter_path: Optional adapter export path
            report_dir: Optional report directory
            logs_dir: Optional logs directory
    """

    status: JobStatusName
    run_dir: str
    adapter_path: str | None = None
    report_dir: str | None = None
    logs_dir: str | None = None

class BatchTrainRequest(BaseSchema):
    """
        Batch training request

        Args:
            jobs: Training jobs list
    """

    jobs: list[TrainRequest]

    @model_validator(mode="after")
    def validate_jobs(self) -> "BatchTrainRequest":
        """
            Validate batch request content

            Returns:
                The validated batch training request

            Raises:
                ValueError: If jobs list is empty
        """

        if not self.jobs:
            raise ValueError("jobs must contain at least one item")
        return self

class BatchTrainResponse(WarningMixin):
    """
        Batch training response

        Args:
            results: Training job results
            batch_size: Number of processed jobs
    """

    results: list[TrainResponse]
    batch_size: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_batch_size(self) -> "BatchTrainResponse":
        """
            Validate batch size consistency

            Returns:
                The validated batch training response

            Raises:
                ValueError: If batch_size is inconsistent
        """

        if self.batch_size != len(self.results):
            raise ValueError("batch_size must match len(results)")
        return self

## ============================================================
## PIPELINE JOB SCHEMAS
## ============================================================
class PipelineTask(BaseSchema):
    """
        Pipeline task schema

        Args:
            task_id: Task identifier
            task_type: Task type
            status: Task status
            progress: Task progress percentage
            input_payload: Task input payload
            output_payload: Task output payload
    """

    task_id: str
    task_type: TaskTypeName
    status: JobStatusName = "pending"
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)

class PipelineJob(BaseSchema):
    """
        Pipeline job schema

        Args:
            job_id: Job identifier
            status: Job status
            tasks: Job tasks
            progress: Job progress percentage
            metadata: Job metadata
    """

    job_id: str
    status: JobStatusName = "pending"
    tasks: list[PipelineTask] = Field(default_factory=list)
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_progress(self) -> "PipelineJob":
        """
            Validate job progress consistency

            Returns:
                The validated pipeline job

            Raises:
                ValueError: If progress is inconsistent with tasks
        """

        if self.tasks and self.progress < min(task.progress for task in self.tasks):
            raise ValueError("job progress cannot be below the minimum task progress")
        return self

## ============================================================
## MONITORING AND COMMON RESPONSES
## ============================================================
class MetricPoint(BaseSchema):
    """
        Monitoring metric schema

        Args:
            name: Metric name
            value: Metric value
            unit: Optional metric unit
            tags: Optional metric tags
    """

    name: str
    value: float
    unit: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)

class MonitoringResponse(WarningMixin):
    """
        Monitoring response schema

        Args:
            metrics: Metric points list
            summary: Aggregated summary
    """

    metrics: list[MetricPoint] = Field(default_factory=list)
    summary: dict[str, float] = Field(default_factory=dict)

class HealthResponse(BaseSchema):
    """
        Healthcheck response schema

        Args:
            status: Service status
            service: Service name
            version: Application version
            timestamp: Response timestamp
    """

    status: str = "ok"
    service: str = "local-finetuning"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ErrorResponse(BaseSchema):
    """
        Standard API error schema

        Args:
            error: Normalized error code
            message: Human-readable message
            origin: Component where the error happened
            details: Diagnostic details
            request_id: Optional request correlation id
    """

    error: str
    message: str
    origin: str = "unknown"
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str = "n/a"

class StatusResponse(BaseSchema):
    """
        Generic status response schema

        Args:
            status: Current status
            message: Optional message
            progress: Optional progress value
            metadata: Optional metadata payload
    """

    status: str
    message: str = ""
    progress: float | None = Field(default=None, ge=0.0, le=100.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

class StructuredLogEvent(BaseSchema):
    """
        Structured log schema

        Args:
            level: Log level
            event: Event name
            message: Human-readable message
            logger_name: Logger name
            context: Additional context
            timestamp: Event timestamp
    """

    level: LogLevelName
    event: str
    message: str
    logger_name: str = "local-finetuning"
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)