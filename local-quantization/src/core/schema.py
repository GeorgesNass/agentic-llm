'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Dataclasses and validation schemas for local-quantization configuration, requests, responses, and pipeline contracts."
'''

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover
    BaseSettings = BaseModel  # type: ignore[misc, assignment]
    SettingsConfigDict = dict  # type: ignore[misc, assignment]

## ============================================================
## COMMON TYPES
## ============================================================
BackendName = Literal["gguf", "awq", "gptq", "bnb_nf4", "onnx"]
PipelineMode = Literal["quantize", "export", "benchmark", "full"]
JobStatusName = Literal["pending", "running", "success", "failed", "cancelled"]
PrecisionName = Literal["fp32", "fp16", "bf16", "int8", "int4"]

## ============================================================
## BASE SCHEMAS
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

        ## Import pandas lazily to avoid a hard dependency at import time
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
## SETTINGS
## ============================================================
class AppSettings(BaseSettings):
    """
        Settings model for local-quantization

        Args:
            app_name: Application name
            environment: Runtime environment
            default_backend: Default quantization backend
            default_bits: Default quantization bit-width
            default_max_tokens: Default benchmark max tokens
            default_benchmark_runs: Default benchmark runs
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        env_prefix="LOCAL_QUANT_",
        case_sensitive=False,
    )

    app_name: str = "local-quantization"
    environment: str = "dev"
    default_backend: BackendName = "gguf"
    default_bits: int = Field(default=4, ge=2, le=16)
    default_max_tokens: int = Field(default=256, ge=1, le=32768)
    default_benchmark_runs: int = Field(default=5, ge=1, le=1000)

## ============================================================
## DATACLASS CONFIGS
## ============================================================
@dataclass(frozen=True)
class ModelConfig:
    """
        Configuration describing the base model and optional adapters

        Args:
            model_name_or_path: Hugging Face repo id or local model path
            adapter_path: Optional LoRA adapter path
            revision: Optional model revision or branch
    """

    model_name_or_path: str
    adapter_path: Path | None = None
    revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

@dataclass(frozen=True)
class QuantizationConfig:
    """
        Configuration describing the quantization strategy

        Args:
            backend: Quantization backend identifier
            bits: Target bit-width
            group_size: Optional group size for weight-only methods
            calibration_dataset: Optional path to calibration dataset
    """

    backend: BackendName
    bits: int
    group_size: int | None = None
    calibration_dataset: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

@dataclass(frozen=True)
class ExportConfig:
    """
        Configuration describing export options for quantized artifacts

        Args:
            output_dir: Directory where exported artifacts are written
            overwrite: Whether to overwrite existing artifacts
    """

    output_dir: Path
    overwrite: bool = False

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

@dataclass(frozen=True)
class BenchmarkConfig:
    """
        Configuration describing benchmarking options

        Args:
            prompts_path: Path to prompt file or dataset
            max_tokens: Maximum number of tokens to generate
            runs: Number of benchmark runs
    """

    prompts_path: Path
    max_tokens: int = 256
    runs: int = 5

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

@dataclass(frozen=True)
class PipelineConfig:
    """
        Top-level configuration for the local-quantization pipeline

        Args:
            mode: Pipeline execution mode
            model: Model configuration
            quantization: Quantization configuration
            export: Optional export configuration
            benchmark: Optional benchmark configuration
    """

    mode: PipelineMode
    model: ModelConfig
    quantization: QuantizationConfig
    export: ExportConfig | None = None
    benchmark: BenchmarkConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

## ============================================================
## PYDANTIC CONFIG SCHEMAS
## ============================================================
class ModelConfigSchema(BaseSchema):
    """
        Pydantic model configuration schema

        Args:
            model_name_or_path: Hugging Face repo id or local model path
            adapter_path: Optional LoRA adapter path
            revision: Optional model revision or branch
            source_precision: Original model precision
    """

    model_name_or_path: str = Field(..., min_length=1)
    adapter_path: str | None = None
    revision: str | None = None
    source_precision: PrecisionName = "fp16"

class QuantizationConfigSchema(BaseSchema):
    """
        Pydantic quantization configuration schema

        Args:
            backend: Quantization backend identifier
            bits: Target bit-width
            group_size: Optional group size
            calibration_dataset: Optional calibration dataset path
            dataset_split: Optional calibration split name
    """

    backend: BackendName
    bits: int = Field(..., ge=2, le=16)
    group_size: int | None = Field(default=None, ge=1)
    calibration_dataset: str | None = None
    dataset_split: str | None = None

    @model_validator(mode="after")
    def validate_backend_rules(self) -> "QuantizationConfigSchema":
        """
            Validate backend-specific quantization rules

            Returns:
                The validated quantization configuration

            Raises:
                ValueError: If one backend rule is violated
        """

        ## Validate common backend-specific constraints
        if self.backend in {"awq", "gptq", "bnb_nf4"} and self.bits != 4:
            raise ValueError(f"{self.backend} typically requires bits=4")

        if self.backend == "onnx" and self.bits not in {8, 16}:
            raise ValueError("onnx supports bits in {8, 16}")

        if self.backend == "gguf" and self.bits not in {2, 3, 4, 5, 6, 8}:
            raise ValueError("gguf supports bits in {2, 3, 4, 5, 6, 8}")

        return self

class ExportConfigSchema(BaseSchema):
    """
        Pydantic export configuration schema

        Args:
            output_dir: Directory where exported artifacts are written
            overwrite: Whether to overwrite existing artifacts
            push_to_hub: Whether artifacts should be pushed to a hub
    """

    output_dir: str = Field(..., min_length=1)
    overwrite: bool = False
    push_to_hub: bool = False

class BenchmarkConfigSchema(BaseSchema):
    """
        Pydantic benchmark configuration schema

        Args:
            prompts_path: Path to prompt file or dataset
            max_tokens: Maximum number of tokens to generate
            runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
    """

    prompts_path: str = Field(..., min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=32768)
    runs: int = Field(default=5, ge=1, le=1000)
    warmup_runs: int = Field(default=1, ge=0, le=100)

    @model_validator(mode="after")
    def validate_benchmark_counts(self) -> "BenchmarkConfigSchema":
        """
            Validate benchmark runs consistency

            Returns:
                The validated benchmark configuration

            Raises:
                ValueError: If warmup runs exceed benchmark runs
        """

        ## Keep benchmark execution counts coherent
        if self.warmup_runs > self.runs:
            raise ValueError("warmup_runs cannot be greater than runs")
        return self

class PipelineConfigSchema(BaseSchema):
    """
        Pydantic top-level pipeline configuration schema

        Args:
            mode: Pipeline execution mode
            model: Model configuration
            quantization: Quantization configuration
            export: Optional export configuration
            benchmark: Optional benchmark configuration
    """

    mode: PipelineMode
    model: ModelConfigSchema
    quantization: QuantizationConfigSchema
    export: ExportConfigSchema | None = None
    benchmark: BenchmarkConfigSchema | None = None

    @model_validator(mode="after")
    def validate_mode_dependencies(self) -> "PipelineConfigSchema":
        """
            Validate mode-specific required sections

            Returns:
                The validated pipeline configuration

            Raises:
                ValueError: If one required section is missing
        """

        ## Enforce config sections required by each execution mode
        if self.mode in {"export", "full"} and self.export is None:
            raise ValueError("export config is required for mode export or full")

        if self.mode in {"benchmark", "full"} and self.benchmark is None:
            raise ValueError("benchmark config is required for mode benchmark or full")

        return self

## ============================================================
## COMMON OPERATIONAL SCHEMAS
## ============================================================
class HealthResponse(BaseSchema):
    """
        Healthcheck response model

        Args:
            status: Service status
            service: Service name
            version: Application version
    """

    status: str = Field(default="ok", min_length=1)
    service: str = Field(default="local-quantization", min_length=1)
    version: str = Field(default="1.0.0", min_length=1)

class ErrorResponse(BaseSchema):
    """
        Standard API error response

        Args:
            error: Normalized error code
            message: Human-readable message
            origin: Component where the error happened
            details: Diagnostic details
            request_id: Optional request correlation id
    """

    error: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    origin: str = Field(default="unknown", min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str = Field(default="n/a", min_length=1)

class StatusResponse(BaseSchema):
    """
        Generic status response schema

        Args:
            status: Current status
            message: Optional message
            progress: Optional progress between 0 and 100
            metadata: Optional metadata
    """

    status: str = Field(..., min_length=1)
    message: str = Field(default="")
    progress: float | None = Field(default=None, ge=0.0, le=100.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

class MetricPoint(BaseSchema):
    """
        Monitoring metric point schema

        Args:
            name: Metric name
            value: Metric value
            unit: Optional metric unit
            tags: Optional metric tags
    """

    name: str = Field(..., min_length=1)
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

## ============================================================
## QUANTIZATION REQUEST AND RESPONSE SCHEMAS
## ============================================================
class QuantizeRequest(BaseSchema):
    """
        Request payload for quantization

        Args:
            config: Full quantization pipeline configuration
            run_name: Optional run name
    """

    config: PipelineConfigSchema
    run_name: str | None = None

class QuantizationArtifact(BaseSchema):
    """
        Quantization artifact schema

        Args:
            path: Artifact path
            artifact_type: Artifact type
            size_mb: Artifact size in MB
    """

    path: str = Field(..., min_length=1)
    artifact_type: str = Field(..., min_length=1)
    size_mb: float | None = Field(default=None, ge=0.0)

class BenchmarkResult(BaseSchema):
    """
        Benchmark result schema

        Args:
            latency_ms: Mean latency in milliseconds
            tokens_per_second: Mean throughput
            peak_memory_mb: Peak memory usage in MB
            runs: Number of measured runs
    """

    latency_ms: float = Field(..., ge=0.0)
    tokens_per_second: float = Field(..., ge=0.0)
    peak_memory_mb: float | None = Field(default=None, ge=0.0)
    runs: int = Field(..., ge=1)

class QuantizeResponse(WarningMixin):
    """
        Response payload after quantization or full pipeline run

        Args:
            status: Final execution status
            backend: Quantization backend used
            bits: Final bit-width
            artifacts: Exported artifacts
            benchmark: Optional benchmark result
    """

    status: JobStatusName
    backend: BackendName
    bits: int = Field(..., ge=2, le=16)
    artifacts: list[QuantizationArtifact] = Field(default_factory=list)
    benchmark: BenchmarkResult | None = None

## ============================================================
## PIPELINE JOB SCHEMAS
## ============================================================
class PipelineTask(BaseSchema):
    """
        Pipeline task schema

        Args:
            task_id: Task identifier
            task_name: Task name
            status: Task status
            progress: Task progress percentage
            input_payload: Task input payload
            output_payload: Task output payload
    """

    task_id: str = Field(..., min_length=1)
    task_name: str = Field(..., min_length=1)
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

    job_id: str = Field(..., min_length=1)
    status: JobStatusName = "pending"
    tasks: list[PipelineTask] = Field(default_factory=list)
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_job_progress(self) -> "PipelineJob":
        """
            Validate progress consistency between the job and its tasks

            Returns:
                The validated pipeline job

            Raises:
                ValueError: If job progress is below the minimum task progress
        """

        ## Keep parent progress coherent with child task progress
        if self.tasks and self.progress < min(task.progress for task in self.tasks):
            raise ValueError("job progress cannot be below the minimum task progress")
        return self