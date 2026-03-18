'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Project settings and Pydantic models for rag-drive-gcp (Drive → OCR → GCS → RAG)."
'''

from __future__ import annotations

## STANDARD IMPORTS
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover
    BaseSettings = BaseModel  # type: ignore[misc, assignment]
    SettingsConfigDict = dict  # type: ignore[misc, assignment]

## PROJECT IMPORTS
from src.core.errors import log_and_raise_missing_env
from src.utils.utils import (
    ensure_directories,
    get_project_root,
    load_dotenv_if_present,
    to_bool,
)

## ============================================================
## COMMON TYPES
## ============================================================
OcrModeName = Literal["local_docker", "remote_service"]
JobStatusName = Literal["pending", "running", "success", "failed", "cancelled"]
TaskTypeName = Literal[
    "list_drive_files",
    "download_files",
    "run_ocr",
    "chunk_documents",
    "embed_documents",
    "upload_text",
    "upload_embeddings",
    "answer_question",
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
## DATACLASS CONFIG SCHEMAS
## ============================================================
@dataclass(frozen=True)
class DriveConfig:
    """
        Drive ingestion configuration

        Args:
            drive_folder_id: Default Google Drive folder id
    """

    drive_folder_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

@dataclass(frozen=True)
class OcrConfig:
    """
        OCR configuration

        Args:
            ocr_mode: OCR execution mode
            ocr_docker_image: Docker image when using local OCR
            ocr_service_url: Base URL when using remote OCR service
    """

    ocr_mode: str = "local_docker"
    ocr_docker_image: str | None = None
    ocr_service_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

@dataclass(frozen=True)
class GcpConfig:
    """
        GCP and Vertex configuration

        Args:
            gcp_project_id: GCP project id
            gcp_region: Vertex region
            vertex_llm_model: Vertex LLM model name
            vertex_embed_model: Vertex embedding model name
    """

    gcp_project_id: str | None = None
    gcp_region: str | None = None
    vertex_llm_model: str | None = None
    vertex_embed_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

@dataclass(frozen=True)
class StorageConfig:
    """
        GCS and local storage configuration

        Args:
            gcs_bucket_text: GCS bucket for text artifacts
            gcs_prefix_text: GCS prefix for text artifacts
            gcs_bucket_embeddings: GCS bucket for embedding artifacts
            gcs_prefix_embeddings: GCS prefix for embedding artifacts
            keep_local: Whether to keep local traces
            data_dir: Local data directory
            logs_dir: Local logs directory
            raw_dir: Local raw directory
            tmp_dir: Local temporary directory
            traces_dir: Local traces directory
    """

    gcs_bucket_text: str | None = None
    gcs_prefix_text: str = "texts/"
    gcs_bucket_embeddings: str | None = None
    gcs_prefix_embeddings: str = "embeddings/"
    keep_local: bool = False
    data_dir: Path = Path("data")
    logs_dir: Path = Path("logs")
    raw_dir: Path = Path("data/raw")
    tmp_dir: Path = Path("data/tmp")
    traces_dir: Path = Path("data/traces")

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

@dataclass(frozen=True)
class RetrievalConfig:
    """
        Retrieval configuration

        Args:
            chunk_size: Chunk size for splitting
            chunk_overlap: Chunk overlap for splitting
            top_k: Number of chunks retrieved
    """

    chunk_size: int = 1024
    chunk_overlap: int = 128
    top_k: int = 8

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

@dataclass(frozen=True)
class AppRuntimeConfig:
    """
        Global runtime configuration

        Args:
            environment: Execution environment
            app_version: Application version
            log_level: Application log level
            drive: Drive configuration
            ocr: OCR configuration
            gcp: GCP configuration
            storage: Storage configuration
            retrieval: Retrieval configuration
    """

    environment: str
    app_version: str
    log_level: str
    drive: DriveConfig
    ocr: OcrConfig
    gcp: GcpConfig
    storage: StorageConfig
    retrieval: RetrievalConfig

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

## ============================================================
## SETTINGS MODEL
## ============================================================
class Settings(BaseModel):
    """
        Central application settings loaded from environment variables

        Notes:
            - This project expects a `.env` file at the project root
            - Environment variables are loaded using a lightweight parser
    """

    environment: str = Field(default="dev", description="Execution environment")
    app_version: str = Field(default="1.0.0", description="Application version")
    log_level: str = Field(default="INFO", description="Application log level")

    drive_folder_id: Optional[str] = Field(
        default=None, description="Default Google Drive folder ID"
    )

    ocr_mode: str = Field(default="local_docker", description="OCR mode")
    ocr_docker_image: Optional[str] = Field(
        default=None, description="OCR docker image"
    )
    ocr_service_url: Optional[str] = Field(
        default=None, description="OCR service base URL"
    )

    gcp_project_id: Optional[str] = Field(default=None, description="GCP project ID")
    gcp_region: Optional[str] = Field(default=None, description="GCP region")
    vertex_llm_model: Optional[str] = Field(
        default=None, description="Vertex LLM model"
    )
    vertex_embed_model: Optional[str] = Field(
        default=None, description="Vertex embedding model"
    )

    gcs_bucket_text: Optional[str] = Field(
        default=None, description="GCS bucket for text artifacts"
    )
    gcs_prefix_text: str = Field(
        default="texts/", description="GCS prefix for text artifacts"
    )
    gcs_bucket_embeddings: Optional[str] = Field(
        default=None, description="GCS bucket for embedding artifacts"
    )
    gcs_prefix_embeddings: str = Field(
        default="embeddings/", description="GCS prefix for embedding artifacts"
    )

    chunk_size: int = Field(default=1024, description="Chunk size")
    chunk_overlap: int = Field(default=128, description="Chunk overlap")
    top_k: int = Field(default=8, description="Top-k retrieval count")

    keep_local: bool = Field(
        default=False, description="Keep local traces after execution"
    )

    data_dir: Path = Field(default_factory=lambda: get_project_root() / "data")
    logs_dir: Path = Field(default_factory=lambda: get_project_root() / "logs")
    raw_dir: Path = Field(default_factory=lambda: get_project_root() / "data" / "raw")
    tmp_dir: Path = Field(default_factory=lambda: get_project_root() / "data" / "tmp")
    traces_dir: Path = Field(
        default_factory=lambda: get_project_root() / "data" / "traces"
    )

    @model_validator(mode="after")
    def validate_settings_consistency(self) -> "Settings":
        """
            Validate cross-field consistency for settings

            Returns:
                The validated settings object

            Raises:
                ValueError: If one configuration is inconsistent
        """

        ## Validate OCR mode dependencies
        if self.ocr_mode == "local_docker" and not self.ocr_docker_image:
            raise ValueError(
                "ocr_docker_image is required when ocr_mode is local_docker"
            )

        if self.ocr_mode == "remote_service" and not self.ocr_service_url:
            raise ValueError(
                "ocr_service_url is required when ocr_mode is remote_service"
            )

        ## Validate retrieval parameters
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be lower than chunk_size")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")

        return self

class EnvSettings(BaseSettings):
    """
        Runtime settings loaded through pydantic-settings

        Args:
            environment: Execution environment
            app_version: Application version
            log_level: Log level
            drive_folder_id: Default Google Drive folder ID
            ocr_mode: OCR mode
            ocr_docker_image: OCR docker image
            ocr_service_url: OCR remote service URL
            gcp_project_id: GCP project ID
            gcp_region: GCP region
            vertex_llm_model: Vertex LLM model
            vertex_embed_model: Vertex embedding model
            gcs_bucket_text: GCS bucket for text artifacts
            gcs_prefix_text: GCS prefix for text artifacts
            gcs_bucket_embeddings: GCS bucket for embeddings artifacts
            gcs_prefix_embeddings: GCS prefix for embeddings artifacts
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap
            top_k: Top-k retrieval
            keep_local: Keep local traces
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        env_prefix="",
        case_sensitive=False,
    )

    environment: str = "dev"
    app_version: str = "1.0.0"
    log_level: LogLevelName = "INFO"
    drive_folder_id: str | None = None
    ocr_mode: OcrModeName = "local_docker"
    ocr_docker_image: str | None = None
    ocr_service_url: str | None = None
    gcp_project_id: str | None = None
    gcp_region: str | None = None
    vertex_llm_model: str | None = None
    vertex_embed_model: str | None = None
    gcs_bucket_text: str | None = None
    gcs_prefix_text: str = "texts/"
    gcs_bucket_embeddings: str | None = None
    gcs_prefix_embeddings: str = "embeddings/"
    chunk_size: int = Field(default=1024, ge=1)
    chunk_overlap: int = Field(default=128, ge=0)
    top_k: int = Field(default=8, ge=1)
    keep_local: bool = False

    @model_validator(mode="after")
    def validate_env_consistency(self) -> "EnvSettings":
        """
            Validate environment settings consistency

            Returns:
                The validated environment settings

            Raises:
                ValueError: If one configuration is inconsistent
        """

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be lower than chunk_size")

        if self.ocr_mode == "local_docker" and not self.ocr_docker_image:
            raise ValueError(
                "ocr_docker_image is required when ocr_mode is local_docker"
            )

        if self.ocr_mode == "remote_service" and not self.ocr_service_url:
            raise ValueError(
                "ocr_service_url is required when ocr_mode is remote_service"
            )

        return self

## ============================================================
## PYDANTIC MODELS (PIPELINE + UI)
## ============================================================
class DriveFileMeta(BaseSchema):
    """
        Metadata for a Google Drive file

        Args:
            file_id: Drive file ID
            name: File name
            mime_type: MIME type
            modified_time: Last modification time
    """

    file_id: str
    name: str
    mime_type: str
    modified_time: Optional[str] = None

class IngestionStatus(BaseSchema):
    """
        Pipeline status output after an ingestion run

        Args:
            drive_folder_id: Ingested folder ID
            downloaded_files: Number of downloaded files
            ocr_processed_files: Number of OCR-processed files
            uploaded_text_files: Number of uploaded text artifacts
            uploaded_embedding_files: Number of uploaded embedding artifacts
            message: Human-readable summary
            details: Optional structured details
    """

    drive_folder_id: str
    downloaded_files: int = 0
    ocr_processed_files: int = 0
    uploaded_text_files: int = 0
    uploaded_embedding_files: int = 0
    message: str = "ok"
    details: Dict[str, Any] = Field(default_factory=dict)

class RagAnswer(BaseSchema):
    """
        RAG answer container

        Args:
            question: Input question
            answer: Generated answer
            sources: Optional source metadata
    """

    question: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)

class IngestionRequest(BaseSchema):
    """
        Ingestion request schema

        Args:
            drive_folder_id: Target Drive folder ID
            recursive: Whether subfolders are traversed
            overwrite: Whether existing outputs can be overwritten
    """

    drive_folder_id: str
    recursive: bool = True
    overwrite: bool = False

class RagQueryRequest(BaseSchema):
    """
        RAG query request schema

        Args:
            question: User question
            top_k: Optional retrieval top-k override
            include_sources: Whether sources are returned
    """

    question: str
    top_k: int | None = Field(default=None, ge=1)
    include_sources: bool = True

class ChunkMetadata(BaseSchema):
    """
        Retrieved chunk metadata schema

        Args:
            chunk_id: Chunk identifier
            file_id: Drive file identifier
            file_name: Source file name
            score: Retrieval score
            text_uri: GCS or local text URI
            embedding_uri: GCS or local embedding URI
    """

    chunk_id: str
    file_id: str | None = None
    file_name: str | None = None
    score: float | None = None
    text_uri: str | None = None
    embedding_uri: str | None = None

class RetrievalResponse(WarningMixin):
    """
        Retrieval response schema

        Args:
            question: Input question
            top_k: Effective top-k retrieval value
            items: Retrieved chunk metadata list
    """

    question: str
    top_k: int = Field(..., ge=1)
    items: list[ChunkMetadata] = Field(default_factory=list)

## ============================================================
## PIPELINE / DATASET / STATUS SCHEMAS
## ============================================================
class DatasetRecord(BaseSchema):
    """
        Generic dataset record schema for RAG ingestion

        Args:
            record_id: Record identifier
            payload: Raw payload content
            metadata: Optional metadata
    """

    record_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

class DatasetInput(BaseSchema):
    """
        Dataset input schema

        Args:
            name: Dataset name
            records: Dataset records
    """

    name: str
    records: list[DatasetRecord]

    @model_validator(mode="after")
    def validate_records(self) -> "DatasetInput":
        """
            Validate dataset records content

            Returns:
                The validated dataset input

            Raises:
                ValueError: If records list is empty
        """

        if not self.records:
            raise ValueError("records must contain at least one item")
        return self

class DatasetOutput(BaseSchema):
    """
        Dataset output schema

        Args:
            name: Dataset name
            row_count: Number of rows
            artifacts: Generated artifacts
    """

    name: str
    row_count: int = Field(..., ge=0)
    artifacts: list[str] = Field(default_factory=list)

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
            Validate progress consistency

            Returns:
                The validated pipeline job

            Raises:
                ValueError: If progress is inconsistent
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
            metrics: Metric list
            summary: Aggregated summary
    """

    metrics: list[MetricPoint] = Field(default_factory=list)
    summary: dict[str, float] = Field(default_factory=dict)

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
    logger_name: str = "rag-drive-gcp"
    context: dict[str, Any] = Field(default_factory=dict)
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
    service: str = "rag-drive-gcp"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

## ============================================================
## PLACEHOLDER VALIDATION
## ============================================================
def _validate_no_placeholders(settings: Settings) -> None:
    """
        Validate that no environment variable still contains a placeholder value

        This project uses placeholders like `<YOUR_VALUE_HERE>` in `.env`.
        If placeholders are detected, the application fails fast.

        Args:
            settings: Loaded settings instance

        Raises:
            ValueError: If one or more settings values contain placeholders
    """

    invalid = []

    for key, value in settings.model_dump().items():
        if isinstance(value, str) and "<YOUR_" in value:
            invalid.append(key)

    if invalid:
        log_and_raise_missing_env(invalid)

## ============================================================
## SETTINGS FACTORY (CACHED)
## ============================================================
@lru_cache(maxsize=1)
def get_settings(env_path: Optional[Path] = None) -> Settings:
    """
        Build and cache the application settings

        Args:
            env_path: Optional .env path override

        Returns:
            Loaded settings instance
    """

    ## Load .env first if present
    load_dotenv_if_present(env_path)

    ## Build settings from environment variables
    settings = Settings(
        environment=os.getenv("ENVIRONMENT", "dev"),
        app_version=os.getenv("APP_VERSION", "1.0.0"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        drive_folder_id=os.getenv("DRIVE_FOLDER_ID"),
        ocr_mode=os.getenv("OCR_MODE", "local_docker"),
        ocr_docker_image=os.getenv("OCR_DOCKER_IMAGE"),
        ocr_service_url=os.getenv("OCR_SERVICE_URL"),
        gcp_project_id=os.getenv("GCP_PROJECT_ID"),
        gcp_region=os.getenv("GCP_REGION"),
        vertex_llm_model=os.getenv("VERTEX_LLM_MODEL"),
        vertex_embed_model=os.getenv("VERTEX_EMBED_MODEL"),
        gcs_bucket_text=os.getenv("GCS_BUCKET_TEXT"),
        gcs_prefix_text=os.getenv("GCS_PREFIX_TEXT", "texts/"),
        gcs_bucket_embeddings=os.getenv("GCS_BUCKET_EMB"),
        gcs_prefix_embeddings=os.getenv("GCS_PREFIX_EMB", "embeddings/"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1024")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "128")),
        top_k=int(os.getenv("TOP_K", "8")),
        keep_local=to_bool(os.getenv("KEEP_LOCAL"), default=False),
    )

    ## Ensure local directories exist
    ensure_directories(
        settings.data_dir,
        settings.logs_dir,
        settings.raw_dir,
        settings.tmp_dir,
        settings.traces_dir,
    )

    ## Fail fast on placeholders
    _validate_no_placeholders(settings)

    return settings