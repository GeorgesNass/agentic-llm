'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Pydantic schemas and typed data contracts for autonomous-ai-platform."
'''

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
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
ProviderName = Literal["auto", "local", "openai", "xai", "grok", "vertex"]
ToolName = Literal[
    "rag_search", "sql_query", "web_search", "read_files", "run_python"
]
MessageRole = Literal["system", "user", "assistant", "tool"]
JobStatusName = Literal["pending", "running", "success", "failed", "cancelled"]
TaskTypeName = Literal[
    "chat", "retrieval", "sql", "text_to_sql", "evaluation", "indexing", "export"
]

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
URL_PATTERN = re.compile(r"^https?://[^\s]+$")
SAFE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9._:/-]+$")

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
## SETTINGS AND CONFIG SCHEMAS
## ============================================================
@dataclass(frozen=True)
class LlmRuntimeConfig:
    """
        Typed runtime configuration for LLM execution

        Args:
            provider: Active provider name
            model: Active model name
            temperature: Sampling temperature
            max_tokens: Maximum generated tokens
            timeout_seconds: Request timeout in seconds
    """

    provider: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: int

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

@dataclass(frozen=True)
class RagRuntimeConfig:
    """
        Typed runtime configuration for retrieval execution

        Args:
            enabled: Whether retrieval is enabled
            top_k: Retrieval depth
            embedding_dimension: Embedding vector size
            score_threshold: Optional score threshold
    """

    enabled: bool
    top_k: int
    embedding_dimension: int
    score_threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

class AppSettings(BaseSettings):
    """
        Settings model for autonomous-ai-platform

        Args:
            app_name: Application name
            environment: Runtime environment
            default_provider: Default provider
            default_model: Default model name
            enable_rag: Default RAG activation flag
            default_top_k: Default retrieval depth
            embedding_dimension: Default embedding dimension
            request_timeout_seconds: Default timeout in seconds
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        env_prefix="APP_",
        case_sensitive=False,
    )

    app_name: str = "autonomous-ai-platform"
    environment: str = "dev"
    default_provider: ProviderName = "auto"
    default_model: str = "default"
    enable_rag: bool = True
    default_top_k: int = Field(default=8, ge=1, le=100)
    embedding_dimension: int = Field(default=768, ge=1, le=65536)
    request_timeout_seconds: int = Field(default=60, ge=1, le=3600)

class ModelGenerationConfig(BaseSchema):
    """
        LLM generation configuration schema

        Args:
            provider: Provider name
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum number of generated tokens
            timeout_seconds: Timeout in seconds
            stream: Whether streaming is enabled
    """

    provider: ProviderName = "auto"
    model: str = Field(..., min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=65536)
    timeout_seconds: int = Field(default=60, ge=1, le=3600)
    stream: bool = False

class RagConfig(BaseSchema):
    """
        Retrieval configuration schema

        Args:
            enabled: Whether retrieval is enabled
            top_k: Retrieval depth
            embedding_dimension: Embedding dimension
            score_threshold: Optional score threshold
            vector_backend: Optional backend name
    """

    enabled: bool = True
    top_k: int = Field(default=8, ge=1, le=100)
    embedding_dimension: int = Field(default=768, ge=1, le=65536)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    vector_backend: str = Field(default="faiss", min_length=1)

class PipelineConfig(BaseSchema):
    """
        Pipeline execution configuration schema

        Args:
            job_name: Pipeline job name
            batch_size: Batch size
            max_workers: Number of workers
            retry_count: Retry count
            stream: Whether streaming is enabled
    """

    job_name: str = Field(default="default-job", min_length=1)
    batch_size: int = Field(default=1, ge=1, le=10000)
    max_workers: int = Field(default=1, ge=1, le=512)
    retry_count: int = Field(default=0, ge=0, le=20)
    stream: bool = False

## ============================================================
## COMMON OPERATIONAL SCHEMAS
## ============================================================
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

class HealthResponse(BaseSchema):
    """
        Health response schema

        Args:
            status: Service status
            version: Service version
            environment: Environment name
    """

    status: str = Field(default="ok", min_length=1)
    version: str = Field(default="1.0.0", min_length=1)
    environment: str = Field(default="dev", min_length=1)

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

class StructuredLogEvent(BaseSchema):
    """
        Structured log schema

        Args:
            level: Log level
            event: Event name
            message: Human-readable message
            logger_name: Logger name
            context: Additional context
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    event: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    logger_name: str = Field(default="app", min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)

class QueueEvent(BaseSchema):
    """
        Message queue or event bus schema

        Args:
            event_id: Unique event identifier
            event_type: Event type
            source: Event source
            payload: Event payload
    """

    event_id: str = Field(..., min_length=1)
    event_type: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id", "event_type", "source")
    @classmethod
    def validate_safe_names(cls, value: str) -> str:
        """
            Validate safe identifier-like strings

            Args:
                value: Candidate identifier string

            Returns:
                The validated identifier string

            Raises:
                ValueError: If the value contains unsupported characters
        """

        ## Ensure event identifiers remain filesystem and API friendly
        if not SAFE_NAME_PATTERN.match(value):
            raise ValueError("value contains unsupported characters")
        return value

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
## DATASET AND PIPELINE SCHEMAS
## ============================================================
class DatasetRecord(BaseSchema):
    """
        Generic dataset record schema

        Args:
            record_id: Record identifier
            text: Main text field
            label: Optional label
            metadata: Optional metadata
    """

    record_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class DatasetInput(BaseSchema):
    """
        Dataset input schema

        Args:
            name: Dataset name
            records: Dataset records
    """

    name: str = Field(..., min_length=1)
    records: list[DatasetRecord] = Field(default_factory=list)

    @field_validator("records")
    @classmethod
    def validate_non_empty_records(
        cls, value: list[DatasetRecord]
    ) -> list[DatasetRecord]:
        """
            Validate that the dataset contains at least one record

            Args:
                value: Dataset records

            Returns:
                The validated records list

            Raises:
                ValueError: If the records list is empty
        """

        ## Prevent empty dataset payloads
        if not value:
            raise ValueError("records must contain at least one item")
        return value

class DatasetOutput(BaseSchema):
    """
        Dataset output schema

        Args:
            name: Dataset name
            row_count: Number of rows
            artifacts: Generated artifacts list
    """

    name: str = Field(..., min_length=1)
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

    task_id: str = Field(..., min_length=1)
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

## ============================================================
## CHAT, LOOP AND RETRIEVAL SCHEMAS
## ============================================================
class LoopRequest(BaseSchema):
    """
        Request model for the autonomous loop endpoint

        Args:
            query: User query
            prefer_local: Whether local provider should be preferred
            use_gpu: Optional override for GPU usage
            export: Whether export is enabled
    """

    query: str = Field(..., min_length=1)
    prefer_local: bool = True
    use_gpu: bool | None = None
    export: bool = True

class LoopResponse(WarningMixin):
    """
        Response model for the autonomous loop endpoint

        Args:
            answer: Final generated answer
            metadata: Additional execution metadata
            steps: Step-by-step execution trace
    """

    answer: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)
    steps: list[dict[str, Any]] = Field(default_factory=list)

class ChatMessage(BaseSchema):
    """
        Chat message schema

        Args:
            role: Message role
            content: Message content
            name: Optional participant name
    """

    role: MessageRole = "user"
    content: str = Field(..., min_length=1)
    name: str | None = None

class ToolCall(BaseSchema):
    """
        Tool call schema

        Args:
            name: Tool name
            arguments: Tool arguments payload
    """

    name: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)

class SourceItem(BaseSchema):
    """
        Retrieval source schema

        Args:
            id: Source identifier
            title: Optional title
            text: Source text
            score: Optional score
            url: Optional source URL
            metadata: Optional metadata
    """

    id: str = Field(..., min_length=1)
    title: str | None = None
    text: str = Field(default="")
    score: float | None = Field(default=None, ge=0.0)
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        """
            Validate the optional source URL

            Args:
                value: Candidate URL

            Returns:
                The validated URL or None

            Raises:
                ValueError: If the URL format is invalid
        """

        ## Validate HTTP(S) links only
        if value is not None and not URL_PATTERN.match(value):
            raise ValueError("url must start with http:// or https://")
        return value

class ChatRequest(BaseSchema):
    """
        Chat request schema

        Args:
            provider: Provider selection or auto
            model: Optional model override
            messages: Messages list
            enable_rag: Whether RAG is allowed
            top_k: Retrieval top_k if RAG is used
            metadata: Optional metadata payload
            stream: Whether streaming is enabled
            user_email: Optional user email
    """

    provider: ProviderName = "auto"
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    enable_rag: bool = True
    top_k: int = Field(default=8, ge=1, le=100)
    metadata: dict[str, Any] = Field(default_factory=dict)
    stream: bool = False
    user_email: str | None = None

    @field_validator("model")
    @classmethod
    def validate_optional_model(cls, value: str | None) -> str | None:
        """
            Validate the optional model name

            Args:
                value: Candidate model name

            Returns:
                The validated model name or None

            Raises:
                ValueError: If the model name is empty after stripping
        """

        ## Reject empty strings when a model override is provided
        if value is not None and not value.strip():
            raise ValueError("model must not be empty when provided")
        return value

    @field_validator("user_email")
    @classmethod
    def validate_email(cls, value: str | None) -> str | None:
        """
            Validate the optional user email field

            Args:
                value: Candidate email value

            Returns:
                The validated email or None

            Raises:
                ValueError: If the email format is invalid
        """

        ## Validate email pattern if provided
        if value is not None and not EMAIL_PATTERN.match(value):
            raise ValueError("user_email is not a valid email")
        return value

    @model_validator(mode="after")
    def validate_chat_request(self) -> "ChatRequest":
        """
            Validate chat request cross-field consistency

            Returns:
                The validated chat request

            Raises:
                ValueError: If the messages list is empty
                ValueError: If top_k is customized while RAG is disabled
        """

        ## Require at least one input message
        if not self.messages:
            raise ValueError("messages must contain at least one item")

        ## Prevent inconsistent RAG-related parameters
        if not self.enable_rag and self.top_k != 8:
            raise ValueError("top_k cannot be customized when enable_rag is False")

        return self

class ChatResponse(WarningMixin):
    """
        Chat response schema

        Args:
            provider: Provider used
            model: Model used
            answer: Final answer
            tool_calls: Tool calls executed
            sources: Retrieval sources
            metrics: Runtime metrics
    """

    provider: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    sources: list[SourceItem] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

class RetrievalRequest(BaseSchema):
    """
        Retrieval request schema

        Args:
            query: User query
            top_k: Top-k documents
            filters: Optional metadata filters
    """

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=1, le=100)
    filters: dict[str, Any] = Field(default_factory=dict)

class RetrievalItem(BaseSchema):
    """
        Retrieval item schema

        Args:
            id: Chunk identifier
            text: Chunk text
            score: Similarity score
            metadata: Chunk metadata
    """

    id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

class RetrievalResponse(BaseSchema):
    """
        Retrieval response schema

        Args:
            items: Retrieved items
    """

    items: list[RetrievalItem] = Field(default_factory=list)

## ============================================================
## SQL, TEXT-TO-SQL, EVALUATION AND BATCH SCHEMAS
## ============================================================
class SqlQueryRequest(BaseSchema):
    """
        SQL query request schema

        Args:
            query: SQL query string
            limit: Optional row limit
    """

    query: str = Field(..., min_length=1)
    limit: int = Field(default=100, ge=1, le=10000)

class SqlQueryResponse(BaseSchema):
    """
        SQL query response schema

        Args:
            columns: Column names
            rows: Rows as dictionaries
            row_count: Number of rows returned
    """

    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_sql_response(self) -> "SqlQueryResponse":
        """
            Validate SQL response row count consistency

            Returns:
                The validated SQL query response

            Raises:
                ValueError: If row_count does not match the number of rows
        """

        ## Ensure exported row_count matches the real payload size
        if self.row_count != len(self.rows):
            raise ValueError("row_count must match len(rows)")
        return self

class TextToSqlRequest(BaseSchema):
    """
        Text-to-SQL request schema

        Args:
            question: Natural language question
            provider: Provider selection or auto
            model: Optional model override
            schema_hint: Optional schema hint string
    """

    question: str = Field(..., min_length=1)
    provider: ProviderName = "auto"
    model: str | None = None
    schema_hint: str = Field(default="")

    @field_validator("model")
    @classmethod
    def validate_t2s_model(cls, value: str | None) -> str | None:
        """
            Validate the optional text-to-SQL model name

            Args:
                value: Candidate model name

            Returns:
                The validated model name or None

            Raises:
                ValueError: If the model name is empty after stripping
        """

        ## Reject empty strings for explicit model overrides
        if value is not None and not value.strip():
            raise ValueError("model must not be empty when provided")
        return value

class TextToSqlResponse(WarningMixin):
    """
        Text-to-SQL response schema

        Args:
            sql: Generated SQL query
            provider: Provider used
            model: Model used
    """

    sql: str = Field(..., min_length=1)
    provider: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)

class EvaluationRequest(BaseSchema):
    """
        Evaluation request schema

        Args:
            reference: Ground truth reference
            prediction: Model prediction
            metrics: Metric names to compute
    """

    reference: str = Field(..., min_length=1)
    prediction: str = Field(..., min_length=1)
    metrics: list[str] = Field(
        default_factory=lambda: ["exact_match", "f1_token"]
    )

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: list[str]) -> list[str]:
        """
            Validate the requested metrics list

            Args:
                value: Candidate metric names list

            Returns:
                A deduplicated and cleaned metric list

            Raises:
                ValueError: If the metrics list is empty after cleaning
        """

        ## Strip empty values and preserve insertion order
        cleaned_values = [item.strip() for item in value if item.strip()]
        if not cleaned_values:
            raise ValueError("metrics must contain at least one metric")
        return list(dict.fromkeys(cleaned_values))

class EvaluationResponse(WarningMixin):
    """
        Evaluation response schema

        Args:
            metrics: Metrics dictionary
    """

    metrics: dict[str, float] = Field(default_factory=dict)

class BatchChatRequest(BaseSchema):
    """
        Batch chat request schema

        Args:
            requests: Chat requests list
    """

    requests: list[ChatRequest] = Field(default_factory=list)

    @field_validator("requests")
    @classmethod
    def validate_batch_inputs(cls, value: list[ChatRequest]) -> list[ChatRequest]:
        """
            Validate batch input requests

            Args:
                value: Candidate chat requests list

            Returns:
                The validated requests list

            Raises:
                ValueError: If the batch is empty
        """

        ## Prevent empty batch submissions
        if not value:
            raise ValueError("requests must contain at least one item")
        return value

class StreamChatChunk(BaseSchema):
    """
        Streaming chat chunk schema

        Args:
            chunk_index: Chunk index
            delta: Generated delta text
            done: Whether the stream is completed
    """

    chunk_index: int = Field(..., ge=0)
    delta: str = Field(default="")
    done: bool = False