'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Pydantic request/response schemas for LLM proxy gateway (cost, completion, embeddings, evaluation)."
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
## COMMON TYPES AND PATTERNS
## ============================================================
ProviderName = Literal[
    "openai", "google", "xai", "vertex", "azure_openai", "local", "auto"
]
TaskMode = Literal["chat", "embeddings"]
MetricName = Literal[
    "exact_match",
    "contains_ref",
    "f1_token",
    "jaccard",
    "rouge",
    "bleu",
    "bertscore",
    "cosine_embedding",
]
MessageRole = Literal["system", "user", "assistant", "tool"]

SAFE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9._:/\-]+$")
URL_PATTERN = re.compile(r"^https?://[^\s]+$")

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
class ProxyRuntimeConfig:
    """
        Typed runtime configuration for the gateway

        Args:
            default_provider: Default provider name
            default_chat_model: Default chat model name
            default_embedding_model: Default embeddings model name
            request_timeout_seconds: Request timeout in seconds
            enable_cost_tracking: Whether cost tracking is enabled
    """

    default_provider: str
    default_chat_model: str
    default_embedding_model: str
    request_timeout_seconds: int
    enable_cost_tracking: bool

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

class AppSettings(BaseSettings):
    """
        Settings model for llm_proxy_gateway

        Args:
            app_name: Application name
            environment: Runtime environment
            default_provider: Default provider
            default_chat_model: Default chat model
            default_embedding_model: Default embeddings model
            request_timeout_seconds: Request timeout in seconds
            enable_cost_tracking: Whether cost tracking is enabled
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        env_prefix="LLM_PROXY_",
        case_sensitive=False,
    )

    app_name: str = "llm_proxy_gateway"
    environment: str = "dev"
    default_provider: ProviderName = "auto"
    default_chat_model: str = "default-chat-model"
    default_embedding_model: str = "default-embedding-model"
    request_timeout_seconds: int = Field(default=60, ge=1, le=3600)
    enable_cost_tracking: bool = True

class GatewayRequestConfig(BaseSchema):
    """
        Generic gateway request configuration schema

        Args:
            provider: Provider identifier
            model: Model name
            timeout_seconds: Request timeout in seconds
            return_raw: Whether to return provider raw response
    """

    provider: ProviderName = "auto"
    model: str = Field(..., min_length=1)
    timeout_seconds: int = Field(default=60, ge=1, le=3600)
    return_raw: bool = False

## ============================================================
## COMMON OPERATIONAL SCHEMAS
## ============================================================
class HealthResponse(BaseSchema):
    """
        Health response schema

        Args:
            status: Service status
            version: Application version
    """

    status: str = Field(default="ok", min_length=1)
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
    logger_name: str = Field(default="llm_proxy_gateway", min_length=1)
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

        ## Ensure identifiers remain API and filesystem friendly
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
## COST SIMULATION
## ============================================================
class CostCompletionOptions(BaseSchema):
    """
        Options used for completion cost simulation

        Args:
            expected_output_tokens: Expected output tokens for simulation
    """

    expected_output_tokens: int = Field(default=256, ge=0)

class CostEmbeddingsOptions(BaseSchema):
    """
        Options used for embeddings cost simulation

        Args:
            chunk_size: Chunk size in characters
            chunk_overlap: Chunk overlap in characters
    """

    chunk_size: int = Field(default=1000, ge=1)
    chunk_overlap: int = Field(default=150, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "CostEmbeddingsOptions":
        """
            Validate chunk overlap consistency

            Returns:
                The validated embeddings options

            Raises:
                ValueError: If overlap is greater than or equal to chunk size
        """

        ## Prevent invalid chunking settings
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be lower than chunk_size")
        return self

class CostSimulateRequest(BaseSchema):
    """
        Cost simulation request schema

        Args:
            mode: chat or embeddings
            providers: List of providers to simulate
            model: Optional model override
            text: Direct text input
            path: Folder path containing txt files
            recursive: Whether to scan folders recursively
            max_chars_per_file: Max chars to read per file
            completion: Completion simulation options
            embeddings: Embeddings simulation options
            include_per_file: Whether to return per-file breakdown
    """

    mode: TaskMode = "chat"
    providers: list[ProviderName] = Field(default_factory=list)
    model: str | None = None
    text: str | None = None
    path: str | None = None
    recursive: bool = True
    max_chars_per_file: int = Field(default=200_000, ge=1)
    completion: CostCompletionOptions = Field(
        default_factory=CostCompletionOptions
    )
    embeddings: CostEmbeddingsOptions = Field(
        default_factory=CostEmbeddingsOptions
    )
    include_per_file: bool = False

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str | None) -> str | None:
        """
            Validate the optional model name

            Args:
                value: Candidate model name

            Returns:
                The validated model name or None

            Raises:
                ValueError: If the model name is empty after stripping
        """

        ## Reject empty explicit model overrides
        if value is not None and not value.strip():
            raise ValueError("model must not be empty when provided")
        return value

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str | None) -> str | None:
        """
            Validate the optional input path

            Args:
                value: Candidate folder path

            Returns:
                The validated path or None

            Raises:
                ValueError: If the path format is invalid
        """

        ## Keep relative-like or URL-safe path strings only
        if value is not None and not SAFE_NAME_PATTERN.match(value):
            raise ValueError("path contains unsupported characters")
        return value

    @model_validator(mode="after")
    def validate_input_source(self) -> "CostSimulateRequest":
        """
            Validate mutually exclusive input source fields

            Returns:
                The validated request

            Raises:
                ValueError: If neither or both text and path are provided
        """

        ## Require exactly one source of input data
        if bool(self.text) == bool(self.path):
            raise ValueError("exactly one of text or path must be provided")
        return self

class CostSimulateSummary(BaseSchema):
    """
        Summary of the cost simulation input

        Args:
            mode: Simulated task mode
            n_files: Number of files processed
            n_chars: Total characters processed
    """

    mode: TaskMode
    n_files: int = Field(..., ge=0)
    n_chars: int = Field(..., ge=0)

class CostSimulateResult(BaseSchema):
    """
        Per-provider cost estimate result

        Args:
            provider: Provider identifier
            model: Model name used
            input_tokens: Estimated input tokens
            output_tokens_assumed: Assumed output tokens
            total_tokens: Total tokens used for pricing
            estimated_cost_usd: Estimated cost in USD
            estimation_mode: Token estimation mode
            price_source: Source of pricing values
    """

    provider: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    input_tokens: int = Field(..., ge=0)
    output_tokens_assumed: int = Field(default=0, ge=0)
    total_tokens: int = Field(..., ge=0)
    estimated_cost_usd: float = Field(..., ge=0.0)
    estimation_mode: str = Field(default="approx", min_length=1)
    price_source: str = Field(default="config", min_length=1)

    @model_validator(mode="after")
    def validate_totals(self) -> "CostSimulateResult":
        """
            Validate token totals consistency

            Returns:
                The validated result

            Raises:
                ValueError: If total_tokens is inconsistent
        """

        ## Keep total token accounting coherent
        if self.total_tokens < self.input_tokens + self.output_tokens_assumed:
            raise ValueError(
                "total_tokens must be >= input_tokens + output_tokens_assumed"
            )
        return self

class CostSimulatePerFile(BaseSchema):
    """
        Optional per-file breakdown entry

        Args:
            file_path: File path
            n_chars: Characters read
            approx_tokens: Approx token estimate
    """

    file_path: str = Field(..., min_length=1)
    n_chars: int = Field(..., ge=0)
    approx_tokens: int = Field(..., ge=0)

class CostSimulateResponse(WarningMixin):
    """
        Cost simulation response schema

        Args:
            summary: Summary of the input data
            results: List of cost estimates by provider/model
            per_file: Optional per-file breakdown
    """

    summary: CostSimulateSummary
    results: list[CostSimulateResult]
    per_file: list[CostSimulatePerFile] | None = None

## ============================================================
## CHAT COMPLETIONS
## ============================================================
class ChatMessage(BaseSchema):
    """
        OpenAI-like chat message

        Args:
            role: Message role
            content: Message content
    """

    role: MessageRole = "user"
    content: str = Field(..., min_length=1)

class ChatCompletionRequest(BaseSchema):
    """
        Generic chat completion request schema

        Args:
            provider: Provider identifier
            model: Model name
            messages: Chat messages list
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            top_p: Top-p sampling
            stream: Stream flag
            extra: Provider-specific passthrough parameters
            return_raw: Whether to return provider raw response
    """

    provider: ProviderName = "auto"
    model: str = Field(..., min_length=1)
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)
    return_raw: bool = False

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        """
            Validate chat messages list

            Args:
                value: Candidate messages list

            Returns:
                The validated messages list

            Raises:
                ValueError: If the list is empty
        """

        ## Prevent empty chat completion payloads
        if not value:
            raise ValueError("messages must contain at least one item")
        return value

class ChatCompletionChoice(BaseSchema):
    """
        Minimal OpenAI-like completion choice schema

        Args:
            index: Choice index
            message: Assistant message
            finish_reason: Finish reason
    """

    index: int = Field(..., ge=0)
    message: ChatMessage
    finish_reason: str | None = None

class UsageInfo(BaseSchema):
    """
        Token usage schema

        Args:
            prompt_tokens: Prompt tokens
            completion_tokens: Completion tokens
            total_tokens: Total tokens
    """

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_usage(self) -> "UsageInfo":
        """
            Validate usage total consistency

            Returns:
                The validated usage info

            Raises:
                ValueError: If total_tokens is inconsistent
        """

        ## Ensure token accounting remains coherent
        if self.total_tokens < self.prompt_tokens + self.completion_tokens:
            raise ValueError(
                "total_tokens must be >= prompt_tokens + completion_tokens"
            )
        return self

class ChatCompletionResponse(WarningMixin):
    """
        Generic chat completion response schema

        Args:
            provider: Provider used
            model: Model used
            choices: Completion choices
            usage: Token usage if available
            cost_usd: Optional cost estimation
            raw: Optional raw provider response
    """

    provider: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    choices: list[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    cost_usd: float | None = Field(default=None, ge=0.0)
    raw: dict[str, Any] | None = None

    @field_validator("choices")
    @classmethod
    def validate_choices(
        cls, value: list[ChatCompletionChoice]
    ) -> list[ChatCompletionChoice]:
        """
            Validate completion choices list

            Args:
                value: Candidate completion choices

            Returns:
                The validated completion choices

            Raises:
                ValueError: If the list is empty
        """

        ## Ensure at least one generated choice is returned
        if not value:
            raise ValueError("choices must contain at least one item")
        return value

## ============================================================
## EMBEDDINGS
## ============================================================
class EmbeddingsRequest(BaseSchema):
    """
        Generic embeddings request schema

        Args:
            provider: Provider identifier
            model: Embeddings model
            input: Text or list of texts
            extra: Provider-specific passthrough parameters
            return_raw: Whether to return provider raw response
    """

    provider: ProviderName = "auto"
    model: str = Field(..., min_length=1)
    input: str | list[str]
    extra: dict[str, Any] = Field(default_factory=dict)
    return_raw: bool = False

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: str | list[str]) -> str | list[str]:
        """
            Validate embeddings input payload

            Args:
                value: Candidate text input or texts list

            Returns:
                The validated input payload

            Raises:
                ValueError: If the payload is empty
        """

        ## Accept either one text or a non-empty list of texts
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("input string must not be empty")
            return value

        if not value:
            raise ValueError("input list must contain at least one item")

        if any(not item.strip() for item in value):
            raise ValueError("input list must not contain empty strings")

        return value

class EmbeddingItem(BaseSchema):
    """
        Embedding item schema

        Args:
            index: Index
            embedding: Vector
    """

    index: int = Field(..., ge=0)
    embedding: list[float]

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: list[float]) -> list[float]:
        """
            Validate embedding vector contents

            Args:
                value: Candidate embedding vector

            Returns:
                The validated embedding vector

            Raises:
                ValueError: If the embedding is empty
        """

        ## Require at least one embedding dimension
        if not value:
            raise ValueError("embedding must contain at least one value")
        return value

class EmbeddingsResponse(WarningMixin):
    """
        Embeddings response schema

        Args:
            provider: Provider used
            model: Model used
            data: Embeddings data
            usage: Token usage if available
            cost_usd: Optional cost estimation
            raw: Optional raw provider response
    """

    provider: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    data: list[EmbeddingItem]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    cost_usd: float | None = Field(default=None, ge=0.0)
    raw: dict[str, Any] | None = None

    @field_validator("data")
    @classmethod
    def validate_data(cls, value: list[EmbeddingItem]) -> list[EmbeddingItem]:
        """
            Validate embeddings response data list

            Args:
                value: Candidate embeddings data list

            Returns:
                The validated data list

            Raises:
                ValueError: If the list is empty
        """

        ## Ensure at least one embedding vector is returned
        if not value:
            raise ValueError("data must contain at least one item")
        return value

## ============================================================
## EVALUATION
## ============================================================
class EvaluationRequest(BaseSchema):
    """
        Evaluation request schema for single or batch mode

        Args:
            reference: Reference string
            prediction: Prediction string
            references: Batch references
            predictions: Batch predictions
            metrics: Metrics list to compute
            use_embeddings: Whether to compute cosine similarity using embeddings
            provider: Provider for embeddings cosine
            embedding_model: Embedding model for cosine
    """

    reference: str | None = None
    prediction: str | None = None
    references: list[str] | None = None
    predictions: list[str] | None = None
    metrics: list[MetricName] = Field(
        default_factory=lambda: [
            "exact_match",
            "contains_ref",
            "f1_token",
            "jaccard",
            "rouge",
            "bleu",
        ]
    )
    use_embeddings: bool = False
    provider: ProviderName = "auto"
    embedding_model: str | None = None

    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_model(cls, value: str | None) -> str | None:
        """
            Validate the optional embedding model name

            Args:
                value: Candidate embedding model name

            Returns:
                The validated model name or None

            Raises:
                ValueError: If the model name is empty after stripping
        """

        ## Reject empty explicit embedding model overrides
        if value is not None and not value.strip():
            raise ValueError("embedding_model must not be empty when provided")
        return value

    @model_validator(mode="after")
    def validate_mode(self) -> "EvaluationRequest":
        """
            Validate single-item versus batch evaluation inputs

            Returns:
                The validated evaluation request

            Raises:
                ValueError: If neither single nor batch mode is valid
        """

        ## Support exactly one mode: single pair or batch pairs
        single_mode = self.reference is not None or self.prediction is not None
        batch_mode = self.references is not None or self.predictions is not None

        if single_mode and batch_mode:
            raise ValueError(
                "use either reference/prediction or references/predictions"
            )

        if single_mode:
            if not self.reference or not self.prediction:
                raise ValueError("reference and prediction must both be provided")
            return self

        if batch_mode:
            if not self.references or not self.predictions:
                raise ValueError("references and predictions must both be provided")
            if len(self.references) != len(self.predictions):
                raise ValueError("references and predictions must have the same length")
            return self

        raise ValueError("an evaluation input pair or batch must be provided")

class EvaluationItem(BaseSchema):
    """
        Evaluation output for a single pair

        Args:
            index: Pair index
            metrics: Computed metrics dictionary
            warnings: Optional warnings for this item
    """

    index: int = Field(..., ge=0)
    metrics: dict[str, float]
    warnings: list[str] = Field(default_factory=list)

class EvaluationResponse(WarningMixin):
    """
        Evaluation response schema

        Args:
            items: Per-item evaluation
            aggregate: Aggregate metrics
    """

    items: list[EvaluationItem]
    aggregate: dict[str, float] = Field(default_factory=dict)

    @field_validator("items")
    @classmethod
    def validate_items(cls, value: list[EvaluationItem]) -> list[EvaluationItem]:
        """
            Validate evaluation items list

            Args:
                value: Candidate evaluation items

            Returns:
                The validated evaluation items

            Raises:
                ValueError: If the list is empty
        """

        ## Ensure at least one evaluation result is returned
        if not value:
            raise ValueError("items must contain at least one item")
        return value