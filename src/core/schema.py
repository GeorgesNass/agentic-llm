'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Pydantic request/response schemas for LLM proxy gateway (cost, completion, embeddings, evaluation)."
'''

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

## ============================================================
## COMMON TYPES
## ============================================================
ProviderName = Literal[
    "openai",
    "google",
    "xai",
    "vertex",
    "azure_openai",
    "local",
    "auto",
]

TaskMode = Literal[
    "chat",
    "embeddings",
]

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

class HealthResponse(BaseModel):
    """
        Health response schema

        Args:
            status: Service status
            version: Application version
    """

    status: str = Field(default="ok")
    version: str = Field(default="1.0.0")

## ============================================================
## COST SIMULATION
## ============================================================
class CostCompletionOptions(BaseModel):
    """
        Options used for completion cost simulation

        Args:
            expected_output_tokens: Expected output tokens for simulation
    """

    expected_output_tokens: int = Field(default=256, ge=0)

class CostEmbeddingsOptions(BaseModel):
    """
        Options used for embeddings cost simulation

        Args:
            chunk_size: Chunk size in characters (simple heuristic)
            chunk_overlap: Chunk overlap in characters
    """

    chunk_size: int = Field(default=1000, ge=1)
    chunk_overlap: int = Field(default=150, ge=0)

class CostSimulateRequest(BaseModel):
    """
        Cost simulation request schema

        Args:
            mode: chat or embeddings
            providers: List of providers to simulate
            model: Optional model override (if absent, provider default is used)
            text: Direct text input (mutually exclusive with path)
            path: Folder path containing txt files (mutually exclusive with text)
            recursive: Whether to scan folders recursively
            max_chars_per_file: Max chars to read per file
            completion: Completion simulation options
            embeddings: Embeddings simulation options
            include_per_file: Whether to return per-file breakdown
    """

    mode: TaskMode = Field(default="chat")
    providers: List[str] = Field(default_factory=list)
    model: Optional[str] = Field(default=None)

    text: Optional[str] = Field(default=None)
    path: Optional[str] = Field(default=None)
    recursive: bool = Field(default=True)
    max_chars_per_file: int = Field(default=200_000, ge=1)

    completion: CostCompletionOptions = Field(default_factory=CostCompletionOptions)
    embeddings: CostEmbeddingsOptions = Field(default_factory=CostEmbeddingsOptions)

    include_per_file: bool = Field(default=False)

class CostSimulateSummary(BaseModel):
    """
        Summary of the cost simulation input

        Args:
            mode: Simulated task mode
            n_files: Number of files processed (0 if text)
            n_chars: Total characters processed
    """

    mode: TaskMode
    n_files: int
    n_chars: int

class CostSimulateResult(BaseModel):
    """
        Per-provider cost estimate result

        Args:
            provider: Provider identifier
            model: Model name used
            input_tokens: Estimated input tokens
            output_tokens_assumed: Assumed output tokens (chat only)
            total_tokens: Total tokens used for pricing
            estimated_cost_usd: Estimated cost in USD
            estimation_mode: Token estimation mode (approx/tokenizer)
            price_source: Source of pricing values (config)
    """

    provider: str
    model: str
    input_tokens: int
    output_tokens_assumed: int = 0
    total_tokens: int
    estimated_cost_usd: float
    estimation_mode: str = "approx"
    price_source: str = "config"

class CostSimulatePerFile(BaseModel):
    """
        Optional per-file breakdown entry

        Args:
            file_path: File path
            n_chars: Characters read
            approx_tokens: Approx token estimate
    """

    file_path: str
    n_chars: int
    approx_tokens: int

class CostSimulateResponse(BaseModel):
    """
        Cost simulation response schema

        Args:
            summary: Summary of the input data
            results: List of cost estimates by provider/model
            per_file: Optional per-file breakdown
            warnings: Optional warnings list
    """

    summary: CostSimulateSummary
    results: List[CostSimulateResult]
    per_file: Optional[List[CostSimulatePerFile]] = None
    warnings: List[str] = Field(default_factory=list)

## ============================================================
## CHAT COMPLETIONS (GENERIC)
## ============================================================
class ChatMessage(BaseModel):
    """
        OpenAI-like chat message

        Args:
            role: Message role
            content: Message content
    """

    role: Literal["system", "user", "assistant", "tool"] = Field(default="user")
    content: str

class ChatCompletionRequest(BaseModel):
    """
        Generic chat completion request schema

        Notes:
            - 'extra' is forwarded to provider payload (pass-through)
            - 'provider' can be 'auto' to select based on model mapping

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

    provider: ProviderName = Field(default="auto")
    model: str
    messages: List[ChatMessage]

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = Field(default=False)

    extra: Dict[str, Any] = Field(default_factory=dict)
    return_raw: bool = Field(default=False)

class ChatCompletionChoice(BaseModel):
    """
        Minimal OpenAI-like completion choice schema

        Args:
            index: Choice index
            message: Assistant message
            finish_reason: Finish reason
    """

    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class UsageInfo(BaseModel):
    """
        Token usage schema

        Args:
            prompt_tokens: Prompt tokens
            completion_tokens: Completion tokens
            total_tokens: Total tokens
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    """
        Generic chat completion response schema

        Args:
            provider: Provider used
            model: Model used
            choices: Completion choices
            usage: Token usage if available
            cost_usd: Optional cost estimation
            raw: Optional raw provider response
            warnings: Warnings list
    """

    provider: str
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    cost_usd: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)

## ============================================================
## EMBEDDINGS (GENERIC)
## ============================================================
class EmbeddingsRequest(BaseModel):
    """
        Generic embeddings request schema

        Args:
            provider: Provider identifier
            model: Embeddings model
            input: Text or list of texts
            extra: Provider-specific passthrough parameters
            return_raw: Whether to return provider raw response
    """

    provider: ProviderName = Field(default="auto")
    model: str
    input: Union[str, List[str]]
    extra: Dict[str, Any] = Field(default_factory=dict)
    return_raw: bool = Field(default=False)

class EmbeddingItem(BaseModel):
    """
        Embedding item schema

        Args:
            index: Index
            embedding: Vector
    """

    index: int
    embedding: List[float]

class EmbeddingsResponse(BaseModel):
    """
        Embeddings response schema

        Args:
            provider: Provider used
            model: Model used
            data: Embeddings data
            usage: Token usage if available
            cost_usd: Optional cost estimation
            raw: Optional raw provider response
            warnings: Warnings list
    """

    provider: str
    model: str
    data: List[EmbeddingItem]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    cost_usd: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)

## ============================================================
## EVALUATION
## ============================================================
class EvaluationRequest(BaseModel):
    """
        Evaluation request schema (single or batch)

        Args:
            reference: Reference string (optional if references provided)
            prediction: Prediction string (optional if predictions provided)
            references: Batch references
            predictions: Batch predictions
            metrics: Metrics list to compute
            use_embeddings: Whether to compute cosine similarity using embeddings
            provider: Provider for embeddings cosine (if enabled)
            embedding_model: Embedding model for cosine (if enabled)
    """

    reference: Optional[str] = None
    prediction: Optional[str] = None

    references: Optional[List[str]] = None
    predictions: Optional[List[str]] = None

    metrics: List[MetricName] = Field(
        default_factory=lambda: [
            "exact_match",
            "contains_ref",
            "f1_token",
            "jaccard",
            "rouge",
            "bleu",
        ]
    )

    use_embeddings: bool = Field(default=False)
    provider: ProviderName = Field(default="auto")
    embedding_model: Optional[str] = None

class EvaluationItem(BaseModel):
    """
        Evaluation output for a single pair

        Args:
            index: Pair index
            metrics: Computed metrics dict
            warnings: Optional warnings for this item
    """

    index: int
    metrics: Dict[str, float]
    warnings: List[str] = Field(default_factory=list)

class EvaluationResponse(BaseModel):
    """
        Evaluation response schema

        Args:
            items: Per-item evaluation
            aggregate: Aggregate metrics (mean)
            warnings: Global warnings
    """

    items: List[EvaluationItem]
    aggregate: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)