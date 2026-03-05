'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Pydantic request/response schemas for autonomous-ai-platform (MCP server, chat, retrieval, SQL, evaluation)."
'''

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

## ============================================================
## COMMON TYPES
## ============================================================
ProviderName = Literal[
    "auto",
    "local",
    "openai",
    "xai",
    "grok",
    "vertex",
]

ToolName = Literal[
    "rag_search",
    "sql_query",
    "web_search",
    "read_files",
    "run_python",
]

class LoopRequest(BaseModel):
    """
        Request model for autonomous loop endpoint
    """

    query: str = Field(..., min_length=1)
    prefer_local: bool = True
    use_gpu: Optional[bool] = None
    export: bool = True

class LoopResponse(BaseModel):
    """
        Response model for autonomous loop endpoint
    """

    answer: str = ""
    metadata: Dict[str, Any] = {}
    steps: list[Dict[str, Any]] = []
 
## ============================================================
## ERROR RESPONSE
## ============================================================
class ErrorResponse(BaseModel):
    """
        Standard API error response

        Args:
            error: Normalized error code
            message: Human-readable message
            origin: Component where error happened
            details: Diagnostic details dict
            request_id: Optional request correlation id
    """

    error: str
    message: str
    origin: str = Field(default="unknown")
    details: Dict[str, Any] = Field(default_factory=dict)
    request_id: str = Field(default="n/a")

## ============================================================
## HEALTH
## ============================================================
class HealthResponse(BaseModel):
    """
        Health response schema

        Args:
            status: Service status
            version: Service version
            environment: Environment name
    """

    status: str = Field(default="ok")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="dev")

## ============================================================
## CHAT
## ============================================================
class ChatMessage(BaseModel):
    """
        Chat message schema

        Args:
            role: Message role
            content: Message content
    """

    role: Literal["system", "user", "assistant", "tool"] = Field(default="user")
    content: str

class ToolCall(BaseModel):
    """
        Tool call schema

        Args:
            name: Tool name
            arguments: Tool arguments payload
    """

    name: ToolName
    arguments: Dict[str, Any] = Field(default_factory=dict)

class ChatRequest(BaseModel):
    """
        Chat request schema

        Args:
            provider: Provider selection (or auto)
            model: Optional model override
            messages: List of messages
            enable_rag: Whether RAG tool is allowed
            top_k: Retrieval top_k if RAG is used
            metadata: Optional metadata payload
    """

    provider: ProviderName = Field(default="auto")
    model: Optional[str] = Field(default=None)
    messages: List[ChatMessage] = Field(default_factory=list)
    enable_rag: bool = Field(default=True)
    top_k: int = Field(default=8, ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatResponse(BaseModel):
    """
        Chat response schema

        Args:
            provider: Provider used
            model: Model used
            answer: Final answer
            tool_calls: Tool calls executed
            sources: Retrieval sources for citations
            metrics: Runtime metrics
            warnings: Optional warnings list
    """

    provider: str
    model: str
    answer: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)

## ============================================================
## RETRIEVAL
## ============================================================
class RetrievalRequest(BaseModel):
    """
        Retrieval request schema

        Args:
            query: User query
            top_k: Top-k docs
            filters: Optional metadata filters
    """

    query: str
    top_k: int = Field(default=8, ge=1)
    filters: Dict[str, Any] = Field(default_factory=dict)

class RetrievalItem(BaseModel):
    """
        Retrieval item schema

        Args:
            id: Chunk identifier
            text: Chunk text
            score: Similarity score
            metadata: Chunk metadata
    """

    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrievalResponse(BaseModel):
    """
        Retrieval response schema

        Args:
            items: Retrieved items
    """

    items: List[RetrievalItem] = Field(default_factory=list)


## ============================================================
## SQLITE / TEXT-TO-SQL
## ============================================================
class SqlQueryRequest(BaseModel):
    """
        SQL query request schema

        Args:
            query: SQL query string
            limit: Optional row limit
    """

    query: str
    limit: int = Field(default=100, ge=1, le=10000)

class SqlQueryResponse(BaseModel):
    """
        SQL query response schema

        Args:
            columns: Column names list
            rows: Rows as list of dicts
            row_count: Number of rows returned
    """

    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = Field(default=0, ge=0)

class TextToSqlRequest(BaseModel):
    """
        Text-to-SQL request schema

        Args:
            question: Natural language question
            provider: Provider selection (or auto)
            model: Optional model override
            schema_hint: Optional schema hint string
    """

    question: str
    provider: ProviderName = Field(default="auto")
    model: Optional[str] = Field(default=None)
    schema_hint: str = Field(default="")

class TextToSqlResponse(BaseModel):
    """
        Text-to-SQL response schema

        Args:
            sql: Generated SQL query
            provider: Provider used
            model: Model used
            warnings: Optional warnings list
    """

    sql: str
    provider: str
    model: str
    warnings: List[str] = Field(default_factory=list)

## ============================================================
## EVALUATION
## ============================================================
class EvaluationRequest(BaseModel):
    """
        Evaluation request schema

        Args:
            reference: Ground truth reference
            prediction: Model prediction
            metrics: Metric names to compute
    """

    reference: str
    prediction: str
    metrics: List[str] = Field(default_factory=lambda: ["exact_match", "f1_token"])

class EvaluationResponse(BaseModel):
    """
        Evaluation response schema

        Args:
            metrics: Metrics dict
            warnings: Optional warnings list
    """

    metrics: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)