'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Tool registry and tool wrappers (RAG search/ingest, SQLite query, web-like tools) with structured errors and safe logging."
'''

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from src.core.errors import ToolExecutionError, ValidationError
from src.orchestrator.retrieval import ingest_folder_to_vector_store, rag_search
from src.utils.logging_utils import get_logger, log_execution_time
from src.utils.sqlite_manager import SqliteManager
from src.utils.safe_utils import _safe_json, _safe_str
from src.utils.validation_utils import _require_str, _require_int, _optional_bool

logger = get_logger(__name__)

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(frozen=True)
class ToolResult:
    """
        Tool execution result

        Args:
            tool_name: Tool name
            ok: Whether tool succeeded
            output: Tool output payload
            metadata: Extra metadata
            duration_sec: Execution duration
    """

    tool_name: str
    ok: bool
    output: Dict[str, Any]
    metadata: Dict[str, Any]
    duration_sec: float

@dataclass(frozen=True)
class ToolSpec:
    """
        Tool specification

        Args:
            name: Tool name
            description: Short description
            handler: Python callable
            input_schema: Lightweight schema doc (for LLM prompting)
    """

    name: str
    description: str
    handler: Callable[..., ToolResult]
    input_schema: Dict[str, Any]

def _resolve_tool_names() -> List[str]:
    """
        Resolve available tool names

        Returns:
            List of tool names
    """

    registry = get_tools_registry()
    return sorted(list(registry.keys()))
    
## ============================================================
## TOOL IMPLEMENTATIONS
## ============================================================
@log_execution_time
def tool_rag_search(payload: Dict[str, Any]) -> ToolResult:
    """
        Retrieve relevant chunks for a query

        Args:
            payload: Dict with query, top_k, embedding_provider, embedding_model, use_gpu

        Returns:
            ToolResult
    """

    start = time.perf_counter()

    ## Validate payload fields
    query = _require_str(payload.get("query"), "query")
    top_k = _require_int(payload.get("top_k", 5), "top_k", min_value=1, max_value=50)

    embedding_provider = str(payload.get("embedding_provider", "local")).strip() or "local"
    embedding_model = payload.get("embedding_model")
    embedding_model_str = str(embedding_model).strip() if embedding_model else None
    use_gpu = _optional_bool(payload.get("use_gpu"), default=False)

    ## Execute retrieval
    try:
        results = rag_search(
            query=query,
            top_k=top_k,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model_str,
            use_gpu=use_gpu,
        )

        out = [
            {
                "chunk_id": r.chunk_id,
                "source": r.source,
                "score": r.score,
                "text": r.text,
                "metadata": r.metadata,
            }
            for r in results
        ]

        duration = time.perf_counter() - start
        return ToolResult(
            tool_name="rag_search",
            ok=True,
            output={"query": query, "top_k": top_k, "results": out},
            metadata={"embedding_provider": embedding_provider, "embedding_model": embedding_model_str, "use_gpu": use_gpu},
            duration_sec=duration,
        )

    except Exception as exc:
        duration = time.perf_counter() - start
        raise ToolExecutionError(
            message="rag_search tool failed",
            error_code="tool_execution_error",
            details={"payload": _safe_json(payload), "duration_sec": duration, "cause": _safe_str(exc)},
            origin="tools",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc

@log_execution_time
def tool_rag_ingest(payload: Dict[str, Any]) -> ToolResult:
    """
        Ingest a folder into the vector store

        Args:
            payload: Dict with folder, embedding_provider, embedding_model, use_gpu, chunk_size, chunk_overlap

        Returns:
            ToolResult
    """

    start = time.perf_counter()

    ## Validate required folder
    folder = _require_str(payload.get("folder"), "folder")

    ## Optional embedding params
    embedding_provider = str(payload.get("embedding_provider", "local")).strip() or "local"
    embedding_model = payload.get("embedding_model")
    embedding_model_str = str(embedding_model).strip() if embedding_model else None
    use_gpu = _optional_bool(payload.get("use_gpu"), default=False)

    ## Optional chunking overrides
    chunk_size = payload.get("chunk_size")
    chunk_overlap = payload.get("chunk_overlap")

    cs = int(chunk_size) if chunk_size is not None else None
    ov = int(chunk_overlap) if chunk_overlap is not None else None

    ## Execute ingestion
    try:
        summary = ingest_folder_to_vector_store(
            folder=folder,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model_str,
            use_gpu=use_gpu,
            chunk_size=cs,
            chunk_overlap=ov,
        )

        duration = time.perf_counter() - start
        return ToolResult(
            tool_name="rag_ingest",
            ok=True,
            output={"summary": summary},
            metadata={"folder": folder, "embedding_provider": embedding_provider, "embedding_model": embedding_model_str},
            duration_sec=duration,
        )

    except Exception as exc:
        duration = time.perf_counter() - start
        raise ToolExecutionError(
            message="rag_ingest tool failed",
            error_code="tool_execution_error",
            details={"payload": _safe_json(payload), "duration_sec": duration, "cause": _safe_str(exc)},
            origin="tools",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc

@log_execution_time
def tool_sql_query(payload: Dict[str, Any]) -> ToolResult:
    """
        Execute a SQL query against local SQLite database

        Args:
            payload: Dict with db_path, query, params, limit

        Returns:
            ToolResult
    """

    start = time.perf_counter()

    ## Validate payload fields
    db_path = _require_str(payload.get("db_path"), "db_path")
    query = _require_str(payload.get("query"), "query")

    ## Optional params
    params = payload.get("params")
    params_dict = params if isinstance(params, dict) else {}

    ## Optional limit for safety
    limit = payload.get("limit", 200)
    limit_int = _require_int(limit, "limit", min_value=1, max_value=5000)

    ## Execute query with manager
    try:
        manager = SQLiteManager(db_path=db_path)

        ## Safe execution path with fetch
        rows = manager.execute(query=query, params=params_dict, fetch=True)

        ## Apply limit in python as an extra safety guard
        rows_limited = rows[:limit_int]

        duration = time.perf_counter() - start
        return ToolResult(
            tool_name="sql_query",
            ok=True,
            output={"db_path": db_path, "query": query, "params": params_dict, "rows": rows_limited, "row_count": len(rows_limited)},
            metadata={"limit": limit_int},
            duration_sec=duration,
        )

    except Exception as exc:
        duration = time.perf_counter() - start
        raise ToolExecutionError(
            message="sql_query tool failed",
            error_code="tool_execution_error",
            details={"payload": _safe_json(payload), "duration_sec": duration, "cause": _safe_str(exc)},
            origin="tools",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc

## ============================================================
## TOOL REGISTRY
## ============================================================
def get_tools_registry() -> Dict[str, ToolSpec]:
    """
        Return tool registry

        Returns:
            Dict mapping tool name to ToolSpec
    """

    ## Keep schemas minimal but explicit for LLM prompting
    return {
        "rag_search": ToolSpec(
            name="rag_search",
            description="Search similar chunks in the vector store using embeddings.",
            handler=tool_rag_search,
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                    "embedding_provider": {"type": "string", "default": "local"},
                    "embedding_model": {"type": "string"},
                    "use_gpu": {"type": "boolean", "default": False},
                },
            },
        ),
        "rag_ingest": ToolSpec(
            name="rag_ingest",
            description="Ingest local folder into vector store (chunk, embed, persist).",
            handler=tool_rag_ingest,
            input_schema={
                "type": "object",
                "required": ["folder"],
                "properties": {
                    "folder": {"type": "string"},
                    "embedding_provider": {"type": "string", "default": "local"},
                    "embedding_model": {"type": "string"},
                    "use_gpu": {"type": "boolean", "default": False},
                    "chunk_size": {"type": "integer"},
                    "chunk_overlap": {"type": "integer"},
                },
            },
        ),
        "sql_query": ToolSpec(
            name="sql_query",
            description="Execute SQL on local SQLite database and return rows.",
            handler=tool_sql_query,
            input_schema={
                "type": "object",
                "required": ["db_path", "query"],
                "properties": {
                    "db_path": {"type": "string"},
                    "query": {"type": "string"},
                    "params": {"type": "object"},
                    "limit": {"type": "integer", "default": 200},
                },
            },
        ),
    }

## ============================================================
## PUBLIC TOOL EXECUTOR
## ============================================================
@log_execution_time
def run_tool(tool_name: str, payload: Dict[str, Any]) -> ToolResult:
    """
        Execute a tool from the registry

        Args:
            tool_name: Tool name
            payload: Tool input payload

        Returns:
            ToolResult
    """

    ## Resolve tool spec
    registry = get_tools_registry()
    spec = registry.get(tool_name)

    if spec is None:
        raise ValidationError(
            message="Unknown tool name",
            error_code="validation_error",
            details={"tool_name": tool_name, "available": list(registry.keys())},
            origin="tools",
            cause=None,
            http_status=400,
            is_retryable=False,
        )

    ## Execute tool handler
    result = spec.handler(payload)

    ## Log tool success
    logger.info(
        "ToolExecuted | tool=%s | ok=%s | duration_sec=%.4f",
        tool_name,
        result.ok,
        result.duration_sec,
    )

    return result