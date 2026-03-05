'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Text-to-SQL agent: generate safe SQLite queries from natural language with schema grounding and structured errors."
'''

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.core.errors import OrchestrationError, SqlExecutionError, ValidationError
from src.orchestrator.routing import route_chat_completion
from src.utils.logging_utils import get_logger, log_execution_time
from src.utils.sqlite_manager import SqliteManager
from src.utils.safe_utils import _safe_str
from src.utils.validation_utils import _must_be_non_empty

logger = get_logger(__name__)

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(frozen=True)
class SqlDraft:
    """
        SQL draft

        Args:
            sql: SQL statement
            rationale: Short reasoning
            safety_notes: Optional safety notes
    """

    sql: str
    rationale: str
    safety_notes: str


@dataclass(frozen=True)
class SqlAnswer:
    """
        SQL execution answer

        Args:
            sql: SQL statement executed
            rows: Result rows
            row_count: Number of rows returned
            metadata: Extra metadata
    """

    sql: str
    rows: List[Dict[str, Any]]
    row_count: int
    metadata: Dict[str, Any]

## ============================================================
## INTERNAL HELPERS
## ============================================================
def _is_read_only_sql(sql: str) -> bool:
    """
        Check if SQL is read-only (SELECT / WITH)

        Args:
            sql: SQL string

        Returns:
            Boolean
    """

    s = sql.strip().lower()

    ## Remove leading comments
    s = re.sub(r"^\s*(--.*\n|/\*.*?\*/\s*)+", "", s, flags=re.DOTALL)

    ## Allow only SELECT or WITH queries
    return s.startswith("select") or s.startswith("with")

def _reject_multiple_statements(sql: str) -> None:
    """
        Reject multiple statements separated by semicolons

        Args:
            sql: SQL string

        Returns:
            None
    """

    ## SQLite allows multiple statements with executescript
    ## We explicitly prevent this for safety
    parts = [p.strip() for p in sql.split(";") if p.strip()]
    if len(parts) > 1:
        raise ValidationError(
            message="Multiple SQL statements are not allowed",
            error_code="validation_error",
            details={"statements_count": len(parts)},
            origin="text_to_sql",
            cause=None,
            http_status=400,
            is_retryable=False,
        )

def _normalize_limit(sql: str, default_limit: int) -> str:
    """
        Ensure a LIMIT is present to prevent huge results

        Args:
            sql: SQL string
            default_limit: Default limit

        Returns:
            SQL string with LIMIT
    """

    s = sql.strip()

    ## If user already has LIMIT, keep it
    if re.search(r"\blimit\b\s+\d+", s, flags=re.IGNORECASE):
        return s

    ## Add limit at end
    return f"{s.rstrip(';')} LIMIT {int(default_limit)}"

def _get_sqlite_schema(db_path: str) -> Dict[str, Any]:
    """
        Extract SQLite schema for grounding

        Args:
            db_path: SQLite db file path

        Returns:
            Schema dict
    """

    ## Read schema via sqlite_master
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' "
                "ORDER BY type, name;"
            )
            rows = [dict(r) for r in cur.fetchall()]
            return {"objects": rows}

    except Exception as exc:
        raise SqlExecutionError(
            message="Failed to read SQLite schema",
            error_code="sql_execution_error",
            details={"db_path": db_path},
            origin="text_to_sql",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc

def _build_text_to_sql_prompt(
    user_question: str,
    schema: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
        Build messages for Text-to-SQL generation

        Args:
            user_question: Natural language question
            schema: SQLite schema dict

        Returns:
            Chat messages
    """

    ## Force JSON output to parse reliably
    system = (
        "You are a Text-to-SQL agent for SQLite.\n"
        "Return ONLY valid JSON.\n"
        "You MUST generate a single read-only SQL query (SELECT or WITH).\n"
        "Do not use INSERT/UPDATE/DELETE/ALTER/DROP.\n"
        "Do not use multiple statements.\n"
        "Schema:\n"
        f"{_safe_json(schema)}\n"
        "JSON schema:\n"
        "{\n"
        '  "sql": "string",\n'
        '  "rationale": "short string",\n'
        '  "safety_notes": "short string"\n'
        "}\n"
    )

    user = f"User question:\n{user_question}"

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def _parse_sql_draft(text: str) -> SqlDraft:
    """
        Parse SQL draft from model output

        Args:
            text: Model output

        Returns:
            SqlDraft
    """

    ## Basic JSON parse
    try:
        import json

        data = json.loads(text)
    except Exception:
        data = {}

    if not isinstance(data, dict):
        data = {}

    sql = str(data.get("sql", "")).strip()
    rationale = str(data.get("rationale", "")).strip()
    safety = str(data.get("safety_notes", "")).strip()

    return SqlDraft(sql=sql, rationale=rationale, safety_notes=safety)

## ============================================================
## PUBLIC API
## ============================================================
@log_execution_time
def generate_sql(
    user_question: str,
    *,
    db_path: str,
    prefer_local: bool = True,
    use_gpu: Optional[bool] = None,
) -> SqlDraft:
    """
        Generate a safe SQLite SELECT query from natural language

        Args:
            user_question: User question
            db_path: SQLite database path
            prefer_local: Prefer local LLM
            use_gpu: GPU flag

        Returns:
            SqlDraft
    """

    ## Validate inputs
    question = _must_be_non_empty(user_question, "user_question")
    dbp = _must_be_non_empty(db_path, "db_path")

    ## Extract schema for grounding
    schema = _get_sqlite_schema(dbp)

    ## Build prompt
    messages = _build_text_to_sql_prompt(question, schema=schema)

    try:
        ## Call LLM through routing layer
        out = route_chat_completion(
            messages=messages,
            prefer_local=prefer_local,
            use_gpu=use_gpu,
            temperature=0.0,
            top_p=1.0,
            max_tokens=600,
        )

        ## Parse JSON response
        draft = _parse_sql_draft(str(out.get("text", "")).strip())

        ## Enforce safety guards
        _reject_multiple_statements(draft.sql)

        if not _is_read_only_sql(draft.sql):
            raise ValidationError(
                message="Generated SQL is not read-only",
                error_code="validation_error",
                details={"sql_preview": draft.sql[:200]},
                origin="text_to_sql",
                cause=None,
                http_status=400,
                is_retryable=False,
            )

        ## Ensure LIMIT for safety
        sql_limited = _normalize_limit(draft.sql, default_limit=200)

        return SqlDraft(
            sql=sql_limited,
            rationale=draft.rationale or "n/a",
            safety_notes=draft.safety_notes or "Read-only enforced, LIMIT applied",
        )

    except (ValidationError, SqlExecutionError):
        raise

    except Exception as exc:
        raise OrchestrationError(
            message="Failed to generate SQL",
            error_code="orchestration_error",
            details={"cause": _safe_str(exc)},
            origin="text_to_sql",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc

@log_execution_time
def execute_text_to_sql(
    user_question: str,
    *,
    db_path: str,
    prefer_local: bool = True,
    use_gpu: Optional[bool] = None,
) -> SqlAnswer:
    """
        Generate SQL then execute it on SQLite

        Args:
            user_question: Natural language question
            db_path: SQLite db path
            prefer_local: Prefer local backend
            use_gpu: GPU flag

        Returns:
            SqlAnswer
    """

    ## Generate safe SQL
    draft = generate_sql(
        user_question=user_question,
        db_path=db_path,
        prefer_local=prefer_local,
        use_gpu=use_gpu,
    )

    ## Execute query through SQLite manager
    try:
        manager = SQLiteManager(db_path=db_path)

        ## Execute and fetch rows
        rows = manager.execute(query=draft.sql, params={}, fetch=True)

        return SqlAnswer(
            sql=draft.sql,
            rows=rows,
            row_count=len(rows),
            metadata={
                "rationale": draft.rationale,
                "safety_notes": draft.safety_notes,
            },
        )

    except Exception as exc:
        raise SqlExecutionError(
            message="Failed to execute generated SQL",
            error_code="sql_execution_error",
            details={"db_path": db_path, "sql_preview": draft.sql[:500], "cause": _safe_str(exc)},
            origin="text_to_sql",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc