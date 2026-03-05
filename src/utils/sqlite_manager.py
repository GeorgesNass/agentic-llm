'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "SQLite database initialization and query execution utilities for autonomous-ai-platform."
'''

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.errors import SqlExecutionError
from src.utils.logging_utils import get_logger, log_execution_time

logger = get_logger(__name__)

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(frozen=True)
class SqliteQueryResult:
    """
        Structured SQLite query result

        Args:
            columns: List of column names
            rows: Rows as list of dicts
            row_count: Number of rows returned
    """

    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int

## ============================================================
## SQLITE MANAGER
## ============================================================
class SqliteManager:
    """
        SQLite manager for local Text-to-SQL workflows

        Design:
            - Uses sqlite3 built-in module
            - Provides safe execution with structured errors
            - Provides schema inspection helpers for prompting

        Args:
            db_path: Path to SQLite database file
    """
    def __init__(self, db_path: str | Path) -> None:
        ## Normalize and resolve the database path
        self.db_path = Path(db_path).expanduser().resolve()

        ## Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        ## If database file does not exist, create an empty DB
        if not self.db_path.exists():
            logger.info(
                "SQLite DB not found, creating empty DB | path=%s",
                self.db_path,
            )
            self._create_empty_db()

    def _create_empty_db(self) -> None:
        """
            Create an empty SQLite DB file

            Returns:
                None
        """

        try:
            ## Open connection to create the file physically
            with sqlite3.connect(str(self.db_path)) as conn:

                ## Enable WAL mode for better concurrency behavior
                conn.execute("PRAGMA journal_mode=WAL;")

                conn.commit()

        except Exception as exc:
            ## Wrap any low-level exception into structured SqlExecutionError
            raise SqlExecutionError(
                message="Failed to create SQLite database",
                error_code="sql_execution_error",
                details={"db_path": str(self.db_path)},
                origin="sqlite_manager",
                cause=exc,
                http_status=500,
                is_retryable=False,
            ) from exc

    def _connect(self) -> sqlite3.Connection:
        """
            Open a SQLite connection

            Returns:
                sqlite3.Connection
        """

        try:
            ## Create SQLite connection
            conn = sqlite3.connect(str(self.db_path))

            ## Enable row access by column name
            conn.row_factory = sqlite3.Row

            return conn

        except Exception as exc:
            ## Convert connection failure into structured error
            raise SqlExecutionError(
                message="Failed to connect to SQLite database",
                error_code="sql_execution_error",
                details={"db_path": str(self.db_path)},
                origin="sqlite_manager",
                cause=exc,
                http_status=500,
                is_retryable=True,
            ) from exc
            
    @log_execution_time
    def execute_query(
        self,
        query: str,
        limit: int = 100,
        parameters: Optional[Tuple[Any, ...]] = None,
    ) -> SqliteQueryResult:
        """
            Execute a SQL query with an optional row limit

            High-level workflow:
                1) Validate query
                2) Connect to SQLite
                3) Execute query
                4) Fetch rows and format as dict list
                5) Return structured result

            Args:
                query: SQL query string
                limit: Max number of rows to return
                parameters: Optional SQL query parameters tuple

            Returns:
                SqliteQueryResult

            Raises:
                SqlExecutionError: If query execution fails
        """

        ## Basic validation
        raw_query = (query or "").strip()
        if not raw_query:
            raise SqlExecutionError(
                message="Empty SQL query",
                error_code="validation_error",
                details={"query": query},
                origin="sqlite_manager",
                cause=None,
                http_status=400,
                is_retryable=False,
            )

        ## Apply LIMIT safely if query is a SELECT
        final_query = raw_query
        if raw_query.lower().startswith("select") and "limit" not in raw_query.lower():
            final_query = f"{raw_query.rstrip(';')} LIMIT {int(limit)};"

        params = parameters or tuple()

        ## Execute query
        try:
            with self._connect() as conn:
                cursor = conn.execute(final_query, params)
                rows = cursor.fetchall()

                ## Convert rows to dict
                columns = [col[0] for col in cursor.description] if cursor.description else []
                row_dicts = [dict(row) for row in rows]

                return SqliteQueryResult(
                    columns=columns,
                    rows=row_dicts,
                    row_count=len(row_dicts),
                )

        except sqlite3.Error as exc:
            raise SqlExecutionError(
                message="SQLite execution error",
                error_code="sql_execution_error",
                details={
                    "db_path": str(self.db_path),
                    "query": final_query,
                },
                origin="sqlite_manager",
                cause=exc,
                http_status=400,
                is_retryable=False,
            ) from exc
        except Exception as exc:
            raise SqlExecutionError(
                message="Unexpected error during SQLite execution",
                error_code="internal_error",
                details={
                    "db_path": str(self.db_path),
                    "query": final_query,
                },
                origin="sqlite_manager",
                cause=exc,
                http_status=500,
                is_retryable=True,
            ) from exc

    @log_execution_time
    def execute(
        self,
        query: str,
        parameters: Optional[Tuple[Any, ...]] = None,
    ) -> None:
        """
            Execute a SQL statement (DDL/DML) without returning rows.

            Args:
                query: SQL statement
                parameters: Optional parameters tuple

            Returns:
                None
        """

        raw_query = (query or "").strip()
        if not raw_query:
            raise SqlExecutionError(
                message="Empty SQL query",
                error_code="validation_error",
                details={"query": query},
                origin="sqlite_manager",
                cause=None,
                http_status=400,
                is_retryable=False,
            )

        params = parameters or tuple()

        try:
            with self._connect() as conn:
                conn.execute(raw_query, params)
                conn.commit()

        except sqlite3.Error as exc:
            raise SqlExecutionError(
                message="SQLite execution error",
                error_code="sql_execution_error",
                details={"db_path": str(self.db_path), "query": raw_query},
                origin="sqlite_manager",
                cause=exc,
                http_status=400,
                is_retryable=False,
            ) from exc
        except Exception as exc:
            raise SqlExecutionError(
                message="Unexpected error during SQLite execution",
                error_code="internal_error",
                details={"db_path": str(self.db_path), "query": raw_query},
                origin="sqlite_manager",
                cause=exc,
                http_status=500,
                is_retryable=True,
            ) from exc

    @log_execution_time
    def query(
        self,
        query: str,
        limit: int = 100,
        parameters: Optional[Tuple[Any, ...]] = None,
    ) -> List[Dict[str, Any]]:
        """
            Execute a SELECT query and return rows as list of dicts.

            Args:
                query: SQL query
                limit: Max rows
                parameters: Optional parameters tuple

            Returns:
                List of rows as dict
        """

        result = self.execute_query(query=query, limit=limit, parameters=parameters)
        return result.rows
        
    def get_schema_overview(self) -> str:
        """
            Generate a compact SQLite schema overview for prompting

            High-level workflow:
                1) List tables
                2) For each table, list columns
                3) Return a readable schema string

            Returns:
                Schema overview string
        """

        ## ----------------------------------------------------
        ## Query schema metadata
        ## ----------------------------------------------------
        try:
            with self._connect() as conn:
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
                ).fetchall()

                if not tables:
                    return "No tables found in SQLite database"

                lines: List[str] = []
                for t in tables:
                    table_name = t["name"]
                    lines.append(f"Table: {table_name}")

                    cols = conn.execute(f"PRAGMA table_info({table_name});").fetchall()
                    for c in cols:
                        col_name = c["name"]
                        col_type = c["type"]
                        lines.append(f"  - {col_name}: {col_type}")

                return "\n".join(lines)

        except Exception as exc:
            raise SqlExecutionError(
                message="Failed to inspect SQLite schema",
                error_code="sql_execution_error",
                details={"db_path": str(self.db_path)},
                origin="sqlite_manager",
                cause=exc,
                http_status=500,
                is_retryable=True,
            ) from exc
            
## ============================================================
## BACKWARD-COMPAT ALIAS (tests expect SQLiteManager)
## ============================================================
SQLiteManager = SqliteManager  ## noqa: N816