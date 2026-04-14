'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main entry point: routes to Streamlit UI or CLI pipelines with runtime validation and standardized execution flow."
'''

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

from src.core.data_consistency import run_data_consistency
from src.core.errors import AutonomousAIPlatformError
from src.pipeline import run_chat, run_evaluation, run_loop, pipeline_module
from src.utils.logging_utils import get_logger
from src.utils.safe_utils import _safe_json, _safe_str
from src.utils.validation_utils import _must_be_non_empty

logger = get_logger(__name__)

APP_VERSION = "1.0.0"
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_PLATFORM_ERROR = 2

## ============================================================
## STREAMLIT DETECTION
## ============================================================
def _is_streamlit_run() -> bool:
    """
        Detect whether the script is executed by Streamlit.

        Returns:
            True if the current process is running under Streamlit
            False otherwise
    """

    return bool(
        os.environ.get("STREAMLIT_SERVER_PORT")
        or os.environ.get("STREAMLIT_SERVER_HEADLESS")
        or os.environ.get("STREAMLIT_RUNTIME")
    )

## ============================================================
## CONFIG VALIDATION
## ============================================================
def _validate_runtime_config() -> dict:
    """
        Validate minimal runtime configuration required by the entry point.

        Validation scope:
            - Ensure the application can start safely
            - Keep validation lightweight at main entry level
            - Leave deep business validation to downstream modules

        Returns:
            A lightweight validation summary dictionary

        Raises:
            RuntimeError: If a critical runtime issue is detected
    """

    ## Keep validation intentionally lightweight for the main entry point
    summary = {
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
        "streamlit_mode": _is_streamlit_run(),
    }

    if not summary["python_executable"]:
        raise RuntimeError("Python executable could not be resolved.")

    return summary

## ============================================================
## GPU FLAG PARSER
## ============================================================
def _parse_use_gpu(value: str) -> Optional[bool]:
    """
        Convert CLI GPU flag into an Optional boolean.

        Args:
            value: Raw CLI value among auto, true, false

        Returns:
            None for auto
            True for true
            False for false
    """

    normalized = str(value).strip().lower()

    if normalized == "auto":
        return None
    if normalized == "true":
        return True
    return False

## ============================================================
## CLI ARGUMENT PARSER
## ============================================================
def _build_cli_parser() -> argparse.ArgumentParser:
    """
        Build the CLI parser for non-Streamlit execution.

        Returns:
            A configured ArgumentParser instance
    """

    parser = argparse.ArgumentParser(
        description="Autonomous AI Platform CLI (non-Streamlit mode).",
        add_help=True,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {APP_VERSION}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and log intended actions without executing them.",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate runtime configuration and exit.",
    )

    ## Pipeline execution
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run full pipeline: ingest -> loop example -> evaluation.",
    )
    parser.add_argument(
        "--run-api",
        action="store_true",
        help="Run MCP API server.",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run ingestion pipeline without UI.",
    )

    ## LLM execution
    parser.add_argument(
        "--chat",
        type=str,
        default="",
        help="Run one chat turn without UI.",
    )
    parser.add_argument(
        "--loop",
        type=str,
        default="",
        help="Run autonomous loop without UI.",
    )

    ## Evaluation
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation without UI.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Evaluation query required with --evaluate.",
    )
    parser.add_argument(
        "--answer",
        type=str,
        default="",
        help="Evaluation answer required with --evaluate.",
    )

    ## Runtime options
    parser.add_argument(
        "--prefer-local",
        action="store_true",
        help="Prefer local runtime.",
    )
    parser.add_argument(
        "--prefer-api",
        action="store_true",
        help="Prefer API runtime.",
    )
    parser.add_argument(
        "--use-gpu",
        choices=["auto", "true", "false"],
        default="auto",
        help="GPU usage policy.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export artifacts or reports.",
    )

    return parser

## ============================================================
## HELPER FUNCTIONS
## ============================================================
def _resolve_prefer_local(args: argparse.Namespace) -> bool:
    """
        Resolve runtime preference between local and API execution.

        Args:
            args: Parsed CLI arguments

        Returns:
            True when local runtime should be preferred
            False when API runtime should be preferred
    """

    prefer_local = True

    ## Keep local as default and allow explicit override
    if bool(args.prefer_api):
        prefer_local = False
    if bool(args.prefer_local):
        prefer_local = True

    return prefer_local

def _get_ingest_function():
    """
        Resolve the ingestion entry point dynamically from src.pipeline.

        Returns:
            The ingestion callable if found

        Raises:
            RuntimeError: If no supported ingestion entry point exists
    """

    ingest_fn = (
        getattr(pipeline_module, "run_ingest", None)
        or getattr(pipeline_module, "run_ingestion", None)
        or getattr(pipeline_module, "run_ingest_pipeline", None)
    )

    if ingest_fn is None:
        raise RuntimeError(
            "Ingestion entrypoint not found in src.pipeline "
            "(expected: run_ingest | run_ingestion | run_ingest_pipeline)."
        )

    return ingest_fn

def _build_execution_summary(
    action: str,
    success: bool,
    started_at: float,
    details: Optional[dict] = None,
) -> dict:
    """
        Build a standardized execution summary.

        Args:
            action: Logical action that was executed
            success: Whether the execution completed successfully
            started_at: Monotonic start timestamp
            details: Optional structured details

        Returns:
            A summary dictionary suitable for logging
    """

    duration_seconds = round(time.monotonic() - started_at, 3)

    return {
        "action": action,
        "success": success,
        "duration_seconds": duration_seconds,
        "details": details or {},
    }

## ============================================================
## CLI EXECUTION
## ============================================================
def _run_cli() -> int:
    """
        Execute CLI commands in non-Streamlit mode.

        Returns:
            Standardized process exit code
    """

    parser = _build_cli_parser()
    args = parser.parse_args()
    started_at = time.monotonic()

    try:
        ## Validate runtime as early as possible
        validation_summary = _validate_runtime_config()

        if bool(args.validate_config):
            logger.info(
                "Runtime configuration validation succeeded | summary=%s",
                _safe_json(validation_summary),
            )
            summary = _build_execution_summary(
                action="validate-config",
                success=True,
                started_at=started_at,
                details=validation_summary,
            )
            logger.info("Execution summary | %s", _safe_json(summary))
            return EXIT_SUCCESS

        prefer_local = _resolve_prefer_local(args)
        use_gpu = _parse_use_gpu(str(args.use_gpu))

        runtime_details = {
            "prefer_local": prefer_local,
            "use_gpu": use_gpu,
            "export": bool(args.export),
            "dry_run": bool(args.dry_run),
        }

        logger.info("CLI runtime resolved | %s", _safe_json(runtime_details))

        ## ============================================================
        ## DATA CONSISTENCY CHECK (AUTONOMOUS PIPELINE)
        ## ============================================================
        consistency_result = run_data_consistency(
            data={
                "tasks": [
                    {
                        "id": "task_1",
                        "agent": "default",
                        "prompt": str(args.chat or args.loop or args.query or ""),
                        "depends_on": [],
                    }
                ],
                "agents": {
                    "default": {
                        "model": "default-model"
                    }
                },
            },
            strict=False,
        )

        logger.info("Consistency OK | %s", consistency_result["is_consistent"])
        ## RUN FULL PIPELINE
        if bool(args.run_all):
            if bool(args.dry_run):
                logger.info("Dry-run | full pipeline would be executed")
                summary = _build_execution_summary(
                    action="run-all",
                    success=True,
                    started_at=started_at,
                    details=runtime_details,
                )
                logger.info("Execution summary | %s", _safe_json(summary))
                return EXIT_SUCCESS

            logger.info("Running full pipeline")

            ingest_fn = _get_ingest_function()

            logger.info("Step 1/3 | ingestion")
            ingest_fn(
                prefer_local=prefer_local,
                use_gpu=use_gpu,
                export=bool(args.export),
            )

            logger.info("Step 2/3 | autonomous loop example")
            run_loop(
                "Example autonomous task",
                prefer_local=prefer_local,
                use_gpu=use_gpu,
                export=bool(args.export),
            )

            logger.info("Step 3/3 | evaluation example")
            run_evaluation(
                "Example question",
                "Example answer",
                use_llm_judge=True,
                prefer_local=prefer_local,
                use_gpu=use_gpu,
                export=bool(args.export),
            )

            summary = _build_execution_summary(
                action="run-all",
                success=True,
                started_at=started_at,
                details=runtime_details,
            )
            logger.info("Full pipeline completed")
            logger.info("Execution summary | %s", _safe_json(summary))
            return EXIT_SUCCESS

        ## RUN API SERVER
        if bool(args.run_api):
            if bool(args.dry_run):
                logger.info("Dry-run | MCP API server would be started")
                summary = _build_execution_summary(
                    action="run-api",
                    success=True,
                    started_at=started_at,
                    details=runtime_details,
                )
                logger.info("Execution summary | %s", _safe_json(summary))
                return EXIT_SUCCESS

            logger.info("Starting MCP API server")

            import uvicorn
            from src.core.mcp_server import app

            uvicorn.run(
                app,
                host="0.0.0.0",
                port=8000,
                log_level="info",
            )
            return EXIT_SUCCESS

        ## INGEST
        if bool(args.ingest):
            if bool(args.dry_run):
                logger.info("Dry-run | ingestion pipeline would be executed")
                summary = _build_execution_summary(
                    action="ingest",
                    success=True,
                    started_at=started_at,
                    details=runtime_details,
                )
                logger.info("Execution summary | %s", _safe_json(summary))
                return EXIT_SUCCESS

            ingest_fn = _get_ingest_function()
            result = ingest_fn(
                prefer_local=prefer_local,
                use_gpu=use_gpu,
                export=bool(args.export),
            )

            logger.info("CLI ingest completed | result=%s", _safe_json(result))
            summary = _build_execution_summary(
                action="ingest",
                success=True,
                started_at=started_at,
                details={"result": result, **runtime_details},
            )
            logger.info("Execution summary | %s", _safe_json(summary))
            return EXIT_SUCCESS

        ## CHAT
        if str(args.chat).strip():
            prompt = _must_be_non_empty(str(args.chat), "chat")

            if bool(args.dry_run):
                logger.info("Dry-run | chat would be executed | prompt=%s", prompt)
                summary = _build_execution_summary(
                    action="chat",
                    success=True,
                    started_at=started_at,
                    details={"prompt": prompt, **runtime_details},
                )
                logger.info("Execution summary | %s", _safe_json(summary))
                return EXIT_SUCCESS

            result = run_chat(
                prompt,
                prefer_local=prefer_local,
                use_gpu=use_gpu,
            )

            logger.info(
                "CLI chat completed | provider=%s | model=%s | result=%s",
                result.get("provider"),
                result.get("model"),
                _safe_json(result),
            )
            summary = _build_execution_summary(
                action="chat",
                success=True,
                started_at=started_at,
                details={"result": result, **runtime_details},
            )
            logger.info("Execution summary | %s", _safe_json(summary))
            return EXIT_SUCCESS

        ## LOOP
        if str(args.loop).strip():
            query = _must_be_non_empty(str(args.loop), "loop")

            if bool(args.dry_run):
                logger.info("Dry-run | loop would be executed | query=%s", query)
                summary = _build_execution_summary(
                    action="loop",
                    success=True,
                    started_at=started_at,
                    details={"query": query, **runtime_details},
                )
                logger.info("Execution summary | %s", _safe_json(summary))
                return EXIT_SUCCESS

            result = run_loop(
                query,
                prefer_local=prefer_local,
                use_gpu=use_gpu,
                export=bool(args.export),
            )

            logger.info("CLI loop completed | result=%s", _safe_json(result))
            summary = _build_execution_summary(
                action="loop",
                success=True,
                started_at=started_at,
                details={"result": result, **runtime_details},
            )
            logger.info("Execution summary | %s", _safe_json(summary))
            return EXIT_SUCCESS

        ## EVALUATE
        if bool(args.evaluate):
            query = _must_be_non_empty(str(args.query), "query")
            answer = _must_be_non_empty(str(args.answer), "answer")

            if bool(args.dry_run):
                logger.info(
                    "Dry-run | evaluation would be executed | query=%s",
                    query,
                )
                summary = _build_execution_summary(
                    action="evaluate",
                    success=True,
                    started_at=started_at,
                    details={"query": query, "answer": answer, **runtime_details},
                )
                logger.info("Execution summary | %s", _safe_json(summary))
                return EXIT_SUCCESS

            report = run_evaluation(
                query,
                answer,
                use_llm_judge=True,
                prefer_local=prefer_local,
                use_gpu=use_gpu,
                export=bool(args.export),
            )

            logger.info("CLI evaluation completed | report=%s", _safe_json(report))
            summary = _build_execution_summary(
                action="evaluate",
                success=True,
                started_at=started_at,
                details={"report": report, **runtime_details},
            )
            logger.info("Execution summary | %s", _safe_json(summary))
            return EXIT_SUCCESS

        ## Default behavior when no actionable flag is provided
        parser.print_help()
        summary = _build_execution_summary(
            action="help",
            success=True,
            started_at=started_at,
            details=runtime_details,
        )
        logger.info("Execution summary | %s", _safe_json(summary))
        return EXIT_SUCCESS

    except KeyboardInterrupt:
        summary = _build_execution_summary(
            action="interrupted",
            success=False,
            started_at=started_at,
        )
        logger.warning("Execution interrupted by user")
        logger.warning("Execution summary | %s", _safe_json(summary))
        return EXIT_FAILURE

    except Exception as exc:
        if isinstance(exc, AutonomousAIPlatformError):
            payload = exc.to_payload()

            logger.error(
                "CLI PlatformError | code=%s | origin=%s | message=%s | details=%s",
                payload.error_code,
                payload.origin,
                payload.message,
                _safe_json(payload.details),
            )

            summary = _build_execution_summary(
                action="platform-error",
                success=False,
                started_at=started_at,
                details={
                    "error_code": payload.error_code,
                    "origin": payload.origin,
                    "message": payload.message,
                },
            )
            logger.error("Execution summary | %s", _safe_json(summary))
            return EXIT_PLATFORM_ERROR

        logger.error(
            "CLI UnhandledException | type=%s | message=%s",
            exc.__class__.__name__,
            _safe_str(exc),
        )

        summary = _build_execution_summary(
            action="unhandled-exception",
            success=False,
            started_at=started_at,
            details={
                "exception_type": exc.__class__.__name__,
                "message": _safe_str(exc),
            },
        )
        logger.error("Execution summary | %s", _safe_json(summary))
        return EXIT_FAILURE

## ============================================================
## MAIN ENTRY POINT
## ============================================================
def main() -> int:
    """
        Run the main application entry point.

        Execution modes:
            - streamlit run main.py
            - python main.py --...

        Returns:
            Standardized process exit code
    """

    ## Route to Streamlit UI when executed by Streamlit runtime
    if _is_streamlit_run():
        from src.core.streamlit_app import run_streamlit_app

        run_streamlit_app()
        return EXIT_SUCCESS

    ## Otherwise use standard CLI flow
    return _run_cli()

if __name__ == "__main__":
    sys.exit(main())