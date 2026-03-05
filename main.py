'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main entry point: routes to Streamlit UI (streamlit run) or CLI pipelines (python main.py --...)."
'''

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from src.core.errors import AutonomousAIPlatformError
from src.pipeline import run_chat, run_evaluation, run_loop
from src.utils.logging_utils import get_logger
from src.utils.safe_utils import _safe_json, _safe_str
from src.utils.validation_utils import _must_be_non_empty

logger = get_logger(__name__)

## ============================================================
## STREAMLIT DETECTION
## ============================================================
## Detect if this script is executed via `streamlit run`
## This prevents importing Streamlit in CLI mode
def _is_streamlit_run() -> bool:
    """
        Detect if the script is executed via `streamlit run`.

        Returns:
            True if running under Streamlit, else False
    """

    return bool(
        os.environ.get("STREAMLIT_SERVER_PORT")
        or os.environ.get("STREAMLIT_SERVER_HEADLESS")
        or os.environ.get("STREAMLIT_RUNTIME")
    )

## ============================================================
## CLI ARGUMENT PARSER
## ============================================================
## Build the CLI interface used when running:
## python main.py --...
def _build_cli_parser() -> argparse.ArgumentParser:
    """
        Build CLI parser (no Streamlit).

        Returns:
            Configured ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description="Autonomous AI Platform CLI (no Streamlit).",
        add_help=True,
    )

    ## Pipeline execution
    parser.add_argument("--run-all", action="store_true", help="Run full pipeline: ingest -> loop example -> evaluation.")
    parser.add_argument("--run-api", action="store_true", help="Run MCP API server.")
    parser.add_argument("--ingest", action="store_true", help="Run ingestion pipeline (no UI).")

    ## LLM execution
    parser.add_argument("--chat", type=str, default="", help="Run one chat turn (no UI).")
    parser.add_argument("--loop", type=str, default="", help="Run autonomous loop (no UI).")

    ## Evaluation
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation (no UI).")
    parser.add_argument("--query", type=str, default="", help="Evaluation query (required with --evaluate).")
    parser.add_argument("--answer", type=str, default="", help="Evaluation answer (required with --evaluate).")

    ## Runtime options
    parser.add_argument("--prefer-local", action="store_true", help="Prefer local runtime.")
    parser.add_argument("--prefer-api", action="store_true", help="Prefer API runtime.")
    parser.add_argument("--use-gpu", choices=["auto", "true", "false"], default="auto", help="GPU usage.")
    parser.add_argument("--export", action="store_true", help="Export artifacts/reports.")

    return parser

## ============================================================
## GPU FLAG PARSER
## ============================================================
## Convert CLI GPU flag into Optional[bool]
def _parse_use_gpu(value: str) -> Optional[bool]:

    if value == "auto":
        return None
    if value == "true":
        return True
    return False

## ============================================================
## CLI EXECUTION
## ============================================================
## Execute CLI commands (no Streamlit)
def _run_cli() -> None:

    parser = _build_cli_parser()
    args = parser.parse_args()

    ## Runtime configuration
    prefer_local = True

    if bool(args.prefer_api):
        prefer_local = False

    if bool(args.prefer_local):
        prefer_local = True

    ## Parse GPU flag before any pipeline usage
    use_gpu = _parse_use_gpu(str(args.use_gpu))

    try:

        ## RUN FULL PIPELINE
        if bool(args.run_all):

            logger.info("Running full pipeline")

            import src.pipeline as pipeline_module

            ingest_fn = (
                getattr(pipeline_module, "run_ingest", None)
                or getattr(pipeline_module, "run_ingestion", None)
                or getattr(pipeline_module, "run_ingest_pipeline", None)
            )

            if ingest_fn:
                logger.info("Step 1/3 | ingestion")

                ingest_fn(
                    prefer_local=bool(prefer_local),
                    use_gpu=use_gpu,
                    export=bool(args.export),
                )

            logger.info("Step 2/3 | autonomous loop example")

            run_loop(
                "Example autonomous task",
                prefer_local=bool(prefer_local),
                use_gpu=use_gpu,
                export=bool(args.export),
            )

            logger.info("Step 3/3 | evaluation example")

            run_evaluation(
                "Example question",
                "Example answer",
                use_llm_judge=True,
                prefer_local=bool(prefer_local),
                use_gpu=use_gpu,
                export=bool(args.export),
            )

            logger.info("Full pipeline completed")

            return

        ## RUN API SERVER
        if bool(args.run_api):

            logger.info("Starting MCP API server")

            import uvicorn
            from src.core.mcp_server import app

            uvicorn.run(
                app,
                host="0.0.0.0",
                port=8000,
                log_level="info",
            )

            return

        ## INGEST
        if bool(args.ingest):

            import src.pipeline as pipeline_module

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

            result = ingest_fn(
                prefer_local=bool(prefer_local),
                use_gpu=use_gpu,
                export=bool(args.export),
            )

            logger.info("CLI ingest completed | result=%s", _safe_json(result))

            return

        ## CHAT
        if str(args.chat).strip():

            prompt = _must_be_non_empty(str(args.chat), "chat")

            result = run_chat(
                prompt,
                prefer_local=bool(prefer_local),
                use_gpu=use_gpu,
            )

            logger.info(
                "CLI chat completed | provider=%s | model=%s | result=%s",
                result.get("provider"),
                result.get("model"),
                _safe_json(result),
            )

            return

        ## LOOP
        if str(args.loop).strip():

            q = _must_be_non_empty(str(args.loop), "loop")

            result = run_loop(
                q,
                prefer_local=bool(prefer_local),
                use_gpu=use_gpu,
                export=bool(args.export),
            )

            logger.info("CLI loop completed | result=%s", _safe_json(result))

            return

        ## EVALUATE
        if bool(args.evaluate):

            q = _must_be_non_empty(str(args.query), "query")
            a = _must_be_non_empty(str(args.answer), "answer")

            report = run_evaluation(
                q,
                a,
                use_llm_judge=True,
                prefer_local=bool(prefer_local),
                use_gpu=use_gpu,
                export=bool(args.export),
            )

            logger.info("CLI evaluation completed | report=%s", _safe_json(report))

            return

        ## If no argument is provided show help
        parser.print_help()

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

            sys.exit(2)

        logger.error(
            "CLI UnhandledException | type=%s | message=%s",
            exc.__class__.__name__,
            _safe_str(exc),
        )

        sys.exit(2)

## ============================================================
## MAIN ENTRY POINT
## ============================================================
def main() -> None:
    """
        Entry point.

        - `streamlit run main.py` -> launches Streamlit UI
        - `python main.py --...` -> runs CLI pipelines

        Returns:
            None
    """

    ## If executed by Streamlit runtime
    if _is_streamlit_run():

        ## Import Streamlit UI lazily to avoid CLI warnings
        from src.core.streamlit_app import run_streamlit_app

        run_streamlit_app()

        return

    ## Otherwise run CLI
    _run_cli()


if __name__ == "__main__":
    main()