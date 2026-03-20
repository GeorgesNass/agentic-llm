'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Prod"
__desc__ = "Main entrypoint for rag-drive-gcp: configuration validation, ingestion, RAG query and optional Streamlit launcher."
'''

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from src.model.settings import get_settings
from src.pipelines import run_drive_ingestion_pipeline, run_rag_query_pipeline
from src.utils.logging_utils import get_logger

## ============================================================
## CONSTANTS
## ============================================================
APP_VERSION = "1.0.0"
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_PLATFORM_ERROR = 2

## ============================================================
## LOGGER INITIALIZATION
## ============================================================
logger = get_logger("main")

## ============================================================
## CLI ARGUMENTS
## ============================================================
def _build_parser() -> argparse.ArgumentParser:
    """
        Build CLI parser for rag-drive-gcp

        Returns:
            Configured ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description="rag-drive-gcp launcher: validate config, ingest Drive files, run RAG query, or launch Streamlit UI.",
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
        help="Validate arguments and log intended action without executing it.",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate settings loading and exit.",
    )

    ## Execution modes
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run Google Drive ingestion pipeline.",
    )
    parser.add_argument(
        "--query",
        action="store_true",
        help="Run one RAG query from CLI.",
    )
    parser.add_argument(
        "--run-ui",
        action="store_true",
        help="Launch the dedicated Streamlit UI file.",
    )

    ## Shared inputs
    parser.add_argument(
        "--folder-id",
        type=str,
        default="",
        help="Google Drive folder ID override for ingestion.",
    )
    parser.add_argument(
        "--run-ocr",
        action="store_true",
        help="Enable OCR during ingestion.",
    )
    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="Keep local files after ingestion.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-K chunks for retrieval. If 0, use settings value.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Chunk size override. If 0, use settings value.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=0,
        help="Chunk overlap override. If 0, use settings value.",
    )

    ## Query input
    parser.add_argument(
        "--question",
        type=str,
        default="",
        help="Question to ask through the RAG pipeline.",
    )

    ## Optional UI launcher path
    parser.add_argument(
        "--ui-file",
        type=str,
        default="",
        help="Optional path to the dedicated Streamlit UI file.",
    )

    return parser

## ============================================================
## HELPERS
## ============================================================
def _build_summary(
    action: str,
    success: bool,
    start: float,
    details: Optional[dict] = None,
) -> dict:
    """
        Build standardized execution summary

        Args:
            action: Executed action name
            success: Execution status
            start: Monotonic start timestamp
            details: Optional structured details

        Returns:
            Standardized summary dictionary
    """

    return {
        "action": action,
        "success": success,
        "duration_seconds": round(time.monotonic() - start, 3),
        "details": details or {},
    }

def _resolve_runtime_params(settings, args: argparse.Namespace) -> dict:
    """
        Resolve effective runtime parameters from settings and CLI overrides

        Args:
            settings: Application settings object
            args: Parsed CLI arguments

        Returns:
            Effective runtime parameters
    """

    return {
        "folder_id": args.folder_id.strip() or settings.drive_folder_id or "",
        "run_ocr": bool(args.run_ocr),
        "keep_local": bool(args.keep_local) if args.keep_local else bool(settings.keep_local),
        "top_k": int(args.top_k) if int(args.top_k) > 0 else int(settings.top_k),
        "chunk_size": (
            int(args.chunk_size) if int(args.chunk_size) > 0 else int(settings.chunk_size)
        ),
        "chunk_overlap": (
            int(args.chunk_overlap)
            if int(args.chunk_overlap) > 0
            else int(settings.chunk_overlap)
        ),
    }

def _validate_action_selection(args: argparse.Namespace) -> None:
    """
        Validate CLI action selection

        Args:
            args: Parsed CLI arguments

        Raises:
            ValueError: If action selection is invalid
    """

    selected_actions = [bool(args.ingest), bool(args.query), bool(args.run_ui)]

    if sum(selected_actions) > 1:
        raise ValueError("Use only one action at a time: --ingest, --query, or --run-ui.")

    if args.query and not args.question.strip():
        raise ValueError("--question is required with --query.")

def _launch_streamlit_ui(ui_file: str) -> None:
    """
        Launch the dedicated Streamlit UI file

        Args:
            ui_file: Path to the Streamlit entry file

        Returns:
            None

        Raises:
            FileNotFoundError: If the UI file does not exist
            RuntimeError: If Streamlit launcher fails
    """

    if not ui_file.strip():
        raise ValueError(
            "--ui-file is required with --run-ui because the Streamlit UI lives in a dedicated file."
        )

    ui_path = Path(ui_file).expanduser().resolve()

    if not ui_path.exists():
        raise FileNotFoundError(f"Streamlit UI file not found: {ui_path}")

    cmd = ["streamlit", "run", str(ui_path)]

    logger.info("Launching Streamlit UI | file=%s", ui_path)

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("streamlit command is not available in the current environment.") from exc

## ============================================================
## MAIN EXECUTION
## ============================================================
def main() -> int:
    """
        Main CLI entrypoint for rag-drive-gcp

        Supported actions:
            - validate configuration
            - run ingestion pipeline
            - run one RAG query from CLI
            - launch external Streamlit UI file

        Returns:
            Standardized process exit code
    """

    start_time = time.monotonic()
    parser = _build_parser()
    args = parser.parse_args()

    try:
        settings = get_settings()
        runtime_params = _resolve_runtime_params(settings, args)

        if args.validate_config:
            logger.info(
                "Configuration validation succeeded | drive_folder_id=%s | top_k=%s | chunk_size=%s | chunk_overlap=%s",
                runtime_params["folder_id"],
                runtime_params["top_k"],
                runtime_params["chunk_size"],
                runtime_params["chunk_overlap"],
            )
            logger.info(
                "Summary | %s",
                _build_summary(
                    action="validate-config",
                    success=True,
                    start=start_time,
                    details=runtime_params,
                ),
            )
            return EXIT_SUCCESS

        _validate_action_selection(args)

        if not any([args.ingest, args.query, args.run_ui]):
            parser.print_help()
            logger.info(
                "Summary | %s",
                _build_summary(
                    action="help",
                    success=True,
                    start=start_time,
                ),
            )
            return EXIT_SUCCESS

        if args.dry_run:
            logger.info(
                "Dry-run | ingest=%s | query=%s | run_ui=%s | params=%s",
                bool(args.ingest),
                bool(args.query),
                bool(args.run_ui),
                runtime_params,
            )
            logger.info(
                "Summary | %s",
                _build_summary(
                    action="dry-run",
                    success=True,
                    start=start_time,
                    details=runtime_params,
                ),
            )
            return EXIT_SUCCESS

        ## INGESTION
        if args.ingest:
            if not runtime_params["folder_id"]:
                raise ValueError("A Google Drive folder ID is required for --ingest.")

            logger.info("Starting ingestion pipeline from CLI")

            status = run_drive_ingestion_pipeline(
                drive_folder_id=runtime_params["folder_id"],
                run_ocr=runtime_params["run_ocr"],
                chunk_size=runtime_params["chunk_size"],
                chunk_overlap=runtime_params["chunk_overlap"],
                keep_local=runtime_params["keep_local"],
            )

            logger.info("Ingestion completed successfully | status=%s", status)
            logger.info(
                "Summary | %s",
                _build_summary(
                    action="ingestion",
                    success=True,
                    start=start_time,
                    details=runtime_params,
                ),
            )
            return EXIT_SUCCESS

        ## RAG QUERY
        if args.query:
            logger.info("Starting RAG CLI query")

            answer = run_rag_query_pipeline(
                question=args.question.strip(),
                top_k=runtime_params["top_k"],
            )

            logger.info("RAG query completed successfully")
            print(answer)

            logger.info(
                "Summary | %s",
                _build_summary(
                    action="rag-query",
                    success=True,
                    start=start_time,
                    details={
                        "top_k": runtime_params["top_k"],
                        "question_length": len(args.question.strip()),
                    },
                ),
            )
            return EXIT_SUCCESS

        ## STREAMLIT LAUNCHER
        if args.run_ui:
            _launch_streamlit_ui(args.ui_file)

            logger.info(
                "Summary | %s",
                _build_summary(
                    action="run-ui",
                    success=True,
                    start=start_time,
                    details={"ui_file": args.ui_file},
                ),
            )
            return EXIT_SUCCESS

        logger.info(
            "Summary | %s",
            _build_summary(
                action="run",
                success=True,
                start=start_time,
            ),
        )
        return EXIT_SUCCESS

    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user")
        logger.warning(
            "Summary | %s",
            _build_summary(
                action="interrupt",
                success=False,
                start=start_time,
            ),
        )
        return EXIT_FAILURE

    except Exception as exc:
        logger.exception("Unhandled error in rag-drive-gcp main")
        logger.error(
            "Summary | %s",
            _build_summary(
                action="unhandled-error",
                success=False,
                start=start_time,
                details={"error": str(exc)},
            ),
        )
        return EXIT_PLATFORM_ERROR

## ============================================================
## SCRIPT ENTRY POINT
## ============================================================
if __name__ == "__main__":
    sys.exit(main())