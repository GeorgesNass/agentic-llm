'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main CLI entry point for llm-proxy-gateway: cost simulation, pipeline execution, evaluation and API service."
'''

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import uvicorn

from src.core.data_consistency import run_data_consistency
from src.core.config import settings
from src.core.errors import (
    ConfigurationError,
    DataError,
    PipelineError,
    ProviderError,
    ValidationError,
)
from src.llm.costing import load_catalogs, simulate_cost
from src.llm.evaluation import evaluate_batch
from src.pipeline import run_full_pipeline
from src.utils.logging_utils import get_logger
from src.utils.utils import _load_lines, _load_messages, _parse_providers

## ============================================================
## CONSTANTS
## ============================================================
APP_VERSION = "1.0.0"
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_PLATFORM_ERROR = 2

## ============================================================
## LOGGER
## ============================================================
LOGGER = get_logger(__name__)

## ============================================================
## CLI ARGUMENTS
## ============================================================
def _build_parser() -> argparse.ArgumentParser:
    """
        Build argument parser for CLI usage

        High-level workflow:
            1) Define actions
            2) Define common parameters
            3) Return configured parser

        Returns:
            Configured ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description=(
            "llm-proxy-gateway (cost simulation, embeddings, "
            "chat completion, evaluation, and serve API)."
        ),
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
        help="Validate arguments and resolved resources without executing actions.",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate catalogs, settings and argument structure, then exit.",
    )

    ## Main action flags
    parser.add_argument(
        "--cost",
        action="store_true",
        help="Run cost simulation only (chat or embeddings).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run pipeline execution after cost simulation.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation metrics on predictions vs references.",
    )
    parser.add_argument(
        "--run-api",
        action="store_true",
        help="Run FastAPI service (uvicorn).",
    )

    ## Providers / models / mode
    parser.add_argument(
        "--mode",
        type=str,
        default="chat",
        help="Mode: chat | embeddings (default: chat).",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default="openai",
        help="Comma-separated providers (default: openai).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name override (optional).",
    )

    ## Inputs
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="Raw text input (mutually exclusive with --path).",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Path to a folder with .txt files (mutually exclusive with --text).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursive folder scan (default: false).",
    )

    ## Chat messages
    parser.add_argument(
        "--messages-json",
        type=str,
        default="",
        help="Path to JSON file containing OpenAI-like messages list for chat.",
    )

    ## Cost / chunking parameters
    parser.add_argument(
        "--expected-output-tokens",
        type=int,
        default=512,
        help="Assumed output tokens for chat cost simulation (default: 512).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters for embeddings simulation (default: 1000).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters for embeddings simulation (default: 200).",
    )
    parser.add_argument(
        "--max-chars-per-file",
        type=int,
        default=200_000,
        help="Max chars to read per file (default: 200000).",
    )

    ## API options
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API host (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API port (default: 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (dev mode).",
    )

    ## Catalog paths
    parser.add_argument(
        "--models-catalog",
        type=str,
        default=str(Path("artifacts") / "resources" / "models_catalog.json"),
        help="Path to models_catalog.json (default: artifacts/resources/models_catalog.json).",
    )
    parser.add_argument(
        "--pricing-catalog",
        type=str,
        default=str(Path("artifacts") / "resources" / "pricing_catalog.json"),
        help="Path to pricing_catalog.json (default: artifacts/resources/pricing_catalog.json).",
    )

    ## Evaluation inputs
    parser.add_argument(
        "--predictions-path",
        type=str,
        default="",
        help="Path to a .txt file containing predictions (one per line).",
    )
    parser.add_argument(
        "--references-path",
        type=str,
        default="",
        help="Path to a .txt file containing references (one per line).",
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

def _resolve_requested_model(model: str) -> Optional[str]:
    """
        Normalize requested model value

        Args:
            model: Raw model CLI value

        Returns:
            Clean model value or None
    """

    return str(model).strip() if str(model).strip() else None

def _validate_cli_consistency(args: argparse.Namespace) -> None:
    """
        Validate CLI argument consistency

        Args:
            args: Parsed CLI arguments

        Raises:
            ValueError: If arguments are inconsistent
    """

    ## At least one main action is required outside help/version
    if not any([args.cost, args.run, args.evaluate, args.run_api]):
        return

    ## Keep original semantics while adding lightweight validation
    if str(args.text).strip() and str(args.path).strip():
        raise ValueError("--text and --path are mutually exclusive")

    if args.evaluate:
        if not str(args.predictions_path).strip() or not str(args.references_path).strip():
            raise ValueError("--predictions-path and --references-path are required for --evaluate")

def _load_catalogs_for_cli(
    models_catalog_path: str,
    pricing_catalog_path: str,
) -> tuple[dict, dict]:
    """
        Load pricing and model catalogs for CLI operations

        Args:
            models_catalog_path: Path to models catalog JSON
            pricing_catalog_path: Path to pricing catalog JSON

        Returns:
            Tuple containing models catalog and pricing catalog
    """

    return load_catalogs(
        models_catalog_path=Path(models_catalog_path),
        pricing_catalog_path=Path(pricing_catalog_path),
    )

## ============================================================
## MAIN EXECUTION
## ============================================================
def main() -> int:
    """
        Main CLI entry point

        Workflow notes:
            - --cost runs cost simulation only
            - --run runs cost simulation then execution via pipeline.py
            - --evaluate runs metrics on predictions vs references
            - --run-api starts FastAPI server via uvicorn

        Returns:
            Standardized process exit code
    """

    start_time = time.monotonic()

    try:
        parser = _build_parser()
        args = parser.parse_args()

        if not any([args.cost, args.run, args.evaluate, args.run_api]):
            parser.print_help()
            LOGGER.info(
                "Summary | %s",
                json.dumps(
                    _build_summary("help", True, start_time),
                    ensure_ascii=False,
                ),
            )
            return EXIT_SUCCESS

        _validate_cli_consistency(args)

        providers = _parse_providers(str(args.providers))
        requested_model = _resolve_requested_model(args.model)

        messages = None
        if str(args.messages_json).strip():
            messages = _load_messages(str(args.messages_json).strip())
            
        ## ============================================================
        ## DATA CONSISTENCY CHECK (LLM REQUEST)
        ## ============================================================
        if settings.data_consistency.enabled:
            consistency_result = run_data_consistency(
                data={
                    "prompt": args.text if args.text else None,
                    "messages": messages,
                    "model": requested_model or providers[0],
                    "max_tokens": args.expected_output_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
                strict=settings.data_consistency.strict_mode,
            )

            LOGGER.info(f"Consistency OK: {consistency_result['is_consistent']}")

            if not consistency_result["is_consistent"] and settings.data_consistency.strict_mode:
                raise DataError(
                    message="Data consistency failed before LLM request",
                    error_code="data_consistency_error",
                    details={"issues": consistency_result["issues"]},
                    origin="main",
                    http_status=400,
                    is_retryable=False,
                )
                
        ## Load catalogs once for CLI usage
        models_catalog, pricing_catalog = _load_catalogs_for_cli(
            models_catalog_path=args.models_catalog,
            pricing_catalog_path=args.pricing_catalog,
        )

        if args.validate_config:
            LOGGER.info(
                "Configuration validation completed | providers=%s | mode=%s",
                ",".join(providers),
                str(args.mode),
            )
            LOGGER.info(
                "Summary | %s",
                json.dumps(
                    _build_summary(
                        "validate-config",
                        True,
                        start_time,
                        {
                            "models_catalog": args.models_catalog,
                            "pricing_catalog": args.pricing_catalog,
                            "debug": bool(getattr(settings, "debug", False)),
                        },
                    ),
                    ensure_ascii=False,
                ),
            )
            return EXIT_SUCCESS

        if args.dry_run:
            LOGGER.info(
                "Dry-run | cost=%s run=%s evaluate=%s run_api=%s",
                bool(args.cost),
                bool(args.run),
                bool(args.evaluate),
                bool(args.run_api),
            )
            LOGGER.info(
                "Summary | %s",
                json.dumps(
                    _build_summary(
                        "dry-run",
                        True,
                        start_time,
                        {
                            "mode": str(args.mode),
                            "providers": providers,
                            "model": requested_model,
                        },
                    ),
                    ensure_ascii=False,
                ),
            )
            return EXIT_SUCCESS

        ## COST ONLY
        if args.cost and not args.run:
            LOGGER.info("Running cost simulation only | mode=%s", args.mode)

            result = simulate_cost(
                mode=str(args.mode),
                providers=providers,
                requested_model=requested_model,
                text=str(args.text) if str(args.text).strip() else None,
                path=str(args.path) if str(args.path).strip() else None,
                recursive=bool(args.recursive),
                max_chars_per_file=int(args.max_chars_per_file),
                expected_output_tokens=int(args.expected_output_tokens),
                chunk_size=int(args.chunk_size),
                chunk_overlap=int(args.chunk_overlap),
                models_catalog=models_catalog,
                pricing_catalog=pricing_catalog,
                include_per_file=True,
            )

            LOGGER.info(
                "Cost simulation completed | payload=%s",
                json.dumps(result, ensure_ascii=False)[:2000],
            )

        ## RUN PIPELINE
        if args.run:
            LOGGER.info(
                "Running full pipeline | mode=%s providers=%s",
                args.mode,
                ",".join(providers),
            )

            messages: Optional[list[dict[str, Any]]] = None
            if str(args.messages_json).strip():
                messages = _load_messages(str(args.messages_json).strip())

            result = run_full_pipeline(
                mode=str(args.mode),
                providers=providers,
                model=requested_model,
                messages=messages,
                text=str(args.text) if str(args.text).strip() else None,
                cost_only=False,
                models_catalog=models_catalog,
                pricing_catalog=pricing_catalog,
                temperature=0.0,
                max_tokens=512,
                top_p=1.0,
                chunk_size=int(args.chunk_size),
                chunk_overlap=int(args.chunk_overlap),
                expected_output_tokens=int(args.expected_output_tokens),
                export_path=None,
            )

            LOGGER.info(
                "Pipeline run completed | payload=%s",
                json.dumps(result, ensure_ascii=False)[:2000],
            )

        ## EVALUATION
        if args.evaluate:
            predictions = _load_lines(str(args.predictions_path).strip())
            references = _load_lines(str(args.references_path).strip())

            if len(predictions) != len(references):
                raise ValueError("predictions and references must have same number of lines")

            LOGGER.info("Running evaluation | n=%d", len(predictions))

            metrics = evaluate_batch(
                predictions=predictions,
                references=references,
                enable_rouge_bleu_bertscore=False,
            )

            LOGGER.info(
                "Evaluation completed | payload=%s",
                json.dumps(metrics, ensure_ascii=False)[:2000],
            )

        ## RUN API
        if args.run_api:
            reload_mode = bool(args.reload) or bool(getattr(settings, "debug", False))

            LOGGER.info(
                "Starting API server | host=%s port=%d reload=%s",
                args.host,
                int(args.port),
                reload_mode,
            )

            uvicorn.run(
                "src.core.service:app",
                host=str(args.host),
                port=int(args.port),
                reload=reload_mode,
            )

        LOGGER.info(
            "Summary | %s",
            json.dumps(_build_summary("run", True, start_time), ensure_ascii=False),
        )
        return EXIT_SUCCESS

    except KeyboardInterrupt:
        LOGGER.warning("Execution interrupted by user")
        LOGGER.warning(
            "Summary | %s",
            json.dumps(_build_summary("interrupt", False, start_time), ensure_ascii=False),
        )
        return EXIT_FAILURE

    except (ConfigurationError, ValidationError, ProviderError, PipelineError, DataError) as exc:
        LOGGER.error("Fatal error | %s", str(exc))
        LOGGER.error(
            "Summary | %s",
            json.dumps(
                _build_summary(
                    "known-error",
                    False,
                    start_time,
                    {"error": str(exc)},
                ),
                ensure_ascii=False,
            ),
        )
        return EXIT_PLATFORM_ERROR

    except Exception as exc:
        LOGGER.exception("Unhandled exception | %s", str(exc))
        LOGGER.error(
            "Summary | %s",
            json.dumps(
                _build_summary(
                    "unhandled-exception",
                    False,
                    start_time,
                    {"error": str(exc)},
                ),
                ensure_ascii=False,
            ),
        )
        return EXIT_FAILURE

## ============================================================
## ENTRYPOINT
## ============================================================
if __name__ == "__main__":
    sys.exit(main())