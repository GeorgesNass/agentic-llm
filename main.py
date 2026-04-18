'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "CLI entrypoint for local-quantization: load env config and run the pipeline."
'''

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from src.config.settings import build_pipeline_config
from src.core.data_consistency import run_data_consistency
from src.core.data_quality import run_data_quality
from src.core.errors import LocalQuantizationError
from src.pipeline import run_pipeline
from src.utils.logging_utils import get_logger

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
## ARG PARSER
## ============================================================
def _build_parser() -> argparse.ArgumentParser:
    """
        Build CLI argument parser

        Returns:
            Configured ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description="local-quantization: quantize/export/benchmark LLMs locally",
        add_help=True,
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-config", action="store_true")

    parser.add_argument("--env-file", type=str, default="", help="Optional path to a .env file to load before building config")

    parser.add_argument("--print-config", action="store_true", help="Print resolved configuration (safe fields only) then exit")

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

def _load_dotenv_if_needed(env_file: str) -> None:
    """
        Load a .env file if provided and python-dotenv is installed

        Args:
            env_file: Path to the .env file

        Returns:
            None
    """

    if env_file.strip() == "":
        return

    env_path = Path(env_file).expanduser().resolve()

    if not env_path.exists():
        LOGGER.warning("Provided .env file does not exist: %s", env_path)
        return

    try:
        from dotenv import load_dotenv
    except Exception:
        LOGGER.warning("python-dotenv not installed, skipping .env loading")
        return

    load_dotenv(dotenv_path=str(env_path), override=False)
    LOGGER.info("Loaded .env file: %s", env_path)

def _print_safe_config() -> int:
    """
        Print safe configuration without sensitive values

        Returns:
            Exit code
    """

    config = build_pipeline_config()

    LOGGER.info("PIPELINE_MODE=%s", config.mode)
    LOGGER.info("MODEL_NAME_OR_PATH=%s", config.model.model_name_or_path)
    LOGGER.info("MODEL_REVISION=%s", config.model.revision or "")
    LOGGER.info("ADAPTER_PATH=%s", str(config.model.adapter_path) if config.model.adapter_path else "")
    LOGGER.info("QUANT_BACKEND=%s", config.quantization.backend)
    LOGGER.info("QUANT_BITS=%s", config.quantization.bits)
    LOGGER.info("QUANT_GROUP_SIZE=%s", config.quantization.group_size or "")
    LOGGER.info(
        "CALIBRATION_DATASET=%s",
        str(config.quantization.calibration_dataset) if config.quantization.calibration_dataset else "",
    )
    LOGGER.info("EXPORT_OUTPUT_DIR=%s", str(config.export.output_dir) if config.export else "")
    LOGGER.info("BENCHMARK_PROMPTS=%s", str(config.benchmark.prompts_path) if config.benchmark else "")

    return EXIT_SUCCESS
    
## ============================================================
## MAIN
## ============================================================
def main() -> int:
    """
        Main CLI entry point

        Workflow:
            - load optional .env
            - optionally print config
            - run quantization pipeline

        Returns:
            Exit code
    """

    start_time = time.monotonic()
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.validate_config:
            config = build_pipeline_config()
            LOGGER.info("Config validation OK | mode=%s", config.mode)
            LOGGER.info("Summary | %s", _build_summary("validate-config", True, start_time))
            return EXIT_SUCCESS

        if args.dry_run:
            LOGGER.info("Dry-run | env_file=%s print_config=%s", args.env_file, bool(args.print_config))
            LOGGER.info("Summary | %s", _build_summary("dry-run", True, start_time))
            return EXIT_SUCCESS

        _load_dotenv_if_needed(args.env_file)

        if args.print_config:
            code = _print_safe_config()
            LOGGER.info("Summary | %s", _build_summary("print-config", True, start_time))
            return code

        config = build_pipeline_config()

        ## DATA CONSISTENCY CHECK
        consistency_result = run_data_consistency(
            data={
                "text": "quantization_run",
                "model_name": config.model.model_name_or_path,
                "bits": config.quantization.bits,
            },
            strict=True,
        )

        LOGGER.info(f"Consistency OK: {consistency_result['is_consistent']}")

        if not consistency_result["is_consistent"]:
            raise LocalQuantizationError("Data consistency failed before pipeline")

        ## DATA QUALITY CHECK
        quality_result = run_data_quality(
            data=[
                config.quantization.bits,
                config.quantization.group_size,
            ],
            method="zscore",
            strict=False,
        )

        LOGGER.info(f"Quality score: {quality_result['score']}")

        if quality_result["errors"] > 0:
            raise LocalQuantizationError("Data quality failed before quantization")

        ## RUN PIPELINE
        run_pipeline(config)

        LOGGER.info("Pipeline completed successfully")
        LOGGER.info("Summary | %s", _build_summary("run", True, start_time))
        return EXIT_SUCCESS

    except KeyboardInterrupt:
        LOGGER.warning("Interrupted")
        LOGGER.warning("Summary | %s", _build_summary("interrupt", False, start_time))
        return EXIT_FAILURE

    except LocalQuantizationError as exc:
        LOGGER.error("Pipeline failed: %s", str(exc))
        LOGGER.error("Summary | %s", _build_summary("known-error", False, start_time))
        return EXIT_PLATFORM_ERROR

    except Exception as exc:
        LOGGER.exception("Unexpected error: %s", str(exc))
        LOGGER.error("Summary | %s", _build_summary("unhandled-exception", False, start_time))
        return EXIT_FAILURE

## ============================================================
## ENTRYPOINT
## ============================================================
if __name__ == "__main__":
    sys.exit(main())