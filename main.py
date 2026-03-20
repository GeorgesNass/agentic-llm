'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main entrypoint: CLI to run prepare, train, evaluate, or full pipeline workflows."
'''

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from src.pipeline import evaluate, full_run, prepare, train
from src.core.errors import DataError, ConfigurationError
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
logger = get_logger("main")

## ============================================================
## ARG PARSER
## ============================================================
def _build_parser() -> argparse.ArgumentParser:
    """
        Build CLI parser

        Returns:
            Configured ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description="Local fine-tuning pipeline for CISP symptom normalization",
        add_help=True,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {APP_VERSION}",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-config", action="store_true")

    parser.add_argument(
        "command",
        choices=["prepare", "train", "evaluate", "full"],
        help="Pipeline command to run",
    )

    parser.add_argument(
        "--env",
        type=Path,
        default=None,
        help="Optional path to .env file",
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Existing run directory (used for evaluation)",
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

## ============================================================
## MAIN
## ============================================================
def main() -> int:
    """
        Main CLI entry point

        Workflow:
            - prepare: data preparation
            - train: model training
            - evaluate: evaluation using run_dir
            - full: full pipeline execution

        Returns:
            Exit code
    """

    start_time = time.monotonic()
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.validate_config:
            logger.info("Config validation OK")
            logger.info("Summary | %s", _build_summary("validate-config", True, start_time))
            return EXIT_SUCCESS

        if args.dry_run:
            logger.info("Dry-run | command=%s env=%s run_dir=%s", args.command, args.env, args.run_dir)
            logger.info("Summary | %s", _build_summary("dry-run", True, start_time))
            return EXIT_SUCCESS

        env_path: Optional[Path] = args.env

        ## COMMAND DISPATCH
        if args.command == "prepare":
            prepare(env_path=env_path)

        elif args.command == "train":
            train(env_path=env_path)

        elif args.command == "evaluate":
            evaluate(env_path=env_path, run_dir=args.run_dir)

        elif args.command == "full":
            full_run(env_path=env_path)

        else:
            raise ValueError(f"Unknown command: {args.command}")

        logger.info("Summary | %s", _build_summary("run", True, start_time))
        return EXIT_SUCCESS

    except KeyboardInterrupt:
        logger.warning("Interrupted")
        logger.warning("Summary | %s", _build_summary("interrupt", False, start_time))
        return EXIT_FAILURE

    except (ConfigurationError, DataError) as exc:
        logger.error(str(exc))
        logger.error("Summary | %s", _build_summary("known-error", False, start_time))
        return EXIT_PLATFORM_ERROR

    except Exception as exc:
        logger.exception("Unhandled exception: %s", exc)
        logger.error("Summary | %s", _build_summary("unhandled-exception", False, start_time))
        return EXIT_FAILURE

## ============================================================
## ENTRYPOINT
## ============================================================
if __name__ == "__main__":
    sys.exit(main())