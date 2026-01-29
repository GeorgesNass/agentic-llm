"""
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main entrypoint: CLI to run prepare, train, evaluate, or full pipeline workflows."
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from src.pipeline import evaluate, full_run, prepare, train
from src.core.errors import DataError, ConfigurationError
from src.utils.logging_utils import get_logger

logger = get_logger("main")

## --------------------------------------------------------------------------------------
## CLI helpers
## --------------------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    """
		Parse command-line arguments

		Returns:
			Parsed argparse namespace
    """
    parser = argparse.ArgumentParser(
        description="Local fine-tuning pipeline for CISP symptom normalization"
    )

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

    return parser.parse_args()

## --------------------------------------------------------------------------------------
## Main
## --------------------------------------------------------------------------------------
def main() -> None:
    """
		Main CLI entrypoint
    """
    args = _parse_args()

    env_path: Optional[Path] = args.env

    if args.command == "prepare":
        prepare(env_path=env_path)

    elif args.command == "train":
        train(env_path=env_path)

    elif args.command == "evaluate":
        evaluate(env_path=env_path, run_dir=args.run_dir)

    elif args.command == "full":
        full_run(env_path=env_path)

    else:
        ## This should never happen due to argparse choices
        raise ValueError(f"Unknown command: {args.command}")

if __name__ == "__main__":
    try:
        main()
    except (ConfigurationError, DataError) as exc:
        logger.error(str(exc))
        raise SystemExit(1)

