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
from pathlib import Path

from src.config.settings import build_pipeline_config
from src.core.errors import LocalQuantizationError
from src.pipeline import run_pipeline
from src.utils.logging_utils import get_logger

## Initialize module-level logger
LOGGER = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """
        Build the CLI argument parser

        Args:
            None

        Returns:
            A configured argparse parser
    """
    parser = argparse.ArgumentParser(
        description="local-quantization: quantize/export/benchmark LLMs locally",
    )

    parser.add_argument(
        "--env-file",
        type=str,
        default="",
        help="Optional path to a .env file to load before building config",
    )

    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved configuration (safe fields only) then exit",
    )

    return parser


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
        Print a safe configuration view without dumping sensitive values

        Args:
            None

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
        str(config.quantization.calibration_dataset)
        if config.quantization.calibration_dataset
        else "",
    )
    LOGGER.info("EXPORT_OUTPUT_DIR=%s", str(config.export.output_dir) if config.export else "")
    LOGGER.info("BENCHMARK_PROMPTS=%s", str(config.benchmark.prompts_path) if config.benchmark else "")
    return 0


def main() -> int:
    """
        Run the CLI entrypoint

        Args:
            None

        Returns:
            Exit code (0 success, non-zero failure)
    """
    parser = _build_parser()
    args = parser.parse_args()

    _load_dotenv_if_needed(args.env_file)

    try:
        if args.print_config:
            return _print_safe_config()

        config = build_pipeline_config()
        run_pipeline(config)
        LOGGER.info("Pipeline completed successfully")
        return 0

    except LocalQuantizationError as exc:
        LOGGER.error("Pipeline failed: %s", str(exc))
        return 2

    except Exception as exc:
        LOGGER.exception("Unexpected error: %s", str(exc))
        return 3


if __name__ == "__main__":
    sys.exit(main())
