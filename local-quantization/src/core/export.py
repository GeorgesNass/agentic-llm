'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Export helpers for quantized artifacts and metadata."
'''

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.config.schemas import PipelineConfig
from src.core.errors import (
    log_and_raise_export_error,
    log_and_raise_pipeline_error,
)
from src.utils.logging_utils import get_logger
from src.utils.utils import ensure_parent_dir

## Initialize module-level logger
LOGGER = get_logger(__name__)


def _utc_run_id() -> str:
    """
        Build a UTC run id string

        Args:
            None

        Returns:
            A run id like run_YYYYMMDD_HHMMSS
    """
    return datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> None:
    """
        Ensure a directory exists

        Args:
            path: Directory path

        Returns:
            None
    """
    path.mkdir(parents=True, exist_ok=True)


def _build_export_dir(
    base_output_dir: Path,
    backend: str,
    run_id: str,
) -> Path:
    """
        Build the export directory layout

        Layout:
            <EXPORT_OUTPUT_DIR>/<backend>/<run_id>/

        Args:
            base_output_dir: Base export output dir
            backend: Backend identifier
            run_id: Run id string

        Returns:
            The resolved export directory path
    """
    return base_output_dir / backend / run_id


def _write_metadata(export_dir: Path, config: PipelineConfig) -> None:
    """
        Write a metadata snapshot for the export

        Args:
            export_dir: Export directory
            config: Pipeline configuration

        Returns:
            None
    """
    metadata: Dict[str, Any] = {
        "run_id": export_dir.name,
        "backend": config.quantization.backend,
        "bits": config.quantization.bits,
        "group_size": config.quantization.group_size,
        "model_name_or_path": config.model.model_name_or_path,
        "model_revision": config.model.revision,
        "adapter_path": str(config.model.adapter_path) if config.model.adapter_path else "",
        "calibration_dataset": (
            str(config.quantization.calibration_dataset)
            if config.quantization.calibration_dataset
            else ""
        ),
        "export_dir": str(export_dir),
        "created_utc": datetime.utcnow().isoformat(),
        "config_snapshot": {
            "mode": config.mode,
            "model": asdict(config.model),
            "quantization": asdict(config.quantization),
            "export": asdict(config.export) if config.export else None,
            "benchmark": asdict(config.benchmark) if config.benchmark else None,
        },
    }

    metadata_path = export_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Wrote export metadata: %s", metadata_path)


def run_export(config: PipelineConfig) -> Path:
    """
        Run the export step for quantized artifacts

        High-level workflow:
            1) Validate export configuration
            2) Prepare structured export directory
            3) Write metadata snapshot
            4) (Placeholder) Export artifacts to the export directory

        Args:
            config: Validated pipeline configuration

        Returns:
            The export directory path
    """
    if config.export is None:
        log_and_raise_pipeline_error(
            step="export",
            reason="Export configuration is missing",
        )

    base_output_dir: Path = config.export.output_dir.expanduser().resolve()
    backend = config.quantization.backend

    run_id = _utc_run_id()
    export_dir = _build_export_dir(base_output_dir, backend, run_id)

    LOGGER.info("Starting export to export_dir=%s", export_dir)

    try:
        _ensure_dir(export_dir)

        ## Placeholder file to make the export directory non-empty and explicit
        ensure_parent_dir(export_dir / "placeholder.txt")
        (export_dir / "placeholder.txt").write_text(
            "Export placeholder. Replace with real exported artifacts.\n",
            encoding="utf-8",
        )

        _write_metadata(export_dir, config)

    except Exception as exc:
        log_and_raise_export_error(
            artifact_name=str(export_dir),
            reason=str(exc),
        )

    LOGGER.info("Export step completed successfully")
    return export_dir
