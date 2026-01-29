"""
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Core ML orchestrator: high-level helpers to locate adapters, load allowed labels, and run core workflows."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

from src.utils.io_utils import read_label_list
from src.utils.utils import ensure_dir

def get_adapter_dir(run_dir: Path) -> Path:
    """
		Get LoRA adapter directory for a given run

		Args:
			run_dir: Training run directory

		Returns:
			Path to LoRA adapter directory

		Raises:
			FileNotFoundError: If adapter directory is missing
    """
    
    adapter_dir = run_dir / "exports" / "lora_adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    return adapter_dir

def get_report_dir(run_dir: Path) -> Path:
    """
		Get report directory for a given run (create if missing)

		Args:
			run_dir: Training run directory

		Returns:
			Path to report directory
    """
    
    return ensure_dir(run_dir / "reports")

def get_logs_dir(run_dir: Path) -> Path:
    """
		Get logs directory for a given run (create if missing)

		Args:
			run_dir: Training run directory

		Returns:
			Path to logs directory
    """
    
    return ensure_dir(run_dir / "logs")

def load_allowed_labels(label_list_file: Optional[Path]) -> Optional[list[str]]:
    """
		Load allowed labels list from disk

		Args:
			label_list_file: Optional path to label list file

		Returns:
			List of allowed labels or None
    """
    
    return read_label_list(label_list_file)

def resolve_run_dir(project_root: Path, run_dir: Optional[Path]) -> Path:
    """
		Resolve a run directory path (absolute or relative to project root)

		Args:
			project_root: Project root directory
			run_dir: Optional run directory path

		Returns:
			Resolved run directory path

		Raises:
			ValueError: If run_dir is None
    """
    
    if run_dir is None:
        raise ValueError("run_dir is required")

    if run_dir.is_absolute():
        return run_dir

    return (project_root / run_dir).resolve()

def validate_processed_files(processed_dir: Path, train_file: str, val_file: str, test_file: str) -> None:
    """
		Validate processed dataset files exist

		Args:
			processed_dir: Processed data directory
			train_file: Train filename
			val_file: Validation filename
			test_file: Test filename

		Raises:
			FileNotFoundError: If any required file is missing
    """
    
    required = [
        processed_dir / train_file,
        processed_dir / val_file,
        processed_dir / test_file,
    ]

    missing = [p for p in required if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing processed file(s): {missing_str}")