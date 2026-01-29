"""
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Generic utilities: path helpers, reproducibility seed, device detection, timers, and config snapshot."
"""

from __future__ import annotations

import json
import os
import random
import time
import numpy as np
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

## --------------------------------------------------------------------------------------
## Path utilities
## --------------------------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    """
		Ensure a directory exists and return it

		Args:
			path: Directory path to create if needed

		Returns:
			The same Path instance, ensured to exist
    """
    
    path.mkdir(parents=True, exist_ok=True)
    return path

def resolve_project_root(start_path: Optional[Path] = None) -> Path:
    """
		Resolve project root directory

		Args:
			start_path: Optional starting path. Defaults to current file location

		Returns:
			Resolved project root path
    """
    
    ## Default to this file location if not provided
    current = start_path or Path(__file__).resolve()

    ## Traverse upwards until a marker file/folder is found
    ## Markers: .env, requirements.txt, or docker/ directory
    for parent in [current] + list(current.parents):
        if (
            (parent / ".env").exists()
            or (parent / "requirements.txt").exists()
            or (parent / "docker").exists()
        ):
            return parent

    ## Fallback: return filesystem root
    return current.anchor

## --------------------------------------------------------------------------------------
## Reproducibility utilities
## --------------------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """
		Set global random seed for reproducibility

		Args:
			seed: Seed value
    """
    ## Python built-in RNG
    random.seed(seed)

    ## NumPy RNG
    np.random.seed(seed)

    ## Optional: set PYTHONHASHSEED for deterministic hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

## --------------------------------------------------------------------------------------
## Device utilities
## --------------------------------------------------------------------------------------
def detect_device(use_gpu: bool = True) -> str:
    """
		Detect computation device

		Args:
			use_gpu: Whether GPU usage is allowed

		Returns:
			Device string ("cuda" or "cpu")
    """
    if not use_gpu:
        return "cpu"

    ## Lazy import to avoid hard dependency
    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"

## --------------------------------------------------------------------------------------
## Timing utilities
## --------------------------------------------------------------------------------------
@contextmanager
def timer(name: str) -> Iterator[None]:
    """
		Simple context manager to measure execution time

		Args:
			name: Name of the timed block
    """
    
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed = end_time - start_time

    ## Print is intentional here to avoid logger dependency at this level
    print(f"[TIMER] {name} took {elapsed:.2f} seconds")

def human_readable_seconds(seconds: float) -> str:
    """
		Convert seconds to a human-readable string

		Args:
			seconds: Duration in seconds

		Returns:
			Human-readable duration string
    """
    
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"

    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"

## --------------------------------------------------------------------------------------
## Configuration snapshot utilities
## --------------------------------------------------------------------------------------
def save_json(
    data: Dict[str, Any],
    output_path: Path,
    indent: int = 2,
) -> None:
    """
		Save a dictionary to a JSON file

		Args:
			data: Data to serialize
			output_path: Target JSON file path
			indent: JSON indentation level
    """
    
    ensure_dir(output_path.parent)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def snapshot_config(
    config_dict: Dict[str, Any],
    run_dir: Path,
    filename: str = "config_snapshot.json",
) -> Path:
    """
		Save a configuration snapshot to disk

		Args:
			config_dict: Configuration dictionary (JSON-serializable)
			run_dir: Directory of the current run
			filename: Snapshot filename

		Returns:
			Path to the saved snapshot file
    """
    
    ensure_dir(run_dir)
    snapshot_path = run_dir / filename
    save_json(config_dict, snapshot_path)
    return snapshot_path