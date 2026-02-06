'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Calibration dataset utilities for PTQ backends (AWQ/GPTQ/ONNX) and sampling helpers."
'''

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

from src.core.errors import log_and_raise_missing_path, log_and_raise_pipeline_error
from src.utils.logging_utils import get_logger

## Initialize module-level logger
LOGGER = get_logger(__name__)


def load_calibration_texts(
    calibration_path: Path,
) -> List[str]:
    """
        Load calibration texts from a local dataset file

        Supported formats:
            - .txt: one sample per line
            - .json: list[str] or list[{"text": "..."}]
            - .jsonl: one json object per line with key "text"

        Args:
            calibration_path: Path to calibration dataset file

        Returns:
            A list of text samples
    """
    if not calibration_path.exists():
        log_and_raise_missing_path(calibration_path, context="calibration_dataset")

    suffix = calibration_path.suffix.lower()

    if suffix == ".txt":
        lines = calibration_path.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()]

    if suffix == ".json":
        data = json.loads(calibration_path.read_text(encoding="utf-8"))
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return [x.strip() for x in data if x.strip()]
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            texts = []
            for obj in data:
                text = str(obj.get("text", "")).strip()
                if text:
                    texts.append(text)
            return texts
        log_and_raise_pipeline_error(
            step="calibration_load_json",
            reason="Unsupported JSON schema for calibration dataset",
        )

    if suffix == ".jsonl":
        texts: List[str] = []
        for line in calibration_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj.get("text", "")).strip()
            if text:
                texts.append(text)
        return texts

    log_and_raise_pipeline_error(
        step="calibration_load",
        reason=f"Unsupported calibration file format: {suffix}",
    )
    return []


def sample_texts(
    texts: List[str],
    max_samples: int = 128,
    seed: Optional[int] = 42,
) -> List[str]:
    """
        Sample a subset of calibration texts for faster PTQ

        Args:
            texts: Full list of calibration texts
            max_samples: Maximum number of samples to return
            seed: Optional RNG seed for reproducibility

        Returns:
            A sampled list of texts
    """
    if not texts:
        return []

    if seed is not None:
        random.seed(seed)

    if len(texts) <= max_samples:
        return texts

    return random.sample(texts, k=max_samples)
