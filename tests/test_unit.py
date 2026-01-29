"""
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Pytest unit tests: dataset prep sanity checks and metrics correctness for CISP normalization workflow."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import os
import pytest
import csv

from src.core.metrics import (
    confusion_matrix,
    exact_match,
    hallucination_rate,
    label_coverage,
    missing_labels,
    most_confused_pairs,
)
from src.core.prepare_dataset import run_prepare_dataset
from src.utils.io_utils import read_jsonl
from src.config.settings import load_settings

## --------------------------------------------------------------------------------------
## Fixtures
## --------------------------------------------------------------------------------------
@pytest.fixture()
def tmp_dirs(tmp_path: Path) -> Dict[str, Path]:
    """
		Create temporary directories structure for tests

		Args:
			tmp_path: Pytest temporary base directory

		Returns:
			Dictionary of created paths
    """
    
    raw_dir = tmp_path / "data" / "raw"
    interim_dir = tmp_path / "data" / "interim"
    processed_dir = tmp_path / "data" / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    return {
        "raw_dir": raw_dir,
        "interim_dir": interim_dir,
        "processed_dir": processed_dir,
    }

class _DummyLogger:
    """
		Dummy logger used for unit tests
    """

    def info(self, msg: str) -> None:
        """
			No-op info method

			Args:
				msg: Log message
        """
        return

## --------------------------------------------------------------------------------------
## Helper writers
## --------------------------------------------------------------------------------------
def _write_raw_csv(raw_path: Path, rows: List[Dict[str, Any]]) -> None:
    """
		Write a CSV raw file for dataset preparation

		Args:
			raw_path: Target CSV file path
			rows: Rows to write
    """

    if not rows:
        raise ValueError("rows must not be empty")

    fieldnames = list(rows[0].keys())
    with raw_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def _write_label_list(path: Path, labels: List[str]) -> None:
    """
		Write a label list file (one label per line)

		Args:
			path: Target file path
			labels: List of labels
    """
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(label + "\n")

## --------------------------------------------------------------------------------------
## Tests: metrics
## --------------------------------------------------------------------------------------
def test_exact_match_basic() -> None:
    """
		Test exact match metric basic behavior
    """
    
    y_true = ["a", "b", "c", "d"]
    y_pred = ["a", "x", "c", "y"]
    assert exact_match(y_true, y_pred) == 0.5

def test_hallucination_rate_with_allowed_set() -> None:
    """
		Test hallucination rate when allowed set is provided
    """
    
    y_pred = ["cephalees", "migraine", "cephalees", "UNKNOWN"]
    allowed = ["cephalees", "migraine"]
    assert hallucination_rate(y_pred, allowed_labels=allowed) == 0.25

def test_confusion_pairs_sorted() -> None:
    """
		Test confusion matrix + most confused pairs ordering
    """
    
    y_true = ["a", "a", "a", "b", "b"]
    y_pred = ["x", "x", "a", "b", "a"]

    conf = confusion_matrix(y_true, y_pred)
    pairs = most_confused_pairs(conf, min_count=1)

    assert pairs[0] == ("a", "x", 2)
    assert ("b", "a", 1) in pairs

def test_label_coverage_and_missing_labels() -> None:
    """
		Test label coverage and missing labels helpers
    """
    
    y_pred = ["a", "a", "b"]
    coverage = label_coverage(y_pred)
    assert coverage["a"] == 2
    assert coverage["b"] == 1

    all_labels = ["a", "b", "c"]
    missing = missing_labels(all_labels, y_pred)
    assert missing == ["c"]

## --------------------------------------------------------------------------------------
## Tests: dataset preparation
## --------------------------------------------------------------------------------------
def test_prepare_dataset_creates_splits(tmp_dirs: Dict[str, Path]) -> None:
    """
		Test dataset preparation writes train/val/test JSONL files

		This test ensures that:
		- Raw input data is correctly split into train/val/test files
		- Output JSONL files are created in the processed directory
		- Duplicate samples are removed during preprocessing

		Args:
			tmp_dirs: Temporary directories fixture containing raw, interim,
			          and processed paths
    """
    
    raw_dir = tmp_dirs["raw_dir"]
    interim_dir = tmp_dirs["interim_dir"]
    processed_dir = tmp_dirs["processed_dir"]

    raw_csv = raw_dir / "sample.csv"
    rows = [
        {"text": "j'ai mal a la tete", "label": "cephalees"},
        {"text": "j'ai mal a la tete", "label": "cephalees"},
        {"text": "nausee et vomissements", "label": "nausees"},
        {"text": "douleur thoracique", "label": "douleur thoracique"},
        {"text": "toux seche", "label": "toux"},
        {"text": "fievre", "label": "fievre"},
        {"text": "fatigue", "label": "fatigue"},
        {"text": "vertiges", "label": "vertiges"},
        {"text": "eczema", "label": "eczema"},
        {"text": "diarrhee", "label": "diarrhee"},
    ]
    _write_raw_csv(raw_csv, rows)

    ## Use no label list (accept all)
    run_prepare_dataset(
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        train_file="train.jsonl",
        val_file="val.jsonl",
        test_file="test.jsonl",
        label_list_file=None,
        split_seed=42,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        logger=_DummyLogger(),
    )

    train_path = processed_dir / "train.jsonl"
    val_path = processed_dir / "val.jsonl"
    test_path = processed_dir / "test.jsonl"

    assert train_path.exists()
    assert val_path.exists()
    assert test_path.exists()

    ## Ensure JSONL is readable and non-empty
    train_records = read_jsonl(train_path)
    val_records = read_jsonl(val_path)
    test_records = read_jsonl(test_path)

    assert len(train_records) > 0
    assert len(val_records) >= 0
    assert len(test_records) >= 0

    ## Ensure deduplication happened (duplicate row present in raw)
    ## We cannot guarantee which split it landed in, but total should be < raw rows
    total = len(train_records) + len(val_records) + len(test_records)
    assert total < len(rows)

def test_prepare_dataset_skips_invalid_labels_with_allowed_set(
    tmp_dirs: Dict[str, Path],
) -> None:
    """
        Ensure dataset preparation skips samples whose labels are not in the allowed label list

        This test verifies that:
        - Rows with labels not present in the provided label list are excluded
        - Only allowed labels appear in the processed train/val/test splits

        Args:
            tmp_dirs: Temporary directories fixture containing raw, interim, and processed paths
    """
    
    raw_dir = tmp_dirs["raw_dir"]
    interim_dir = tmp_dirs["interim_dir"]
    processed_dir = tmp_dirs["processed_dir"]

    raw_csv = raw_dir / "sample.csv"
    rows = [
        {"text": "j'ai mal a la tete", "label": "cephalees"},
        {"text": "j'ai mal au ventre", "label": "douleur abdominale"},
        {"text": "symptome inconnu", "label": "label_inexistant"},
    ]
    _write_raw_csv(raw_csv, rows)

    labels_file = tmp_dirs["raw_dir"].parent / "labels.txt"
    _write_label_list(labels_file, ["cephalees", "douleur abdominale"])

    run_prepare_dataset(
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        train_file="train.jsonl",
        val_file="val.jsonl",
        test_file="test.jsonl",
        label_list_file=labels_file,
        split_seed=42,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        logger=_DummyLogger(),
    )

    ## Ensure no invalid label is present in processed outputs
    outputs: List[str] = []
    for split_name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        split_path = processed_dir / split_name
        records = read_jsonl(split_path)
        outputs.extend([str(r.get("output", "")) for r in records])

    assert "label_inexistant" not in outputs
    assert "cephalees" in outputs or "douleur abdominale" in outputs

def test_profile_switch_cpu_gpu() -> None:
    """
		Test PROFILE switch applies CPU_* / GPU_* training settings
    """

    ## Preserve original env values to avoid cross-test side effects
    original_profile = os.environ.get("PROFILE")
    original_cpu_use_gpu = os.environ.get("CPU_USE_GPU")
    original_gpu_use_gpu = os.environ.get("GPU_USE_GPU")

    try:
        ## Force CPU profile
        os.environ["PROFILE"] = "cpu"
        os.environ["CPU_USE_GPU"] = "false"
        os.environ["GPU_USE_GPU"] = "true"

        settings_cpu = load_settings(env_path=None)
        assert settings_cpu.training.use_gpu is False

        ## Force GPU profile
        os.environ["PROFILE"] = "gpu"
        settings_gpu = load_settings(env_path=None)
        assert settings_gpu.training.use_gpu is True

    finally:
        ## Restore env
        if original_profile is None:
            os.environ.pop("PROFILE", None)
        else:
            os.environ["PROFILE"] = original_profile

        if original_cpu_use_gpu is None:
            os.environ.pop("CPU_USE_GPU", None)
        else:
            os.environ["CPU_USE_GPU"] = original_cpu_use_gpu

        if original_gpu_use_gpu is None:
            os.environ.pop("GPU_USE_GPU", None)
        else:
            os.environ["GPU_USE_GPU"] = original_gpu_use_gpu
