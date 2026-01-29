"""
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Dataset preparation: load raw symptom/label data, clean, deduplicate, split, and export JSONL for LoRA SFT."
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

## --------------------------------------------------------------------------------------
## Constants
## --------------------------------------------------------------------------------------
_DEFAULT_INSTRUCTION = (
    "You are a medical coding assistant. "
    "Map the patient-described symptom text to the official CISP label. "
    "Output ONLY the exact label from the allowed label set."
)

## --------------------------------------------------------------------------------------
## Helpers
## --------------------------------------------------------------------------------------
def _normalize_whitespace(text: str) -> str:
    """
		Normalize whitespace in a text

		Args:
			text: Raw input text

		Returns:
			Cleaned text
    """
    
    ## Replace multiple spaces/tabs/newlines with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _read_label_list(label_list_file: Optional[Path]) -> Optional[List[str]]:
    """
		Read an allowed label list file

		Args:
			label_list_file: File path containing one label per line

		Returns:
			List of labels or None if not provided
    """
    
    if label_list_file is None:
        return None

    labels: List[str] = []
    with label_list_file.open("r", encoding="utf-8") as f:
        for line in f:
            label = line.strip()
            if label:
                labels.append(label)

    return labels if labels else None

def _detect_raw_file(raw_dir: Path) -> Path:
    """
		Detect a raw input file inside raw_dir

		Args:
			raw_dir: Directory containing raw files

		Returns:
			Path to the first matching raw file

		Raises:
			FileNotFoundError: If no supported raw file is found
    """
    
    ## Supported formats in priority order
    candidates: List[Path] = []
    candidates.extend(sorted(raw_dir.glob("*.csv")))
    candidates.extend(sorted(raw_dir.glob("*.jsonl")))
    candidates.extend(sorted(raw_dir.glob("*.json")))

    if not candidates:
        raise FileNotFoundError(
            f"No raw data file found in {raw_dir}. Expected .csv, .jsonl, or .json"
        )

    return candidates[0]

def _load_rows_from_csv(file_path: Path) -> List[Dict[str, Any]]:
    """
		Load rows from a CSV file

		Args:
			file_path: CSV file path

		Returns:
			List of row dictionaries
    """
    
    rows: List[Dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    return rows

def _load_rows_from_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
		Load rows from a JSONL file

		Args:
			file_path: JSONL file path

		Returns:
			List of row dictionaries
    """
    
    rows: List[Dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return rows

def _load_rows_from_json(file_path: Path) -> List[Dict[str, Any]]:
    """
		Load rows from a JSON file (list of objects)

		Args:
			file_path: JSON file path

		Returns:
			List of row dictionaries
    """
    
    with file_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return [dict(x) for x in payload]

    if isinstance(payload, dict):
        for key in ["data", "rows", "items", "examples"]:
            if key in payload and isinstance(payload[key], list):
                return [dict(x) for x in payload[key]]

    raise ValueError("Unsupported JSON structure. Expected a list of objects.")

def _extract_text_and_label(row: Dict[str, Any]) -> Tuple[str, str]:
    """
		Extract symptom text and label from a raw row

		Args:
			row: Raw record dictionary

		Returns:
			Tuple (text, label)

		Raises:
			KeyError: If required fields are missing
    """
    
    ## Accept common column names
    text_keys = ["text", "symptom_text", "symptoms", "input", "utterance", "description"]
    label_keys = ["label", "cisp", "cisp_label", "output", "target", "code"]

    text_val: Optional[str] = None
    label_val: Optional[str] = None

    for key in text_keys:
        if key in row and row[key] is not None and str(row[key]).strip():
            text_val = str(row[key])
            break

    for key in label_keys:
        if key in row and row[key] is not None and str(row[key]).strip():
            label_val = str(row[key])
            break

    if text_val is None:
        raise KeyError(f"Missing symptom text field. Tried keys: {text_keys}")
    if label_val is None:
        raise KeyError(f"Missing label field. Tried keys: {label_keys}")

    return text_val, label_val

def _deduplicate(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
		Deduplicate records by (input, output)

		Args:
			records: List of instruction records

		Returns:
			Deduplicated list
    """
    
    seen = set()
    deduped: List[Dict[str, Any]] = []

    for rec in records:
        key = (rec.get("input", ""), rec.get("output", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)

    return deduped

def _split_indices(
    n: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
		Create deterministic split indices

		Args:
			n: Number of samples
			seed: Split seed
			train_ratio: Train split ratio
			val_ratio: Validation split ratio
			test_ratio: Test split ratio

		Returns:
			Train, val, test index arrays
    """
    
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return (
        indices[:train_end],
        indices[train_end:val_end],
        indices[val_end:],
    )

def _write_jsonl(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    """
		Write records to a JSONL file

		Args:
			records: List of dictionaries to write
			output_path: Target JSONL file path
    """
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

## --------------------------------------------------------------------------------------
## Main entry
## --------------------------------------------------------------------------------------
def run_prepare_dataset(
    raw_dir: Path,
    interim_dir: Path,
    processed_dir: Path,
    train_file: str,
    val_file: str,
    test_file: str,
    label_list_file: Optional[Path],
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    logger: Any,
) -> None:
    """
		Prepare dataset for LoRA SFT

		Args:
			raw_dir: Directory containing raw input file(s)
			interim_dir: Directory for intermediate cleaned outputs
			processed_dir: Directory for final JSONL train/val/test
			train_file: Train JSONL filename
			val_file: Validation JSONL filename
			test_file: Test JSONL filename
			label_list_file: Optional allowed label list file (one label per line)
			split_seed: Seed for deterministic splitting
			train_ratio: Train split ratio
			val_ratio: Validation split ratio
			test_ratio: Test split ratio
			logger: Logger instance

		Raises:
			FileNotFoundError: If no raw file is found
			ValueError: If dataset is empty or labels are invalid
			KeyError: If required columns are missing
    """
    
    ## Ensure directories exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    ## Detect and load raw file
    raw_file = _detect_raw_file(raw_dir)
    logger.info(f"Raw file detected: {raw_file.name}")

    if raw_file.suffix.lower() == ".csv":
        raw_rows = _load_rows_from_csv(raw_file)
    elif raw_file.suffix.lower() == ".jsonl":
        raw_rows = _load_rows_from_jsonl(raw_file)
    elif raw_file.suffix.lower() == ".json":
        raw_rows = _load_rows_from_json(raw_file)
    else:
        raise ValueError(f"Unsupported raw file format: {raw_file.suffix}")

    if not raw_rows:
        raise ValueError("Raw dataset is empty")

    ## Load allowed labels if provided
    allowed_labels = _read_label_list(label_list_file)
    allowed_set = set(allowed_labels) if allowed_labels is not None else None

    ## Build instruction-format records
    records: List[Dict[str, Any]] = []
    invalid_label_count = 0
    dropped_empty_count = 0

    for row in raw_rows:
        text, label = _extract_text_and_label(row)

        text_clean = _normalize_whitespace(text)
        label_clean = _normalize_whitespace(label)

        if not text_clean or not label_clean:
            dropped_empty_count += 1
            continue

        if allowed_set is not None and label_clean not in allowed_set:
            invalid_label_count += 1
            continue

        records.append(
            {
                "instruction": _DEFAULT_INSTRUCTION,
                "input": text_clean,
                "output": label_clean,
            }
        )

    logger.info(f"Raw rows: {len(raw_rows)}")
    logger.info(f"Records built: {len(records)}")
    logger.info(f"Dropped empty rows: {dropped_empty_count}")
    if allowed_set is not None:
        logger.info(f"Invalid labels skipped: {invalid_label_count}")

    if not records:
        raise ValueError("No valid records after cleaning/validation")

    ## Deduplicate
    records = _deduplicate(records)
    logger.info(f"Records after deduplication: {len(records)}")

    ## Save interim snapshot
    interim_path = interim_dir / "cleaned_deduped.jsonl"
    _write_jsonl(records, interim_path)
    logger.info(f"Interim file written: {interim_path}")

    ## Split deterministically
    train_idx, val_idx, test_idx = _split_indices(
        n=len(records),
        seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]

    logger.info(
        f"Split sizes => train: {len(train_records)}, "
        f"val: {len(val_records)}, "
        f"test: {len(test_records)}"
    )

    ## Export final JSONL
    _write_jsonl(train_records, processed_dir / train_file)
    _write_jsonl(val_records, processed_dir / val_file)
    _write_jsonl(test_records, processed_dir / test_file)

    logger.info("Dataset preparation completed successfully")