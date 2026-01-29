"""
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "I/O utilities: read/write JSONL, read label lists, and save JSON reports in a reusable and consistent way."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
		Read a JSONL file into a list of dictionaries

		Args:
			file_path: JSONL file path

		Returns:
			List of parsed JSON objects

		Raises:
			FileNotFoundError: If file_path does not exist
    """
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    records: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records

def write_jsonl(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
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

def read_label_list(label_list_file: Optional[Path]) -> Optional[List[str]]:
    """
		Read an allowed label list file (one label per line)

		Args:
			label_list_file: File path containing one label per line

		Returns:
			List of labels or None if not provided or file does not exist
    """
    
    if label_list_file is None:
        return None

    if not label_list_file.exists():
        return None

    labels: List[str] = []
    with label_list_file.open("r", encoding="utf-8") as f:
        for line in f:
            label = line.strip()
            if label:
                labels.append(label)

    return labels if labels else None

def save_json(data: Dict[str, Any], output_path: Path, indent: int = 2) -> None:
    """
		Save a dictionary to a JSON file

		Args:
			data: JSON-serializable dictionary
			output_path: Target JSON file path
			indent: JSON indentation level
    """
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def ensure_allowed_set(allowed_labels: Optional[Iterable[str]]) -> Optional[set[str]]:
    """
		Convert an optional iterable of labels into a set

		Args:
			allowed_labels: Optional iterable of allowed labels

		Returns:
			Set of allowed labels or None if input is None
    """
    
    if allowed_labels is None:
        return None

    return set(allowed_labels)