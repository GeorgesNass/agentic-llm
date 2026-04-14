'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Utility functions for local finetuning: normalization, dataset validation, types, business rules and quality."
'''

from __future__ import annotations

import re
from typing import Any, Dict, List

from src.utils import get_logger

## ============================================================
## LOGGER INITIALIZATION
## ============================================================
logger = get_logger("data_utils")

def normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
        Normalize data fields

        Args:
            data: Input dictionary

        Returns:
            Normalized dictionary
    """

    normalized = {}

    for key, value in data.items():

        ## Normalize string
        if isinstance(value, str):
            logger.debug(f"Normalizing string: {key}")
            value = value.strip().lower()
            value = re.sub(r"\s+", " ", value)

        ## Normalize list
        if isinstance(value, list):
            logger.debug(f"Normalizing list: {key}")
            value = [
                v.strip().lower() if isinstance(v, str) else v
                for v in value
            ]

        normalized[key] = value

    return normalized

def validate_schema(data: Dict[str, Any]) -> List[Dict]:
    """
        Validate required fields for finetuning

        Args:
            data: Input dictionary

        Returns:
            List of issues
    """

    issues = []

    ## Required fields
    if "text" not in data:
        logger.error("Missing text field")
        issues.append({
            "rule": "schema_text",
            "level": "error",
            "message": "text is required",
        })

    if "model_name" not in data:
        logger.error("Missing model_name")
        issues.append({
            "rule": "schema_model",
            "level": "error",
            "message": "model_name is required",
        })

    if "dataset" not in data:
        logger.error("Missing dataset")
        issues.append({
            "rule": "schema_dataset",
            "level": "error",
            "message": "dataset is required",
        })

    return issues

def validate_types(data: Dict[str, Any]) -> List[Dict]:
    """
        Validate field types

        Args:
            data: Input dictionary

        Returns:
            List of issues
    """

    issues = []

    ## Text type
    if "text" in data and not isinstance(data["text"], str):
        logger.error("Invalid text type")
        issues.append({
            "rule": "type_text",
            "level": "error",
            "message": "text must be string",
        })

    ## Model type
    if "model_name" in data and not isinstance(data["model_name"], str):
        logger.error("Invalid model_name type")
        issues.append({
            "rule": "type_model",
            "level": "error",
            "message": "model_name must be string",
        })

    ## Dataset type
    if "dataset" in data and not isinstance(data["dataset"], list):
        logger.error("Invalid dataset type")
        issues.append({
            "rule": "type_dataset",
            "level": "error",
            "message": "dataset must be list",
        })

    return issues

def check_business_rules(data: Dict[str, Any]) -> List[Dict]:
    """
        Apply finetuning business rules

        Args:
            data: Input dictionary

        Returns:
            List of issues
    """

    issues = []

    ## Text length
    if "text" in data and len(data["text"]) < 3:
        logger.warning("Text too short")
        issues.append({
            "rule": "business_text_length",
            "level": "warning",
            "message": "Text too short",
        })

    ## Dataset size
    if "dataset" in data:
        if isinstance(data["dataset"], list) and len(data["dataset"]) < 2:
            logger.warning("Dataset too small")
            issues.append({
                "rule": "business_dataset_size",
                "level": "warning",
                "message": "Dataset too small for training",
            })

    return issues

def compute_quality_score(data: Dict[str, Any]) -> float:
    """
        Compute quality score

        Args:
            data: Input dictionary

        Returns:
            Score
    """

    text = data.get("text", "")

    if not text:
        logger.warning("Empty text for scoring")
        return 0.0

    valid_chars = sum(c.isalnum() for c in text)
    score = valid_chars / len(text)

    logger.debug(f"Quality score: {score}")

    return score