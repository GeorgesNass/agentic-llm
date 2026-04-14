'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Utility functions for RAG data consistency: normalization, schema validation, types, cross-source, business rules and quality."
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

        ## Normalize strings
        if isinstance(value, str):
            logger.debug(f"Normalizing string: {key}")
            value = value.strip().lower()
            value = re.sub(r"\s+", " ", value)

        ## Normalize lists
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
        Validate required RAG fields

        Args:
            data: Input dictionary

        Returns:
            List of issues
    """

    issues = []

    ## Require at least query OR text
    if "query" not in data and "text" not in data:
        logger.error("Missing query/text")
        issues.append({
            "rule": "schema",
            "level": "error",
            "message": "At least one of query or text must be present",
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

    ## Query type
    if "query" in data and not isinstance(data["query"], str):
        logger.error("Invalid query type")
        issues.append({
            "rule": "type_query",
            "level": "error",
            "message": "query must be string",
        })

    ## Text type
    if "text" in data and not isinstance(data["text"], str):
        logger.error("Invalid text type")
        issues.append({
            "rule": "type_text",
            "level": "error",
            "message": "text must be string",
        })

    ## Chunks type
    if "chunks" in data and not isinstance(data["chunks"], list):
        logger.error("Invalid chunks type")
        issues.append({
            "rule": "type_chunks",
            "level": "error",
            "message": "chunks must be list",
        })

    ## Embeddings type
    if "embeddings" in data and not isinstance(data["embeddings"], list):
        logger.error("Invalid embeddings type")
        issues.append({
            "rule": "type_embeddings",
            "level": "error",
            "message": "embeddings must be list",
        })

    return issues

def compare_sources(data: Dict[str, Any]) -> List[Dict]:
    """
        Compare cross-source fields

        Args:
            data: Input dictionary

        Returns:
            List of issues
    """

    issues = []

    ## Compare query vs text
    if "query" in data and "text" in data:
        if data["query"] == data["text"]:
            logger.warning("Query identical to text")
            issues.append({
                "rule": "cross_query_text",
                "level": "warning",
                "message": "Query identical to text",
            })

    ## Compare metadata text
    if "text" in data and "metadata_text" in data:
        if data["text"] != data["metadata_text"]:
            logger.warning("Mismatch text vs metadata")
            issues.append({
                "rule": "cross_text",
                "level": "warning",
                "message": "Mismatch between text and metadata_text",
            })

    return issues

def check_business_rules(data: Dict[str, Any]) -> List[Dict]:
    """
        Apply RAG business rules

        Args:
            data: Input dictionary

        Returns:
            List of issues
    """

    issues = []

    ## Query length
    if "query" in data and len(data["query"]) < 3:
        logger.warning("Query too short")
        issues.append({
            "rule": "business_query_length",
            "level": "warning",
            "message": "Query too short",
        })

    ## Text length
    if "text" in data and len(data["text"]) < 3:
        logger.warning("Text too short")
        issues.append({
            "rule": "business_text_length",
            "level": "warning",
            "message": "Text too short",
        })

    ## Chunks consistency
    if "chunks" in data and isinstance(data["chunks"], list):
        if len(data["chunks"]) == 0:
            logger.warning("No chunks retrieved")
            issues.append({
                "rule": "business_chunks",
                "level": "warning",
                "message": "No chunks retrieved",
            })

    ## Embedding dimension
    if "embeddings" in data:
        if isinstance(data["embeddings"], list) and len(data["embeddings"]) < 10:
            logger.warning("Embedding too small")
            issues.append({
                "rule": "business_embedding_dim",
                "level": "warning",
                "message": "Embedding dimension too small",
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

    text = data.get("text") or data.get("query", "")

    if not text:
        logger.warning("Empty text/query for scoring")
        return 0.0

    ## Ratio alphanumeric
    valid_chars = sum(c.isalnum() for c in text)
    score = valid_chars / len(text)

    logger.debug(f"Quality score: {score}")

    return score

def detect_duplicates(data: Dict[str, Any]) -> List[Any]:
    """
        Detect duplicate values

        Args:
            data: Input dictionary

        Returns:
            List of duplicates
    """

    seen = set()
    duplicates = []

    for value in data.values():

        ## Skip complex types
        if isinstance(value, (list, dict)):
            continue

        if value in seen:
            logger.warning(f"Duplicate detected: {value}")
            duplicates.append(value)
        else:
            seen.add(value)

    return duplicates