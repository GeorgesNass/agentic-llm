'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Utility functions for LLM gateway: normalization, validation of messages, params and quality"
'''

from __future__ import annotations

from typing import Any, Dict, List

from src.utils import get_logger

## ============================================================
## LOGGER INITIALIZATION
## ============================================================
logger = get_logger("data_utils")

def normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
        Normalize input payload

        Args:
            data: Input dictionary

        Returns:
            Dict[str, Any]
    """

    normalized = {}

    for key, value in data.items():

        ## Normalize prompt
        if key == "prompt" and isinstance(value, str):
            logger.debug("Normalizing prompt")
            value = value.strip()

        ## Normalize messages
        elif key == "messages" and isinstance(value, list):
            logger.debug("Normalizing messages")

            normalized_messages = []

            for msg in value:
                if isinstance(msg, dict):
                    normalized_messages.append({
                        "role": msg.get("role"),
                        "content": msg.get("content", "").strip(),
                    })

            value = normalized_messages

        normalized[key] = value

    return normalized

def validate_schema(data: Dict[str, Any]) -> List[Dict]:
    """
        Validate required fields for LLM request

        Args:
            data: Input dictionary

        Returns:
            List[Dict]
    """

    issues = []

    ## At least one input must exist
    if "prompt" not in data and "messages" not in data:
        logger.error("Missing prompt/messages")
        issues.append({
            "rule": "schema_input",
            "level": "error",
            "message": "prompt or messages required",
        })

    ## Model required
    if "model" not in data:
        logger.error("Missing model")
        issues.append({
            "rule": "schema_model",
            "level": "error",
            "message": "model is required",
        })

    return issues

def validate_types(data: Dict[str, Any]) -> List[Dict]:
    """
        Validate field types

        Args:
            data: Input dictionary

        Returns:
            List[Dict]
    """

    issues = []

    ## Prompt type
    if "prompt" in data and not isinstance(data["prompt"], str):
        logger.error("Invalid prompt type")
        issues.append({
            "rule": "type_prompt",
            "level": "error",
            "message": "prompt must be string",
        })

    ## Messages type
    if "messages" in data and not isinstance(data["messages"], list):
        logger.error("Invalid messages type")
        issues.append({
            "rule": "type_messages",
            "level": "error",
            "message": "messages must be list",
        })

    ## Model type
    if "model" in data and not isinstance(data["model"], str):
        logger.error("Invalid model type")
        issues.append({
            "rule": "type_model",
            "level": "error",
            "message": "model must be string",
        })

    return issues

def validate_params_basic(data: Dict[str, Any]) -> List[Dict]:
    """
        Validate basic parameter types

        Args:
            data: Input dictionary

        Returns:
            List[Dict]
    """

    issues = []

    ## max_tokens
    if "max_tokens" in data and not isinstance(data["max_tokens"], int):
        logger.error("Invalid max_tokens type")
        issues.append({
            "rule": "param_max_tokens",
            "level": "error",
            "message": "max_tokens must be int",
        })

    ## temperature
    if "temperature" in data and not isinstance(data["temperature"], (int, float)):
        logger.error("Invalid temperature type")
        issues.append({
            "rule": "param_temperature",
            "level": "error",
            "message": "temperature must be float",
        })

    ## top_p
    if "top_p" in data and not isinstance(data["top_p"], (int, float)):
        logger.error("Invalid top_p type")
        issues.append({
            "rule": "param_top_p",
            "level": "error",
            "message": "top_p must be float",
        })

    return issues

def compute_quality_score(data: Dict[str, Any]) -> float:
    """
        Compute quality score based on prompt/messages length

        Args:
            data: Input dictionary

        Returns:
            float
    """

    prompt = data.get("prompt", "")
    messages = data.get("messages", [])

    ## Use prompt if exists
    if prompt:
        length = len(prompt)
    elif messages:
        ## Sum all message content lengths
        length = sum(len(m.get("content", "")) for m in messages if isinstance(m, dict))
    else:
        logger.warning("No input for scoring")
        return 0.0

    if length == 0:
        return 0.0

    ## Simple normalized score
    score = min(length / 1000.0, 1.0)

    logger.debug(f"Quality score: {score}")

    return score