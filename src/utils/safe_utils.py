'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Safe helpers: safe string/JSON for logs and robust JSON parsing utilities."
'''

from __future__ import annotations

import json
import re
from typing import Any, Dict

import requests

## ============================================================
## INTERNAL HELPERS
## ============================================================
def _safe_json(value: Any, max_len: int = 2000) -> str:
    """
        Convert a value into a safe JSON-like string for logs

        Args:
            value: Any python object
            max_len: Max output length

        Returns:
            Safe string
    """

    try:
        ## Attempt JSON serialization
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        ## Fallback to string representation
        text = str(value)

    ## Truncate if too long
    if len(text) > max_len:
        return text[:max_len] + "...(truncated)"

    return text

def _safe_str(value: Any, max_len: int = 2000) -> str:
    """
        Convert a value into a safe string for logs

        Args:
            value: Any python object
            max_len: Max output length

        Returns:
            Safe string
    """

    text = str(value)

    ## Truncate if necessary
    if len(text) > max_len:
        return text[:max_len] + "...(truncated)"

    return text

def _safe_int(value: Any, default: int = 0) -> int:
    """
        Safe int conversion

        Args:
            value: Any input
            default: Default integer

        Returns:
            Integer
    """

    try:
        return int(value)
    except Exception:
        return default

def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
        Safe JSON loading

        Args:
            text: JSON string

        Returns:
            Dict or empty dict
    """

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _extract_first_json_object(text: str) -> Dict[str, Any]:
    """
        Extract first JSON object from LLM output

        Args:
            text: Raw model output

        Returns:
            Parsed dict
    """

    ## Attempt direct parse first
    direct = _safe_json_loads(text.strip())
    if direct:
        return direct

    ## Attempt regex-based extraction
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}

    return _safe_json_loads(match.group(0))

def _safe_json_parse(resp: requests.Response) -> Dict[str, Any]:
    """
        Parse response JSON safely

        Args:
            resp: requests.Response

        Returns:
            Parsed JSON dict

        Raises:
            ValueError: If JSON parsing fails
    """

    parsed = resp.json()
    if isinstance(parsed, dict):
        return parsed
    return {"_raw": parsed}