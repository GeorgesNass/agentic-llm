'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Text processing helpers: normalization, detection, statistics for autonomous-ai-platform."
'''

from __future__ import annotations

import re
from typing import List

## ============================================================
## TEXT HELPERS
## ============================================================
def _count_words(text: str) -> int:
    """
        Count words in a simple way

        Args:
            text: Input text

        Returns:
            Word count
    """

    tokens = re.findall(r"\S+", text)
    return len(tokens)

def _detect_urls(text: str) -> bool:
    """
        Detect presence of URLs

        Args:
            text: Input text

        Returns:
            Boolean
    """

    return bool(re.search(r"https?://", text, flags=re.IGNORECASE))

def _detect_code_block(text: str) -> bool:
    """
        Detect presence of markdown code blocks

        Args:
            text: Input text

        Returns:
            Boolean
    """

    return "```" in text

def _normalize_text(text: str) -> str:
    """
        Normalize text for chunking and embedding

        Args:
            text: Input text

        Returns:
            Normalized text
    """

    ## Remove null bytes
    cleaned = text.replace("\x00", " ")

    ## Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
    