'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Tokenization utilities for metrics and cost estimation (simple tokenizer + optional tiktoken integration)."
'''

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

## ============================================================
## OPTIONAL DEPENDENCIES
## ============================================================
## tiktoken is optional and used only when installed
try:
    import tiktoken
except Exception:
    tiktoken = None

## ============================================================
## LOCAL NORMALIZER (BREAK CIRCULAR IMPORT)
## ============================================================
def normalize_text_basic(text: str) -> str:
    """
        Normalize text for simple matching

        High-level workflow:
            1) Lowercase
            2) Strip spaces
            3) Collapse repeated whitespace

        Args:
            text: Raw text

        Returns:
            Normalized text
    """

    lowered = text.lower().strip()

    ## Replace multiple whitespace with single space
    return re.sub(r"\s+", " ", lowered)

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(frozen=True)
class TokenizationReport:
    """
        Tokenization report describing how tokens were computed

        Args:
            method: Tokenization method used
            model: Optional model name used for tokenization
            n_tokens: Computed number of tokens
            warnings: List of warnings
    """

    method: str
    model: Optional[str]
    n_tokens: int
    warnings: List[str]

## ============================================================
## SIMPLE TOKENIZER (METRICS)
## ============================================================
def tokenize_alnum(text: str) -> List[str]:
    """
        Tokenize text using a simple alphanumeric tokenizer

        High-level workflow:
            1) Normalize text (lowercase, collapse spaces)
            2) Extract alphanumeric tokens

        Args:
            text: Raw text

        Returns:
            List of tokens
    """

    ## Normalize basic text
    cleaned = normalize_text_basic(text)

    ## Extract tokens (letters + digits)
    return re.findall(r"[a-z0-9]+", cleaned)

## ============================================================
## TOKEN COUNT ESTIMATION
## ============================================================
def approximate_token_count(text: str) -> int:
    """
        Approximate token count using a char-based heuristic

        Notes:
            - Rough approximation: 1 token ~ 4 characters
            - Used as fallback when no tokenizer is available

        Args:
            text: Input text

        Returns:
            Approximate token count
    """

    stripped = text.strip()
    if stripped == "":
        return 0

    ## Use simple heuristic for token estimate
    return max(1, (len(stripped) + 3) // 4)

def count_tokens_openai_tiktoken(
    text: str,
    model: str,
) -> TokenizationReport:
    """
        Count tokens using tiktoken for OpenAI-compatible models

        Notes:
            - Requires tiktoken installed
            - Falls back to approximate count if encoding cannot be resolved

        Args:
            text: Input text
            model: Model name for tokenizer resolution

        Returns:
            TokenizationReport
    """

    warnings: List[str] = []

    if tiktoken is None:
        ## Fallback when tiktoken is missing
        warnings.append("tiktoken not installed, falling back to approximate_token_count")
        return TokenizationReport(
            method="approx",
            model=model,
            n_tokens=approximate_token_count(text),
            warnings=warnings,
        )

    ## Resolve encoding from model
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        ## Fallback to a known encoding
        warnings.append("encoding_for_model failed, using cl100k_base")
        encoding = tiktoken.get_encoding("cl100k_base")

    ## Encode and count tokens
    try:
        n_tokens = len(encoding.encode(text))
        return TokenizationReport(
            method="tiktoken",
            model=model,
            n_tokens=int(n_tokens),
            warnings=warnings,
        )
    except Exception as exc:
        warnings.append(f"tiktoken encode failed: {str(exc)}")
        return TokenizationReport(
            method="approx",
            model=model,
            n_tokens=approximate_token_count(text),
            warnings=warnings,
        )

def count_tokens(
    text: str,
    provider: str,
    model: Optional[str] = None,
) -> TokenizationReport:
    """
        Count tokens for a provider/model with best-effort strategy

        High-level workflow:
            1) If provider is openai and model is provided -> try tiktoken
            2) Otherwise -> fallback to approximate count

        Args:
            text: Input text
            provider: Provider name (openai, google, xai, etc.)
            model: Optional model name

        Returns:
            TokenizationReport
    """

    normalized_provider = provider.strip().lower()
    warnings: List[str] = []

    ## Prefer tiktoken only for OpenAI-compatible models
    if normalized_provider in {"openai", "azure_openai"} and model:
        return count_tokens_openai_tiktoken(text=text, model=model)

    ## Default fallback to approximation
    warnings.append("tokenizer not available for provider, using approximate count")
    return TokenizationReport(
        method="approx",
        model=model,
        n_tokens=approximate_token_count(text),
        warnings=warnings,
    )