'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Costing helpers: pricing resolution, chunking-based token estimation, and USD cost computation."
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.core.errors import (
    log_and_raise_pipeline_error,
    log_and_raise_validation_error,
)

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(frozen=True)
class PricingRow:
    """
        Pricing row for a given provider/model

        Args:
            input_per_1k: Cost per 1k input tokens (USD)
            output_per_1k: Cost per 1k output tokens (USD)
            embedding_per_1k: Cost per 1k embedding tokens (USD)
    """

    input_per_1k: float
    output_per_1k: float
    embedding_per_1k: float

@dataclass(frozen=True)
class ScanStats:
    """
        Aggregated scan stats for text or folder processing

        Args:
            n_files: Number of files processed (0 for text mode)
            n_chars: Total characters processed
            input_tokens: Total estimated input tokens
    """

    n_files: int
    n_chars: int
    input_tokens: int

## ============================================================
## MODEL AND PRICING RESOLUTION
## ============================================================
def resolve_model_for_provider(
    provider: str,
    requested_model: Optional[str],
    models_catalog: dict[str, Any],
) -> str:
    """
        Resolve the effective model name for a provider

        Notes:
            - If requested_model is provided, it is used as-is
            - Otherwise, tries to use a default model from models_catalog
            - If missing, falls back to a provider-based placeholder

        Args:
            provider: Provider identifier
            requested_model: Optional model name override
            models_catalog: Models catalog dictionary

        Returns:
            Effective model name
    """
    
    ## Use explicit model if provided
    if requested_model is not None and requested_model.strip() != "":
        return requested_model.strip()

    ## Try catalog default resolution
    defaults = models_catalog.get("defaults", {})
    if isinstance(defaults, dict):
        default_model = defaults.get(provider)
        if isinstance(default_model, str) and default_model.strip() != "":
            return default_model.strip()

    ## Fallback to provider placeholder
    return f"{provider}:default"

def get_pricing_row(
    provider: str,
    model: str,
    pricing_catalog: dict[str, Any],
) -> PricingRow:
    """
        Retrieve pricing row for a provider/model from pricing_catalog

        Expected JSON shapes supported:
            A) pricing_catalog[provider][model] = {input_per_1k, output_per_1k, embedding_per_1k}
            B) pricing_catalog["providers"][provider][model] = {...}

        Design choice:
            - If model is missing, we attempt provider-level "default"
            - This keeps the pipeline robust even when catalog is incomplete

        Args:
            provider: Provider identifier
            model: Model name
            pricing_catalog: Pricing catalog dictionary

        Returns:
            PricingRow instance
    """
    
    ## Try common layouts
    provider_block: Any = pricing_catalog.get(provider)
    if provider_block is None:
        providers_root = pricing_catalog.get("providers", {})
        if isinstance(providers_root, dict):
            provider_block = providers_root.get(provider)

    if not isinstance(provider_block, dict):
        log_and_raise_pipeline_error(
            step="pricing_lookup",
            reason=f"Missing provider in pricing_catalog: provider={provider}",
        )

    ## Try exact model pricing then fallback to provider default
    model_block: Any = provider_block.get(model)
    if model_block is None:
        model_block = provider_block.get("default")

    if not isinstance(model_block, dict):
        log_and_raise_pipeline_error(
            step="pricing_lookup",
            reason=f"Missing model pricing in pricing_catalog: provider={provider} model={model}",
        )

    ## Extract with safe defaults
    input_per_1k = float(model_block.get("input_per_1k", 0.0))
    output_per_1k = float(model_block.get("output_per_1k", 0.0))
    embedding_per_1k = float(model_block.get("embedding_per_1k", 0.0))

    return PricingRow(
        input_per_1k=input_per_1k,
        output_per_1k=output_per_1k,
        embedding_per_1k=embedding_per_1k,
    )

## ============================================================
## TOKENIZATION AND CHUNKING
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

def iter_char_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
        Split text into overlapping character chunks

        High-level workflow:
            1) Validate chunk_size and chunk_overlap
            2) Use a sliding window with step = chunk_size - chunk_overlap
            3) Keep non-empty chunks only

        Args:
            text: Input text
            chunk_size: Chunk size in characters
            chunk_overlap: Overlap size in characters

        Returns:
            List of chunk strings
    """
    
    if chunk_size <= 0:
        log_and_raise_validation_error(
            reason=f"chunk_size must be > 0, got {chunk_size}",
            context="chunking",
        )

    if chunk_overlap < 0:
        log_and_raise_validation_error(
            reason=f"chunk_overlap must be >= 0, got {chunk_overlap}",
            context="chunking",
        )

    if chunk_overlap >= chunk_size:
        log_and_raise_validation_error(
            reason="chunk_overlap must be < chunk_size",
            context="chunking",
        )

    ## Early exit for short text
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = chunk_size - chunk_overlap

    ## Build overlapping windows
    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        ## Keep non-empty chunks to avoid inflated token counts
        if chunk.strip() != "":
            chunks.append(chunk)

        if end >= len(text):
            break

    return chunks

def estimate_input_tokens_for_embeddings(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """
        Estimate total input tokens for embeddings by chunking text

        Design choice:
            - This MVP uses a heuristic tokenizer (approximate_token_count)
            - Later we can replace by provider/model tokenizers without changing API

        Args:
            text: Raw input text
            chunk_size: Chunk size in characters
            chunk_overlap: Chunk overlap in characters

        Returns:
            Total estimated tokens across all chunks
    """
    
    ## Chunk text and sum approximate tokens
    chunks = iter_char_chunks(text, chunk_size, chunk_overlap)
    
    return sum(approximate_token_count(c) for c in chunks)

## ============================================================
## COST COMPUTATION
## ============================================================
def cost_usd_for_chat(
    pricing: PricingRow,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
        Compute chat completion cost in USD

        High-level workflow:
            1) Convert tokens to 1k units
            2) Multiply by pricing rates
            3) Sum input + output costs

        Args:
            pricing: PricingRow
            input_tokens: Estimated input tokens
            output_tokens: Assumed output tokens

        Returns:
            Estimated USD cost
    """
    
    ## Convert tokens to 1k units
    in_units = float(input_tokens) / 1000.0
    out_units = float(output_tokens) / 1000.0

    ## Cost = input component + output component
    return (in_units * pricing.input_per_1k) + (out_units * pricing.output_per_1k)

def cost_usd_for_embeddings(
    pricing: PricingRow,
    input_tokens: int,
) -> float:
    """
        Compute embeddings cost in USD

        High-level workflow:
            1) Convert tokens to 1k units
            2) Multiply by embeddings pricing rate

        Args:
            pricing: PricingRow
            input_tokens: Estimated embedding tokens

        Returns:
            Estimated USD cost
    """
    
    ## Convert tokens to 1k units
    in_units = float(input_tokens) / 1000.0
    
    return in_units * pricing.embedding_per_1k