'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "End-to-end orchestration layer: cost simulation, embeddings execution, chat completion and optional evaluation/export."
'''

from __future__ import annotations

from typing import Any, Optional

from src.llm.completion import run_chat_completion
from src.llm.embeddings import run_embeddings
from src.llm.costing import simulate_cost
from src.llm.evaluation import evaluate_batch
from src.utils.logging_utils import get_logger, log_execution_time_and_path
from src.utils.utils import export_dicts_to_csv

LOGGER = get_logger(__name__)

## ============================================================
## COST → EMBEDDINGS → COMPLETION PIPELINE
## ============================================================
@log_execution_time_and_path
def run_full_pipeline(
    mode: str,
    providers: list[str],
    model: Optional[str],
    messages: Optional[list[dict[str, Any]]],
    text: Optional[str],
    cost_only: bool,
    models_catalog: dict[str, Any],
    pricing_catalog: dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    expected_output_tokens: int = 512,
    export_path: Optional[str] = None,
) -> dict[str, Any]:
    """
        Run the end-to-end LLM pipeline

        High-level workflow:
            1) Run cost simulation (mandatory first step)
            2) If cost_only=True → return immediately
            3) If mode="embeddings" → call embeddings provider
            4) If mode="chat" → call completion provider
            5) Optionally export results

        Design choice:
            - Cost simulation always runs first for transparency
            - Embeddings and chat paths are mutually exclusive

        Args:
            mode: "chat" or "embeddings"
            providers: List of providers
            model: Optional model override
            messages: Chat messages (for chat mode)
            text: Raw text (for embeddings or cost estimation)
            cost_only: Stop after cost simulation
            models_catalog: Models configuration dict
            pricing_catalog: Pricing configuration dict
            temperature: Sampling temperature
            max_tokens: Max output tokens
            top_p: Top-p sampling
            chunk_size: Embeddings chunk size
            chunk_overlap: Embeddings chunk overlap
            expected_output_tokens: Assumed output tokens for cost
            export_path: Optional CSV export path

        Returns:
            Dictionary containing cost results and optional execution outputs
    """

    ## --------------------------------------------------------
    ## STEP 1: Cost simulation
    ## --------------------------------------------------------
    cost_result = simulate_cost(
        mode=mode,
        providers=providers,
        requested_model=model,
        text=text,
        path=None,
        recursive=False,
        max_chars_per_file=200_000,
        expected_output_tokens=expected_output_tokens,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        models_catalog=models_catalog,
        pricing_catalog=pricing_catalog,
        include_per_file=False,
    )

    if cost_only:
        LOGGER.info("Pipeline executed in cost_only mode")
        return {"cost": cost_result}

    execution_results: list[dict[str, Any]] = []

    ## --------------------------------------------------------
    ## STEP 2: Execution per provider
    ## --------------------------------------------------------
    for provider in providers:

        if mode == "embeddings":

            ## Run embeddings
            emb_response = run_embeddings(
                provider=provider,
                model=model or "",
                inp=text or "",
                extra={},
            )

            execution_results.append(
                {
                    "provider": provider,
                    "type": "embeddings",
                    "model": emb_response.model,
                    "n_vectors": len(emb_response.vectors),
                    "usage": emb_response.usage,
                }
            )

        elif mode == "chat":

            ## Run chat completion
            completion_response = run_chat_completion(
                provider=provider,
                model=model or "",
                messages=messages or [],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=False,
                extra={},
            )

            execution_results.append(
                {
                    "provider": provider,
                    "type": "chat",
                    "model": completion_response.model,
                    "text": completion_response.text,
                    "usage": completion_response.usage,
                }
            )

        else:
            raise ValueError(f"Unsupported mode={mode}")

    ## --------------------------------------------------------
    ## STEP 3: Optional export
    ## --------------------------------------------------------

    if export_path:
        export_dicts_to_csv(execution_results, export_path)
        LOGGER.info("Pipeline results exported to %s", export_path)

    return {
        "cost": cost_result,
        "execution": execution_results,
    }

## ============================================================
## EVALUATION WRAPPER
## ============================================================
@log_execution_time_and_path
def run_evaluation(
    predictions: list[str],
    references: list[str],
) -> dict[str, Any]:
    """
        Run evaluation metrics on completion outputs

        High-level workflow:
            1) Compare predictions with references
            2) Compute metrics (BLEU, ROUGE-L, etc.)
            3) Return aggregated results

        Args:
            predictions: List of generated texts
            references: List of ground-truth texts

        Returns:
            Dictionary of evaluation metrics
    """

    return evaluate_batch(predictions, references)