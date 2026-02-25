'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Evaluation orchestration: compute requested metrics per pair, aggregate results, optional cosine via provided embeddings."
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.core.errors import log_and_raise_validation_error
from src.utils.utils import as_float, mean
from src.utils.logging_utils import get_logger, log_execution_time_and_path
from src.utils.metrics_utils import (
    bertscore,
    bleu,
    contains_ref,
    cosine_similarity,
    exact_match,
    f1_token,
    jaccard,
    rouge_l,
)

LOGGER = get_logger(__name__)

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(frozen=True)
class EvaluationResult:
    """
        Evaluation result for a single reference/prediction pair

        Args:
            metrics: Computed metrics
            warnings: Optional warnings for this pair
    """

    metrics: Dict[str, float]
    warnings: List[str]

## ============================================================
## PUBLIC API
## ============================================================
@log_execution_time_and_path
def evaluate_pair(
    reference: str,
    prediction: str,
    metrics: List[str],
    enable_bertscore: bool = False,
    bert_lang: str = "en",
    use_embeddings: bool = False,
    embed_vectors: Optional[Tuple[List[float], List[float]]] = None,
) -> EvaluationResult:
    """
        Evaluate a single (reference, prediction) pair

        Notes:
            - If cosine is requested, embed_vectors must be provided

        Args:
            reference: Reference string
            prediction: Prediction string
            metrics: Metrics list to compute
            enable_bertscore: Allow heavy BERTScore computation
            bert_lang: Language for BERTScore
            use_embeddings: Whether to compute cosine similarity
            embed_vectors: Optional (ref_vec, pred_vec)

        Returns:
            EvaluationResult
    """
    
    ## Validate inputs
    if reference is None or prediction is None:
        log_and_raise_validation_error(
            reason="reference and prediction must not be None",
            context="evaluate_pair",
        )

    warnings: List[str] = []
    out: Dict[str, float] = {}

    ## CORE TEXT METRICS
    if "exact_match" in metrics:
        out["exact_match"] = exact_match(reference, prediction)

    if "contains_ref" in metrics:
        out["contains_ref"] = contains_ref(reference, prediction)

    if "f1_token" in metrics:
        out["f1_token"] = f1_token(reference, prediction)

    if "jaccard" in metrics:
        out["jaccard"] = jaccard(reference, prediction)

    ## ADVANCED TEXT METRICS (OPTIONAL DEPENDENCIES)
    if "rouge" in metrics:
        value, warn = rouge_l(reference, prediction)
        out["rouge"] = _as_float(value)
        if warn:
            warnings.append(warn)

    if "bleu" in metrics:
        value, warn = bleu(reference, prediction)
        out["bleu"] = _as_float(value)
        if warn:
            warnings.append(warn)

    if "bertscore" in metrics:
        if not enable_bertscore:
            out["bertscore"] = 0.0
            warnings.append("bertscore disabled (enable_bertscore=false)")
        else:
            value, warn = bertscore(reference, prediction, lang=bert_lang)
            out["bertscore"] = _as_float(value)
            if warn:
                warnings.append(warn)

    ## VECTOR METRIC (OPTIONAL)
    if "cosine_embedding" in metrics or use_embeddings:
        if embed_vectors is None:
            out["cosine_embedding"] = 0.0
            warnings.append("cosine_embedding requested but embed_vectors is missing")
        else:
            ref_vec, pred_vec = embed_vectors
            out["cosine_embedding"] = float(cosine_similarity(ref_vec, pred_vec))

    return EvaluationResult(metrics=out, warnings=warnings)

@log_execution_time_and_path
def aggregate_mean_metrics(items: List[EvaluationResult]) -> Dict[str, float]:
    """
        Aggregate mean metrics across multiple EvaluationResult items

        Args:
            items: List of evaluation results

        Returns:
            Dictionary of mean metrics
    """
    
    if not items:
        return {}

    ## Collect all metric keys
    keys: set[str] = set()
    for it in items:
        keys.update(it.metrics.keys())

    ## Compute mean value per metric key
    agg: Dict[str, float] = {}
    for k in sorted(keys):
        values = [float(it.metrics.get(k, 0.0)) for it in items]
        agg[k] = _mean(values)

    return agg

@log_execution_time_and_path
def evaluate_batch(
    references: List[str],
    predictions: List[str],
    metrics: List[str],
    enable_bertscore: bool = False,
    bert_lang: str = "en",
) -> Tuple[List[EvaluationResult], Dict[str, float], List[str]]:
    """
        Evaluate multiple reference/prediction pairs

        Args:
            references: List of references
            predictions: List of predictions
            metrics: Metrics to compute
            enable_bertscore: Allow heavy BERTScore computation
            bert_lang: Language for BERTScore

        Returns:
            Tuple(items, aggregate, warnings)
    """
    
    ## Validate batch lengths
    if len(references) != len(predictions):
        log_and_raise_validation_error(
            reason="references and predictions must have the same length",
            context="evaluate_batch",
        )

    items: List[EvaluationResult] = []
    warnings: List[str] = []

    ## Evaluate each pair
    for idx, (ref, pred) in enumerate(zip(references, predictions)):
        ## Compute per-item metrics
        result = evaluate_pair(
            reference=ref,
            prediction=pred,
            metrics=metrics,
            enable_bertscore=enable_bertscore,
            bert_lang=bert_lang,
        )
        items.append(result)

        ## Collect per-item warnings
        if result.warnings:
            warnings.append(f"index={idx} | " + " ; ".join(result.warnings))

    ## Compute aggregate means
    aggregate = aggregate_mean_metrics(items)

    LOGGER.info(
        "Evaluation completed | n_items=%s | metrics=%s",
        len(items),
        ",".join(metrics),
    )

    return items, aggregate, warnings
    