'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Text and vector evaluation metrics utilities (EM, contains, F1, Jaccard, ROUGE-L, BLEU, BERTScore, cosine)."
'''

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger
from src.utils.utils import normalize_text_basic

LOGGER = get_logger(__name__)

## ============================================================
## OPTIONAL DEPENDENCIES
## ============================================================

## Light dependencies
try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None

try:
    import sacrebleu
except Exception:
    sacrebleu = None

## Heavy dependency
try:
    from bert_score import score as bert_score_fn
except Exception:
    bert_score_fn = None

## ============================================================
## INTERNAL HELPERS
## ============================================================
def safe_div(num: float, den: float) -> float:
    """
        Safe division helper

        Args:
            num: Numerator
            den: Denominator

        Returns:
            num / den or 0.0 if den is zero
    """
    
    ## Prevent division by zero
    if den == 0.0:
        return 0.0

    return float(num) / float(den)

def tokenize_simple(text: str) -> List[str]:
    """
        Tokenize text using a simple alphanumeric tokenizer

        High-level workflow:
            1) Normalize whitespace and lowercase
            2) Extract alphanumeric tokens with regex

        Args:
            text: Raw text

        Returns:
            List of tokens
    """
    
    ## Normalize basic text (lowercase, strip, collapse spaces)
    cleaned = normalize_text_basic(text)

    ## Extract tokens (letters + digits)
    return re.findall(r"[a-z0-9]+", cleaned)

## ============================================================
## CORE TEXT METRICS
## ============================================================
def exact_match(reference: str, prediction: str) -> float:
    """
        Compute exact match after basic normalization

        Args:
            reference: Reference text
            prediction: Predicted text

        Returns:
            1.0 if match else 0.0
    """
    
    ## Normalize both strings
    ref = normalize_text_basic(reference)
    pred = normalize_text_basic(prediction)

    return 1.0 if ref == pred else 0.0

def contains_ref(reference: str, prediction: str) -> float:
    """
        Check if normalized reference is contained in normalized prediction

        Args:
            reference: Reference text
            prediction: Predicted text

        Returns:
            1.0 if contained else 0.0
    """
    ## Normalize both strings
    ref = normalize_text_basic(reference)
    pred = normalize_text_basic(prediction)

    ## Avoid trivial empty reference
    if ref == "":
        return 0.0

    return 1.0 if ref in pred else 0.0

def f1_token(reference: str, prediction: str) -> float:
    """
        Compute token-level F1 score using multiset overlap

        High-level workflow:
            1) Tokenize reference and prediction
            2) Count token overlaps with multiset logic
            3) Compute precision, recall, and F1

        Args:
            reference: Reference text
            prediction: Predicted text

        Returns:
            Token F1 score in [0, 1]
    """
    
    ## Tokenize both texts
    ref_toks = tokenize_simple(reference)
    pred_toks = tokenize_simple(prediction)

    ## Handle edge cases
    if not ref_toks and not pred_toks:
        return 1.0

    if not ref_toks or not pred_toks:
        return 0.0

    ## Build reference multiset counts
    ref_counts: Dict[str, int] = {}
    for tok in ref_toks:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    ## Count overlap using multiset decrement
    overlap = 0
    for tok in pred_toks:
        if ref_counts.get(tok, 0) > 0:
            overlap += 1
            ref_counts[tok] -= 1

    ## Compute precision and recall
    precision = safe_div(float(overlap), float(len(pred_toks)))
    recall = safe_div(float(overlap), float(len(ref_toks)))

    ## Compute F1
    return safe_div(2.0 * precision * recall, precision + recall)

def jaccard(reference: str, prediction: str) -> float:
    """
        Compute Jaccard similarity over token sets

        Args:
            reference: Reference text
            prediction: Predicted text

        Returns:
            Jaccard score in [0, 1]
    """
    
    ## Tokenize then convert to sets
    ref_set = set(tokenize_simple(reference))
    pred_set = set(tokenize_simple(prediction))

    ## Handle edge cases
    if not ref_set and not pred_set:
        return 1.0

    if not ref_set or not pred_set:
        return 0.0

    ## Compute intersection and union
    inter = len(ref_set.intersection(pred_set))
    union = len(ref_set.union(pred_set))

    return safe_div(float(inter), float(union))

## ============================================================
## ADVANCED TEXT METRICS
## ============================================================
def rouge_l(reference: str, prediction: str) -> Tuple[Optional[float], Optional[str]]:
    """
        Compute ROUGE-L F-measure if rouge-score is installed

        Args:
            reference: Reference text
            prediction: Predicted text

        Returns:
            Tuple(score, warning)
    """
    
    ## Validate optional dependency
    if rouge_scorer is None:
        return None, "rouge-score not installed"

    ## Compute ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)

    return float(scores["rougeL"].fmeasure), None

def bleu(reference: str, prediction: str) -> Tuple[Optional[float], Optional[str]]:
    """
        Compute BLEU score using sacrebleu if installed

        Args:
            reference: Reference text
            prediction: Predicted text

        Returns:
            Tuple(score, warning)
    """
    
    ## Validate optional dependency
    if sacrebleu is None:
        return None, "sacrebleu not installed"

    ## Compute corpus BLEU then normalize to [0, 1]
    bleu_obj = sacrebleu.corpus_bleu([prediction], [[reference]])

    return float(bleu_obj.score) / 100.0, None

def bertscore(
    reference: str,
    prediction: str,
    lang: str = "en",
) -> Tuple[Optional[float], Optional[str]]:
    """
        Compute BERTScore F1 if bert-score is installed

        Notes:
            - Heavy dependency (torch + transformers)
            - Enable only when you want semantic matching beyond lexical overlap

        Args:
            reference: Reference text
            prediction: Predicted text
            lang: Language code (en, fr)

        Returns:
            Tuple(score, warning)
    """
    
    ## Validate optional dependency
    if bert_score_fn is None:
        return None, "bert-score not installed"

    ## Compute BERTScore F1
    try:
        _, _, f1 = bert_score_fn(
            cands=[prediction],
            refs=[reference],
            lang=lang,
            verbose=False,
        )
        return float(f1.mean().item()), None

    except Exception as exc:
        return None, f"bertscore failed: {str(exc)}"

## ============================================================
## VECTOR METRICS
## ============================================================
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
        Compute cosine similarity between two vectors

        Args:
            v1: Vector 1
            v2: Vector 2

        Returns:
            Cosine similarity in [-1, 1]
    """
    
    ## Validate vector shapes
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0

    ## Compute dot product and L2 norms
    dot = 0.0
    n1 = 0.0
    n2 = 0.0

    for a, b in zip(v1, v2):
        dot += float(a) * float(b)
        n1 += float(a) * float(a)
        n2 += float(b) * float(b)

    den = math.sqrt(n1) * math.sqrt(n2)

    return safe_div(dot, den)