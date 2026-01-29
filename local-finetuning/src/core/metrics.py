"""
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Evaluation metrics for symptom normalization: exact match, hallucination rate, confusion analysis, and coverage."
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

def exact_match(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """
		Compute exact match accuracy

		Args:
			y_true: Ground-truth labels
			y_pred: Predicted labels

		Returns:
			Exact match accuracy in [0, 1]
    """
    
    if not y_true:
        return 0.0

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)

def hallucination_rate(
    y_pred: Sequence[str],
    allowed_labels: Optional[Iterable[str]] = None,
) -> float:
    """
		Compute hallucination rate (prediction not in allowed label set)

		Args:
			y_pred: Predicted labels
			allowed_labels: Optional iterable of allowed labels (CISP)

		Returns:
			Hallucination rate in [0, 1]
    """
    
    if not y_pred:
        return 0.0

    if allowed_labels is None:
        ## Without an allowed set, hallucination cannot be measured strictly
        return 0.0

    allowed_set = set(allowed_labels)
    hallucinated = sum(1 for yp in y_pred if yp not in allowed_set)
    return hallucinated / len(y_pred)

def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    """
		Build a confusion matrix

		Args:
			y_true: Ground-truth labels
			y_pred: Predicted labels

		Returns:
			Confusion matrix: true_label -> predicted_label -> count
    """
    
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for yt, yp in zip(y_true, y_pred):
        matrix[yt][yp] += 1

    return matrix

def most_confused_pairs(
    confusion: Dict[str, Dict[str, int]],
    min_count: int = 1,
) -> List[Tuple[str, str, int]]:
    """
		Extract most frequent confusion pairs (true != predicted)

		Args:
			confusion: Confusion matrix
			min_count: Minimum count threshold

		Returns:
			List of (true_label, predicted_label, count) sorted by count desc
    """
    
    pairs: List[Tuple[str, str, int]] = []

    for true_label, preds in confusion.items():
        for pred_label, count in preds.items():
            if true_label != pred_label and count >= min_count:
                pairs.append((true_label, pred_label, count))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs

def label_coverage(y_pred: Sequence[str]) -> Dict[str, int]:
    """
		Compute label prediction counts

		Args:
			y_pred: Predicted labels

		Returns:
			Dictionary label -> count
    """
    
    return dict(Counter(y_pred))

def missing_labels(
    all_labels: Iterable[str],
    predicted_labels: Iterable[str],
) -> List[str]:
    """
		Find labels never predicted by the model

		Args:
			all_labels: Full expected label set
			predicted_labels: Labels predicted by the model

		Returns:
		Sorted list of labels never predicted
    """
    
    all_set = set(all_labels)
    pred_set = set(predicted_labels)
    return sorted(all_set - pred_set)