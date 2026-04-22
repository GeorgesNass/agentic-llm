'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Drift utilities for local-finetuning: statistical tests and text metrics."
'''

from __future__ import annotations

from typing import Dict, Tuple, Any

import json
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import ks_2samp, chi2_contingency
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
    
from src.utils.logging_utils import get_logger

try:
    from src.core.errors import ValidationError, DataError
except Exception:
    ValidationError = ValueError
    DataError = RuntimeError

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("drift_utils")

def compute_ks_test(ref: pd.Series, cur: pd.Series) -> Tuple[float, float]:
    """
        Compute Kolmogorov-Smirnov test

        High-level workflow:
            1) Remove NaN values
            2) Handle empty inputs
            3) Compute KS statistic

        Args:
            ref: Reference series
            cur: Current series

        Returns:
            statistic, p_value
    """

    ## remove NaN
    ref_clean = ref.dropna()
    cur_clean = cur.dropna()

    ## handle empty
    if ref_clean.empty or cur_clean.empty:
        return 0.0, 1.0

    ## compute test
    stat, p_value = ks_2samp(ref_clean, cur_clean)

    return float(stat), float(p_value)

def compute_chi2_test(ref: pd.Series, cur: pd.Series) -> Tuple[float, float]:
    """
        Compute Chi-square test

        High-level workflow:
            1) Compute value counts
            2) Align categories
            3) Build contingency table
            4) Compute Chi-square

        Args:
            ref: Reference series
            cur: Current series

        Returns:
            statistic, p_value
    """

    ## value counts
    ref_counts = ref.value_counts()
    cur_counts = cur.value_counts()

    ## align categories
    all_index = ref_counts.index.union(cur_counts.index)
    ref_aligned = ref_counts.reindex(all_index, fill_value=0)
    cur_aligned = cur_counts.reindex(all_index, fill_value=0)

    ## contingency table
    table = np.array([ref_aligned.values, cur_aligned.values])

    ## handle empty
    if table.sum() == 0:
        return 0.0, 1.0

    ## compute test
    stat, p_value, _, _ = chi2_contingency(table)

    return float(stat), float(p_value)

def compute_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
        Compute text-based features for drift detection

        High-level workflow:
            1) Extract text columns
            2) Compute length metrics
            3) Compute token-like metrics

        Args:
            df: Input dataset

        Returns:
            DataFrame with text features
    """

    data: Dict[str, pd.Series] = {}

    ## TEXT COLUMN
    if "text" in df.columns:
        text_series = df["text"].fillna("").astype(str)

        ## length
        data["text_length"] = text_series.str.len()

        ## word count (simple token proxy)
        data["text_word_count"] = text_series.str.split().apply(len)

        ## unique tokens proxy
        data["text_unique_tokens"] = text_series.apply(
            lambda x: len(set(x.split()))
        )

    return pd.DataFrame(data)

def compute_embedding_stats(embeddings: np.ndarray) -> Dict[str, float]:
    """
        Compute embedding statistics

        High-level workflow:
            1) Flatten embeddings
            2) Remove NaN values
            3) Compute mean and std

        Args:
            embeddings: Embedding matrix

        Returns:
            Dictionary with stats
    """

    if embeddings.size == 0:
        return {}

    ## flatten
    flat = embeddings.flatten()

    ## remove NaN
    flat = flat[~np.isnan(flat)]

    if flat.size == 0:
        return {}

    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
    }

def generate_drift_report(
    metrics: Dict[str, Any],
    output_dir: str = "reports",
) -> Dict[str, str]:
    """
        Generate drift report files

        High-level workflow:
            1) Create directory
            2) Save JSON
            3) Save HTML

        Args:
            metrics: Drift metrics
            output_dir: Output directory

        Returns:
            Dict with report paths
    """

    ## create dir
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    ## JSON
    json_path = path / "drift_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    ## HTML
    html_path = path / "drift_report.html"
    html_content = "<html><body><h1>Finetuning Drift Report</h1><pre>"
    html_content += json.dumps(metrics, indent=2)
    html_content += "</pre></body></html>"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return {
        "report_json": str(json_path),
        "report_html": str(html_path),
    }
    
def generate_evidently_report(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
    output_dir: str = "reports",
) -> Dict[str, str]:
    """
        Generate Evidently data drift report

        High-level workflow:
            1) Initialize Evidently report
            2) Run drift analysis
            3) Save HTML report

        Args:
            df_ref: Reference dataset
            df_cur: Current dataset
            output_dir: Output directory

        Returns:
            Dictionary with report path
    """

    report = Report(metrics=[DataDriftPreset()])

    report.run(
        reference_data=df_ref,
        current_data=df_cur,
    )

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    html_path = path / "evidently_report.html"
    report.save_html(str(html_path))

    return {
        "evidently_html": str(html_path),
    }