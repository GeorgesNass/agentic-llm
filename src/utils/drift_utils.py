'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Drift utilities for LLM proxy: statistical tests and LLM metrics extraction."
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
    """

    ## drop NaN
    ref_clean = ref.dropna()
    cur_clean = cur.dropna()

    if ref_clean.empty or cur_clean.empty:
        return 0.0, 1.0

    stat, p_value = ks_2samp(ref_clean, cur_clean)

    return float(stat), float(p_value)

def compute_chi2_test(ref: pd.Series, cur: pd.Series) -> Tuple[float, float]:
    """
        Compute Chi-square test
    """

    ref_counts = ref.value_counts()
    cur_counts = cur.value_counts()

    all_index = ref_counts.index.union(cur_counts.index)
    ref_aligned = ref_counts.reindex(all_index, fill_value=0)
    cur_aligned = cur_counts.reindex(all_index, fill_value=0)

    table = np.array([ref_aligned.values, cur_aligned.values])

    if table.sum() == 0:
        return 0.0, 1.0

    stat, p_value, _, _ = chi2_contingency(table)

    return float(stat), float(p_value)

def compute_llm_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
        Compute LLM-related features for drift detection

        Args:
            df: Input dataset

        Returns:
            DataFrame with LLM features
    """

    data: Dict[str, pd.Series] = {}

    ## latency
    if "response_time" in df.columns:
        data["response_time"] = df["response_time"].astype(float)

    ## tokens
    if "input_tokens" in df.columns:
        data["input_tokens"] = df["input_tokens"].astype(float)

    if "output_tokens" in df.columns:
        data["output_tokens"] = df["output_tokens"].astype(float)

    ## prompt length
    if "prompt" in df.columns:
        prompt_series = df["prompt"].fillna("").astype(str)
        data["prompt_length"] = prompt_series.str.len()

    ## response length
    if "response" in df.columns:
        response_series = df["response"].fillna("").astype(str)
        data["response_length"] = response_series.str.len()

    return pd.DataFrame(data)

def generate_drift_report(metrics: Dict[str, Any], output_dir: str = "reports") -> Dict[str, str]:
    """
        Generate drift report files
    """

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    ## JSON
    json_path = path / "drift_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    ## HTML
    html_path = path / "drift_report.html"
    html_content = "<html><body><h1>LLM Drift Report</h1><pre>"
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
        Generate Evidently data drift report for LLM proxy

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