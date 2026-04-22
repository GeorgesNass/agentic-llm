'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Drift utilities: statistical tests, text/embedding stats and Evidently reporting."
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
            1) Remove NaN values from both series
            2) Handle empty inputs safely
            3) Compute KS statistic and p-value

        Args:
            ref: Reference series
            cur: Current series

        Returns:
            statistic and p_value from KS test
    """

    ref_clean = ref.dropna()
    cur_clean = cur.dropna()

    if ref_clean.empty or cur_clean.empty:
        return 0.0, 1.0

    stat, p_value = ks_2samp(ref_clean, cur_clean)

    return float(stat), float(p_value)

def compute_chi2_test(ref: pd.Series, cur: pd.Series) -> Tuple[float, float]:
    """
        Compute Chi-square test

        High-level workflow:
            1) Compute value counts for both series
            2) Align categories between reference and current
            3) Build contingency table
            4) Compute Chi-square statistic and p-value

        Args:
            ref: Reference series
            cur: Current series

        Returns:
            statistic and p_value from Chi-square test
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

def compute_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
        Compute text-based features

        High-level workflow:
            1) Extract text column from dataset
            2) Compute text length
            3) Compute word count

        Args:
            df: Input dataset

        Returns:
            DataFrame with text statistics
    """

    data: Dict[str, pd.Series] = {}

    if "text" in df.columns:
        text_series = df["text"].fillna("").astype(str)

        data["text_length"] = text_series.str.len()
        data["text_word_count"] = text_series.str.split().apply(len)

    return pd.DataFrame(data)

def compute_embedding_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
        Compute embedding-based features

        High-level workflow:
            1) Extract embedding vectors
            2) Convert to numpy arrays
            3) Compute norm and mean value

        Args:
            df: Input dataset

        Returns:
            DataFrame with embedding statistics
    """

    data: Dict[str, pd.Series] = {}

    if "embedding" in df.columns:
        emb = df["embedding"].apply(
            lambda x: np.array(x) if isinstance(x, (list, tuple)) else np.array([])
        )

        data["embedding_norm"] = emb.apply(
            lambda x: np.linalg.norm(x) if x.size else 0.0
        )

        data["embedding_mean"] = emb.apply(
            lambda x: float(np.mean(x)) if x.size else 0.0
        )

    return pd.DataFrame(data)

def generate_evidently_report(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
    output_dir: str = "reports",
) -> Dict[str, str]:
    """
        Generate Evidently data drift report

        High-level workflow:
            1) Initialize Evidently report with DataDriftPreset
            2) Run drift analysis between reference and current datasets
            3) Save HTML report to output directory

        Args:
            df_ref: Reference dataset
            df_cur: Current dataset
            output_dir: Output directory

        Returns:
            Dictionary containing report path
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

def generate_drift_report(
    metrics: Dict[str, Any],
    output_dir: str = "reports",
) -> Dict[str, str]:
    """
        Generate custom drift report

        High-level workflow:
            1) Save metrics to JSON file
            2) Generate simple HTML report

        Args:
            metrics: Drift metrics dictionary
            output_dir: Output directory

        Returns:
            Dictionary containing report paths
    """

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    json_path = path / "drift_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    html_path = path / "drift_report.html"
    html_content = "<html><body><h1>Drift Report</h1><pre>"
    html_content += json.dumps(metrics, indent=2)
    html_content += "</pre></body></html>"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return {
        "report_json": str(json_path),
        "report_html": str(html_path),
    }