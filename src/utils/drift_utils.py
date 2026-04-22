
'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Drift utilities for local-quantization: statistical tests and weight metrics."
'''

from __future__ import annotations

from typing import Dict, Tuple, Any

import json
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import ks_2samp
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

def compute_ks_test(ref: np.ndarray, cur: np.ndarray) -> Tuple[float, float]:
    """
        Compute Kolmogorov-Smirnov test for arrays

        High-level workflow:
            1) Flatten arrays
            2) Remove NaN values
            3) Handle empty inputs
            4) Compute KS statistic

        Args:
            ref: Reference array
            cur: Current array

        Returns:
            statistic, p_value
    """

    ## flatten arrays
    ref_flat = ref.flatten()
    cur_flat = cur.flatten()

    ## remove NaN
    ref_clean = ref_flat[~np.isnan(ref_flat)]
    cur_clean = cur_flat[~np.isnan(cur_flat)]

    ## handle empty
    if ref_clean.size == 0 or cur_clean.size == 0:
        return 0.0, 1.0

    ## compute test
    stat, p_value = ks_2samp(ref_clean, cur_clean)

    return float(stat), float(p_value)

def compute_weight_stats(weights: np.ndarray) -> Dict[str, float]:
    """
        Compute basic statistics on model weights

        High-level workflow:
            1) Flatten weights
            2) Remove NaN values
            3) Compute summary statistics

        Args:
            weights: Model weights array

        Returns:
            Dictionary with statistics
    """

    if weights.size == 0:
        return {}

    ## flatten
    flat = weights.flatten()

    ## remove NaN
    flat = flat[~np.isnan(flat)]

    if flat.size == 0:
        return {}

    ## compute stats
    stats = {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }

    return stats

def compute_metric_delta(ref: float, cur: float) -> float:
    """
        Compute relative difference between two metrics

        High-level workflow:
            1) Handle zero reference case
            2) Compute relative difference

        Args:
            ref: Reference metric
            cur: Current metric

        Returns:
            Relative difference
    """

    if ref == 0:
        return 0.0

    return float(abs(cur - ref) / ref)

def generate_drift_report(
    metrics: Dict[str, Any],
    output_dir: str = "reports",
) -> Dict[str, str]:
    """
        Generate drift report files

        High-level workflow:
            1) Create output directory
            2) Save JSON report
            3) Save HTML report

        Args:
            metrics: Drift metrics
            output_dir: Output directory

        Returns:
            Dict with report paths
    """

    ## create dir
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    ## JSON report
    json_path = path / "drift_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    ## HTML report
    html_path = path / "drift_report.html"
    html_content = "<html><body><h1>Quantization Drift Report</h1><pre>"
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
        Generate Evidently data drift report for quantization

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