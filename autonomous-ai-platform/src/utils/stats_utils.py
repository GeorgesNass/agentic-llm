'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Statistical utilities for autonomous-ai-platform: mean/std, IQR, extremes, winsorization."
'''

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

try:
    from src.core.errors import ValidationError, DataError
except Exception:
    ValidationError = ValueError
    DataError = RuntimeError

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("stats_utils")

def compute_mean_std(series: pd.Series) -> Tuple[float, float]:
    """
        Compute mean and standard deviation

        High-level workflow:
            1) Validate input
            2) Convert to numeric
            3) Compute statistics

        Args:
            series: Numerical pandas Series

        Returns:
            Tuple (mean, std)
    """

    try:
        ## validate input
        if series is None or len(series) == 0:
            logger.error("Empty series provided")
            raise ValidationError("Series is empty")

        ## convert to float
        s = series.astype(float)

        ## compute statistics
        mean = float(s.mean())
        std = float(s.std(ddof=0))

        logger.debug(f"Mean={mean}, Std={std}")

        return mean, std

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Error computing mean/std: {exc}")
        raise DataError("Failed to compute mean/std") from exc

def compute_iqr_bounds(
    series: pd.Series,
    multiplier: float = 1.5,
) -> Tuple[float, float]:
    """
        Compute IQR bounds

        High-level workflow:
            1) Validate input
            2) Compute quartiles
            3) Compute bounds

        Args:
            series: Numerical pandas Series
            multiplier: IQR multiplier

        Returns:
            Tuple (lower_bound, upper_bound)
    """

    try:
        ## validate input
        if series is None or len(series) == 0:
            logger.error("Empty series for IQR")
            raise ValidationError("Series is empty")

        ## convert to float
        s = series.astype(float)

        ## compute quartiles
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1

        ## compute bounds
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        logger.debug(f"IQR bounds: {lower}, {upper}")

        return lower, upper

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Error computing IQR: {exc}")
        raise DataError("Failed to compute IQR") from exc

def compute_extremes(series: pd.Series) -> Tuple[float, float]:
    """
        Compute min and max values

        High-level workflow:
            1) Validate input
            2) Convert to numeric
            3) Compute extremes

        Args:
            series: Numerical pandas Series

        Returns:
            Tuple (min, max)
    """

    try:
        ## validate input
        if series is None or len(series) == 0:
            logger.error("Empty series for extremes")
            raise ValidationError("Series is empty")

        ## convert to float
        s = series.astype(float)

        ## compute extremes
        min_val = float(s.min())
        max_val = float(s.max())

        logger.debug(f"Extremes: min={min_val}, max={max_val}")

        return min_val, max_val

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Error computing extremes: {exc}")
        raise DataError("Failed to compute extremes") from exc

def winsorize_series(
    series: pd.Series,
    lower: float,
    upper: float,
) -> pd.Series:
    """
        Apply winsorization

        High-level workflow:
            1) Validate input
            2) Clip values within bounds

        Args:
            series: Numerical pandas Series
            lower: Lower bound
            upper: Upper bound

        Returns:
            Winsorized series
    """

    try:
        ## validate input
        if series is None or len(series) == 0:
            logger.error("Empty series for winsorization")
            raise ValidationError("Series is empty")

        ## convert to float
        s = series.astype(float)

        ## clip values
        clipped = s.clip(lower=lower, upper=upper)

        logger.debug("Winsorization applied")

        return clipped

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Error during winsorization: {exc}")
        raise DataError("Winsorization failed") from exc