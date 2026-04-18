'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Statistical utilities for local-finetuning: mean/std, IQR, extremes and winsorization."
'''

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

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

def compute_mean_std(arr: Union[np.ndarray, list]) -> Tuple[float, float]:
    """
        Compute mean and standard deviation

        High-level workflow:
            1) Validate input
            2) Convert to numpy
            3) Compute statistics

        Args:
            arr: Tensor-like input

        Returns:
            Tuple (mean, std)
    """

    try:
        ## validate input
        if arr is None or len(arr) == 0:
            raise ValidationError("Empty input data")

        ## convert to numpy
        data = np.asarray(arr, dtype=float).flatten()

        ## compute statistics
        mean = float(np.mean(data))
        std = float(np.std(data))

        return mean, std

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Error computing mean/std: {exc}")
        raise DataError("Failed to compute mean/std") from exc

def compute_iqr_bounds(
    arr: Union[np.ndarray, list],
    multiplier: float = 1.5,
) -> Tuple[float, float]:
    """
        Compute IQR bounds

        High-level workflow:
            1) Validate input
            2) Compute quartiles
            3) Compute bounds

        Args:
            arr: Tensor-like input
            multiplier: IQR multiplier

        Returns:
            Tuple (lower_bound, upper_bound)
    """

    try:
        ## validate input
        if arr is None or len(arr) == 0:
            raise ValidationError("Empty input data")

        ## convert to numpy
        data = np.asarray(arr, dtype=float).flatten()

        ## compute quartiles
        q1 = float(np.percentile(data, 25))
        q3 = float(np.percentile(data, 75))
        iqr = q3 - q1

        ## compute bounds
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        return lower, upper

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Error computing IQR: {exc}")
        raise DataError("Failed to compute IQR") from exc

def compute_extremes(arr: Union[np.ndarray, list]) -> Tuple[float, float]:
    """
        Compute min and max values

        Args:
            arr: Tensor-like input

        Returns:
            Tuple (min, max)
    """

    try:
        ## validate input
        if arr is None or len(arr) == 0:
            raise ValidationError("Empty input data")

        ## convert to numpy
        data = np.asarray(arr, dtype=float).flatten()

        ## compute extremes
        min_val = float(np.min(data))
        max_val = float(np.max(data))

        return min_val, max_val

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Error computing extremes: {exc}")
        raise DataError("Failed to compute extremes") from exc

def winsorize_array(
    arr: Union[np.ndarray, list],
    lower: float,
    upper: float,
) -> np.ndarray:
    """
        Apply winsorization

        High-level workflow:
            1) Validate input
            2) Clip values

        Args:
            arr: Tensor-like input
            lower: Lower bound
            upper: Upper bound

        Returns:
            Winsorized numpy array
    """

    try:
        ## validate input
        if arr is None or len(arr) == 0:
            raise ValidationError("Empty input data")

        ## convert to numpy
        data = np.asarray(arr, dtype=float)

        ## clip values
        clipped = np.clip(data, lower, upper)

        return clipped

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Error during winsorization: {exc}")
        raise DataError("Winsorization failed") from exc