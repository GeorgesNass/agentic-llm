'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Inference runners and benchmarking logic for quantized models."
'''

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from src.config.schemas import PipelineConfig
from src.core.errors import (
    BenchmarkError,
    log_and_raise_missing_path,
    log_and_raise_pipeline_error,
)
from src.utils.logging_utils import get_logger

## Initialize module-level logger
LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class BenchmarkResult:
    """
        Benchmark result container

        Args:
            runs: Number of benchmark runs executed
            prompts_count: Number of prompts used per run
            avg_seconds: Average run duration in seconds
            p50_seconds: Median run duration in seconds
            p95_seconds: P95 run duration in seconds
            min_seconds: Minimum run duration in seconds
            max_seconds: Maximum run duration in seconds
    """

    runs: int
    prompts_count: int
    avg_seconds: float
    p50_seconds: float
    p95_seconds: float
    min_seconds: float
    max_seconds: float


def _load_prompts(prompts_path: Path) -> List[str]:
    """
        Load benchmark prompts from a text file

        Format:
            - One prompt per line
            - Empty lines are ignored

        Args:
            prompts_path: Path to prompt file

        Returns:
            List of prompt strings
    """
    if not prompts_path.exists():
        log_and_raise_missing_path(prompts_path, context="benchmark_prompts")

    lines = prompts_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _percentile(values: List[float], pct: float) -> float:
    """
        Compute a percentile from a list of floats

        Args:
            values: List of values
            pct: Percentile between 0 and 100

        Returns:
            Percentile value
    """
    if not values:
        return 0.0

    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)

    if f == c:
        return sorted_vals[f]

    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def _run_placeholder_inference(prompts: List[str]) -> None:
    """
        Placeholder inference runner

        Design choice:
            - This project is built incrementally
            - Until backend runners are implemented, we simulate a minimal workload
            - Prompts are still iterated to preserve realistic loop structure

        Args:
            prompts: Prompts to iterate over

        Returns:
            None
    """
    ## Minimal simulated work to avoid near-zero timings
    for _prompt in prompts:
        _ = hash(_prompt)


def _compute_stats(latencies: List[float]) -> Tuple[float, float, float, float, float]:
    """
        Compute benchmark summary statistics

        Args:
            latencies: Run durations in seconds

        Returns:
            Tuple(avg, p50, p95, min, max)
    """
    avg_val = statistics.mean(latencies)
    p50_val = statistics.median(latencies)
    p95_val = _percentile(latencies, 95.0)
    min_val = min(latencies)
    max_val = max(latencies)
    return avg_val, p50_val, p95_val, min_val, max_val


def run_benchmark(config: PipelineConfig) -> BenchmarkResult:
    """
        Run inference benchmarks for a quantized model

        High-level workflow:
            1) Load benchmark prompts
            2) Run multiple inference passes
            3) Measure latency
            4) Return and log summary statistics

        Args:
            config: Validated pipeline configuration

        Returns:
            BenchmarkResult summary
    """
    if config.benchmark is None:
        log_and_raise_pipeline_error(
            step="benchmark",
            reason="Benchmark configuration is missing",
        )

    prompts = _load_prompts(config.benchmark.prompts_path)
    if not prompts:
        log_and_raise_pipeline_error(
            step="benchmark",
            reason="Benchmark prompts file is empty",
        )

    LOGGER.info("Loaded %d benchmark prompts", len(prompts))

    latencies: List[float] = []

    try:
        for run_idx in range(config.benchmark.runs):
            start_time = time.perf_counter()

            ## Placeholder runner until backend-specific runners are implemented
            _run_placeholder_inference(prompts)

            elapsed = time.perf_counter() - start_time
            latencies.append(elapsed)
            LOGGER.info(
                "Benchmark run %d/%d completed in %.6f s",
                run_idx + 1,
                config.benchmark.runs,
                elapsed,
            )

    except Exception as exc:
        raise BenchmarkError(str(exc)) from exc

    if not latencies:
        raise BenchmarkError("No benchmark latency recorded")

    avg_latency, p50_latency, p95_latency, min_latency, max_latency = _compute_stats(
        latencies
    )

    result = BenchmarkResult(
        runs=len(latencies),
        prompts_count=len(prompts),
        avg_seconds=avg_latency,
        p50_seconds=p50_latency,
        p95_seconds=p95_latency,
        min_seconds=min_latency,
        max_seconds=max_latency,
    )

    LOGGER.info(
        "Benchmark summary | runs=%d | prompts=%d | avg=%.6f s | p50=%.6f s | p95=%.6f s | min=%.6f s | max=%.6f s",
        result.runs,
        result.prompts_count,
        result.avg_seconds,
        result.p50_seconds,
        result.p95_seconds,
        result.min_seconds,
        result.max_seconds,
    )

    return result
