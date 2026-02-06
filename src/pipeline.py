'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main pipeline orchestration: quantization, export, benchmarking, and full workflow."
'''

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.config.schemas import PipelineConfig
from src.core.errors import (
    LocalQuantizationError,
    PipelineError,
    log_and_raise_pipeline_error,
)
from src.core.export import run_export
from src.core.quantize import QuantizationResult, run_quantization
from src.inference.runners import BenchmarkResult, run_benchmark
from src.utils.logging_utils import get_logger

## Initialize module-level logger
LOGGER = get_logger(__name__)


def run_pipeline(config: PipelineConfig) -> None:
    """
        Execute the local-quantization pipeline based on configuration

        High-level workflow:
            - quantize
            - export
            - benchmark
            - full (quantize -> export -> benchmark)

        Args:
            config: Validated pipeline configuration

        Returns:
            None
    """
    quant_result: Optional[QuantizationResult] = None
    export_dir: Optional[Path] = None
    benchmark_result: Optional[BenchmarkResult] = None

    try:
        if config.mode == "quantize":
            LOGGER.info("Starting quantization step")
            quant_result = run_quantization(config)

        elif config.mode == "export":
            LOGGER.info("Starting export step")
            export_dir = run_export(config)

        elif config.mode == "benchmark":
            LOGGER.info("Starting benchmark step")
            benchmark_result = run_benchmark(config)

        elif config.mode == "full":
            LOGGER.info("Starting full pipeline")
            quant_result = run_quantization(config)

            if config.export is not None:
                export_dir = run_export(config)
                LOGGER.info("Export completed | export_dir=%s", export_dir)

            if config.benchmark is not None:
                benchmark_result = run_benchmark(config)
                LOGGER.info(
                    "Benchmark completed | avg_seconds=%.6f",
                    benchmark_result.avg_seconds,
                )

        else:
            log_and_raise_pipeline_error(
                step="pipeline_dispatch",
                reason=f"Unsupported pipeline mode: {config.mode}",
            )

        ## Final summary (best-effort)
        if quant_result is not None:
            LOGGER.info(
                "Pipeline summary | quant_backend=%s | bits=%d | calib_used=%s",
                quant_result.backend,
                quant_result.bits,
                quant_result.calibration_used,
            )

        if export_dir is not None:
            LOGGER.info("Pipeline summary | export_dir=%s", export_dir)

        if benchmark_result is not None:
            LOGGER.info(
                "Pipeline summary | bench_avg=%.6f s | bench_p50=%.6f s | bench_p95=%.6f s",
                benchmark_result.avg_seconds,
                benchmark_result.p50_seconds,
                benchmark_result.p95_seconds,
            )

    except PipelineError:
        ## Already logged at source
        raise

    except LocalQuantizationError:
        ## Already structured as project error
        raise

    except Exception as exc:
        ## Catch-all to enforce consistent error reporting
        log_and_raise_pipeline_error(
            step="pipeline_runtime",
            reason=str(exc),
        )
