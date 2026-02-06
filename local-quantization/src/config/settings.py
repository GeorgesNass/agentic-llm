'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Environment-based settings loader and PipelineConfig builder."
'''

from __future__ import annotations

from pathlib import Path
from typing import Optional, cast

from src.config.schemas import (
    BackendName,
    BenchmarkConfig,
    ExportConfig,
    ModelConfig,
    PipelineConfig,
    PipelineMode,
    QuantizationConfig,
)
from src.core.errors import (
    log_and_raise_missing_env,
    log_and_raise_missing_path,
    log_and_raise_pipeline_error,
)
from src.utils.utils import get_env_bool, get_env_int, get_env_str


def _get_required_env(key: str) -> str:
    """
        Read a required environment variable or raise

        Args:
            key: Environment variable name

        Returns:
            The resolved value
    """
    value = get_env_str(key, "")
    if value == "":
        log_and_raise_missing_env(key)
    return value


def _resolve_optional_path(raw: Optional[str]) -> Optional[Path]:
    """
        Resolve an optional string into a Path if provided

        Args:
            raw: Raw string value

        Returns:
            A Path if defined, otherwise None
    """
    if raw is None or raw.strip() == "":
        return None

    path = Path(raw).expanduser().resolve()
    if not path.exists():
        log_and_raise_missing_path(path, context="optional_path")
    return path


def _parse_pipeline_mode(raw: str) -> PipelineMode:
    """
        Parse and validate PIPELINE_MODE

        Args:
            raw: Raw env value

        Returns:
            A validated PipelineMode
    """
    normalized = raw.strip().lower()
    allowed = {"quantize", "export", "benchmark", "full"}

    if normalized not in allowed:
        log_and_raise_pipeline_error(
            step="settings_validation",
            reason=f"Invalid PIPELINE_MODE={raw} | allowed={sorted(list(allowed))}",
        )

    return cast(PipelineMode, normalized)


def _parse_backend_name(raw: str) -> BackendName:
    """
        Parse and validate QUANT_BACKEND

        Args:
            raw: Raw env value

        Returns:
            A validated BackendName
    """
    normalized = raw.strip().lower()
    allowed = {"gguf", "awq", "gptq", "bnb_nf4", "onnx"}

    if normalized not in allowed:
        log_and_raise_pipeline_error(
            step="settings_validation",
            reason=f"Invalid QUANT_BACKEND={raw} | allowed={sorted(list(allowed))}",
        )

    return cast(BackendName, normalized)


def _parse_quant_bits(raw_bits: int, backend: BackendName) -> int:
    """
        Parse and validate QUANT_BITS with backend-aware constraints

        Args:
            raw_bits: Raw bit-width integer
            backend: Selected backend

        Returns:
            A validated bit-width integer
    """
    if raw_bits not in {4, 8}:
        log_and_raise_pipeline_error(
            step="settings_validation",
            reason=f"Invalid QUANT_BITS={raw_bits} | allowed=[4, 8]",
        )

    ## Conservative constraints (can be relaxed later)
    if backend in {"awq", "gptq", "bnb_nf4"} and raw_bits != 4:
        log_and_raise_pipeline_error(
            step="settings_validation",
            reason=f"Backend={backend} only supports QUANT_BITS=4 in this project setup",
        )

    if backend == "gguf" and raw_bits not in {4, 8}:
        log_and_raise_pipeline_error(
            step="settings_validation",
            reason=f"Backend=gguf supports QUANT_BITS in [4, 8], got {raw_bits}",
        )

    if backend == "onnx" and raw_bits not in {4, 8}:
        log_and_raise_pipeline_error(
            step="settings_validation",
            reason=f"Backend=onnx supports QUANT_BITS in [4, 8], got {raw_bits}",
        )

    return raw_bits


def build_pipeline_config() -> PipelineConfig:
    """
        Build a PipelineConfig from environment variables

        Environment variables:
            - PIPELINE_MODE
            - MODEL_NAME_OR_PATH
            - MODEL_REVISION
            - ADAPTER_PATH
            - QUANT_BACKEND
            - QUANT_BITS
            - QUANT_GROUP_SIZE
            - CALIBRATION_DATASET
            - EXPORT_OUTPUT_DIR
            - EXPORT_OVERWRITE
            - BENCHMARK_PROMPTS
            - BENCHMARK_MAX_TOKENS
            - BENCHMARK_RUNS

        Returns:
            A validated PipelineConfig instance
    """
    ## Pipeline mode
    mode = _parse_pipeline_mode(_get_required_env("PIPELINE_MODE"))

    ## Model config
    model = ModelConfig(
        model_name_or_path=_get_required_env("MODEL_NAME_OR_PATH"),
        adapter_path=_resolve_optional_path(get_env_str("ADAPTER_PATH", "")),
        revision=get_env_str("MODEL_REVISION", "") or None,
    )

    ## Quantization config
    backend = _parse_backend_name(_get_required_env("QUANT_BACKEND"))
    bits = _parse_quant_bits(get_env_int("QUANT_BITS", 4), backend)

    quantization = QuantizationConfig(
        backend=backend,
        bits=bits,
        group_size=get_env_int("QUANT_GROUP_SIZE", 0) or None,
        calibration_dataset=_resolve_optional_path(
            get_env_str("CALIBRATION_DATASET", "")
        ),
    )

    ## Export config (optional)
    export_cfg: Optional[ExportConfig] = None
    export_dir = get_env_str("EXPORT_OUTPUT_DIR", "")
    if export_dir != "":
        export_cfg = ExportConfig(
            output_dir=Path(export_dir).expanduser().resolve(),
            overwrite=get_env_bool("EXPORT_OVERWRITE", False),
        )

    ## Benchmark config (optional)
    bench_cfg: Optional[BenchmarkConfig] = None
    bench_prompts = get_env_str("BENCHMARK_PROMPTS", "")
    if bench_prompts != "":
        bench_cfg = BenchmarkConfig(
            prompts_path=Path(bench_prompts).expanduser().resolve(),
            max_tokens=get_env_int("BENCHMARK_MAX_TOKENS", 256),
            runs=get_env_int("BENCHMARK_RUNS", 5),
        )

    return PipelineConfig(
        mode=mode,
        model=model,
        quantization=quantization,
        export=export_cfg,
        benchmark=bench_cfg,
    )
