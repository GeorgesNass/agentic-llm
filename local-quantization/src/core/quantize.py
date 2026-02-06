'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Quantization entry point dispatching to backend-specific implementations."
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.config.schemas import PipelineConfig
from src.core.backends import check_backend_available
from src.core.calibration import load_calibration_texts, sample_texts
from src.core.errors import log_and_raise_pipeline_error
from src.core.model_loader import (
    load_hf_model_and_tokenizer,
    resolve_model_source,
    validate_optional_adapter,
)
from src.utils.logging_utils import get_logger

## Initialize module-level logger
LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class QuantizationResult:
    """
        Quantization result container

        Design choice:
            - Quantization may produce different artifact types per backend
            - This container provides a stable interface for later steps

        Args:
            backend: Backend identifier
            bits: Quantization bit-width
            model_source: Resolved model source (HF id or local path)
            calibration_used: Whether calibration samples were used
            notes: Optional notes for debugging and reporting
    """

    backend: str
    bits: int
    model_source: str
    calibration_used: bool
    notes: str = ""


def _needs_calibration(backend: str) -> bool:
    """
        Determine whether a backend typically requires calibration texts

        Args:
            backend: Backend identifier

        Returns:
            True if calibration texts are expected, otherwise False
    """
    return backend in {"awq", "gptq", "onnx"}


def _get_calibration_samples(config: PipelineConfig) -> Optional[List[str]]:
    """
        Load and sample calibration texts if configured and required

        Args:
            config: Validated pipeline configuration

        Returns:
            A list of sampled texts if available, otherwise None
    """
    backend = config.quantization.backend
    if not _needs_calibration(backend):
        return None

    if config.quantization.calibration_dataset is None:
        LOGGER.warning(
            "Backend=%s usually requires calibration, but CALIBRATION_DATASET is missing",
            backend,
        )
        return None

    texts = load_calibration_texts(config.quantization.calibration_dataset)
    samples = sample_texts(texts=texts, max_samples=128, seed=42)

    LOGGER.info(
        "Loaded calibration dataset samples=%d (backend=%s)",
        len(samples),
        backend,
    )
    return samples


def _dispatch_placeholder(
    backend: str,
    bits: int,
    calibration_samples: Optional[List[str]],
) -> None:
    """
        Placeholder dispatch for backend-specific quantization

        Notes:
            - Real implementations will be added incrementally
            - This placeholder keeps the pipeline structure stable

        Args:
            backend: Backend identifier
            bits: Bit-width
            calibration_samples: Optional calibration samples

        Returns:
            None
    """
    LOGGER.info(
        "Quantization placeholder executed (backend=%s, bits=%s, calib=%s)",
        backend,
        bits,
        "yes" if calibration_samples else "no",
    )


def run_quantization(config: PipelineConfig) -> QuantizationResult:
    """
        Run the quantization step for the given pipeline configuration

        High-level workflow:
            1) Validate backend availability
            2) Validate model/adapters inputs
            3) Resolve model source (HF or local)
            4) Load model/tokenizer (HF path)
            5) Load calibration samples when required
            6) Dispatch to backend-specific quantization logic (placeholder for now)

        Args:
            config: Validated pipeline configuration

        Returns:
            QuantizationResult summary
    """
    backend = config.quantization.backend
    bits = config.quantization.bits

    LOGGER.info("Starting quantization using backend=%s", backend)

    ## Validate backend availability early
    check_backend_available(backend)

    ## Validate optional adapter path
    validate_optional_adapter(config.model.adapter_path)

    ## Resolve model source (HF repo id or local path)
    model_source = resolve_model_source(config.model.model_name_or_path)
    LOGGER.info("Resolved model source=%s", model_source)

    ## Load model + tokenizer for HF-based backends
    ## GGUF conversion may later use direct conversion tools, but HF loading helps validation
    try:
        model, tokenizer = load_hf_model_and_tokenizer(
            model_name_or_path=model_source,
            revision=config.model.revision,
        )
        LOGGER.info("Model and tokenizer loaded successfully")
        _ = (model, tokenizer)
    except Exception as exc:
        log_and_raise_pipeline_error(
            step="quantization_model_load",
            reason=str(exc),
        )

    ## Optional calibration samples
    calibration_samples = _get_calibration_samples(config)

    try:
        _dispatch_placeholder(
            backend=backend,
            bits=bits,
            calibration_samples=calibration_samples,
        )
    except Exception as exc:
        log_and_raise_pipeline_error(
            step="quantization_dispatch",
            reason=str(exc),
        )

    result = QuantizationResult(
        backend=backend,
        bits=bits,
        model_source=model_source,
        calibration_used=bool(calibration_samples),
        notes="Quantization placeholder only. Implement backend-specific logic next.",
    )

    LOGGER.info(
        "Quantization summary | backend=%s | bits=%d | calib_used=%s",
        result.backend,
        result.bits,
        result.calibration_used,
    )

    return result
