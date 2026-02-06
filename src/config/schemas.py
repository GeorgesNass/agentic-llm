'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Dataclasses and validation schemas for local-quantization configuration."
'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


## Allowed backend identifiers
BackendName = Literal[
    "gguf",
    "awq",
    "gptq",
    "bnb_nf4",
    "onnx",
]


## Allowed pipeline modes
PipelineMode = Literal[
    "quantize",
    "export",
    "benchmark",
    "full",
]


@dataclass(frozen=True)
class ModelConfig:
    """
        Configuration describing the base model and optional adapters

        Args:
            model_name_or_path: Hugging Face repo id or local model path
            adapter_path: Optional LoRA adapter path
            revision: Optional model revision or branch
    """

    model_name_or_path: str
    adapter_path: Optional[Path] = None
    revision: Optional[str] = None


@dataclass(frozen=True)
class QuantizationConfig:
    """
        Configuration describing the quantization strategy

        Args:
            backend: Quantization backend identifier
            bits: Target bit-width (4 or 8 depending on backend)
            group_size: Optional group size for weight-only methods
            calibration_dataset: Optional path to calibration dataset
    """

    backend: BackendName
    bits: int
    group_size: Optional[int] = None
    calibration_dataset: Optional[Path] = None


@dataclass(frozen=True)
class ExportConfig:
    """
        Configuration describing export options for quantized artifacts

        Args:
            output_dir: Directory where exported artifacts are written
            overwrite: Whether to overwrite existing artifacts
    """

    output_dir: Path
    overwrite: bool = False


@dataclass(frozen=True)
class BenchmarkConfig:
    """
        Configuration describing benchmarking options

        Args:
            prompts_path: Path to prompt file or dataset
            max_tokens: Maximum number of tokens to generate
            runs: Number of benchmark runs
    """

    prompts_path: Path
    max_tokens: int = 256
    runs: int = 5


@dataclass(frozen=True)
class PipelineConfig:
    """
        Top-level configuration for the local-quantization pipeline

        Args:
            mode: Pipeline execution mode
            model: Model configuration
            quantization: Quantization configuration
            export: Optional export configuration
            benchmark: Optional benchmark configuration
    """

    mode: PipelineMode
    model: ModelConfig
    quantization: QuantizationConfig
    export: Optional[ExportConfig] = None
    benchmark: Optional[BenchmarkConfig] = None
