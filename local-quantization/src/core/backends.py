'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Quantization backend registry and capability checks."
'''

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Dict, Optional

from src.core.errors import log_and_raise_backend_unavailable
from src.utils.logging_utils import get_logger
from src.utils.utils import get_env_str

## Initialize module-level logger
LOGGER = get_logger(__name__)


## Type alias for backend handlers
BackendHandler = Callable[..., None]


def _check_binary_exists(binary_name: str, hint: Optional[str] = None) -> None:
    """
        Check that a required system binary exists in PATH

        Args:
            binary_name: Name of the binary to check
            hint: Optional hint to help the user fix the issue

        Returns:
            None
    """
    if shutil.which(binary_name) is None:
        reason = "Required binary not found in PATH"
        if hint:
            reason = f"{reason} | hint={hint}"

        log_and_raise_backend_unavailable(
            backend_name=binary_name,
            reason=reason,
        )


def _check_python_module(module_name: str) -> None:
    """
        Check that a required Python module can be imported

        Args:
            module_name: Python module name

        Returns:
            None
    """
    try:
        __import__(module_name)
    except ImportError:
        log_and_raise_backend_unavailable(
            backend_name=module_name,
            reason="Required Python module not installed",
        )


def _check_optional_path_exists(raw_path: str, name: str) -> None:
    """
        Check that an optional path exists when provided

        Args:
            raw_path: Raw path string
            name: Friendly name for error messages

        Returns:
            None
    """
    if raw_path.strip() == "":
        return

    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        log_and_raise_backend_unavailable(
            backend_name=name,
            reason=f"Provided path does not exist: {path}",
        )


def _check_gguf_tooling() -> None:
    """
        Check GGUF conversion tooling availability

        Design choice:
            - GGUF conversion usually relies on llama.cpp tools
            - Binary names vary across installs, so we support two strategies:
                1) User provides LLAMA_CPP_BIN_DIR pointing to compiled binaries
                2) We fall back to PATH checks for common binary names

        Environment variables:
            - LLAMA_CPP_BIN_DIR: Optional directory containing llama.cpp binaries

        Args:
            None

        Returns:
            None
    """
    bin_dir = get_env_str("LLAMA_CPP_BIN_DIR", "")
    if bin_dir.strip() != "":
        _check_optional_path_exists(bin_dir, "LLAMA_CPP_BIN_DIR")
        LOGGER.info("GGUF tooling: using LLAMA_CPP_BIN_DIR=%s", bin_dir)
        return

    ## Common llama.cpp binary names depending on build/install method
    ## We check several and accept if at least one exists
    candidates = [
        "llama-cli",
        "llama-quantize",
        "quantize",
        "main",
    ]

    for binary in candidates:
        if shutil.which(binary) is not None:
            LOGGER.info("GGUF tooling: found binary in PATH: %s", binary)
            return

    log_and_raise_backend_unavailable(
        backend_name="gguf",
        reason=(
            "No llama.cpp binary found in PATH and LLAMA_CPP_BIN_DIR is not set | "
            "hint=build llama.cpp and export LLAMA_CPP_BIN_DIR or add binaries to PATH"
        ),
    )


def check_backend_available(backend: str) -> None:
    """
        Validate that a given quantization backend is available

        Args:
            backend: Backend identifier

        Returns:
            None
    """
    LOGGER.info("Checking availability for backend=%s", backend)

    if backend == "gguf":
        _check_gguf_tooling()

    elif backend == "awq":
        ## Package name is usually "autoawq"
        _check_python_module("autoawq")

    elif backend == "gptq":
        ## Package name is usually "auto_gptq"
        _check_python_module("auto_gptq")

    elif backend == "bnb_nf4":
        _check_python_module("bitsandbytes")

    elif backend == "onnx":
        _check_python_module("onnxruntime")

    else:
        log_and_raise_backend_unavailable(
            backend_name=backend,
            reason="Unknown backend identifier",
        )


## Registry exposed for pipeline usage
BACKEND_REGISTRY: Dict[str, BackendHandler] = {}
