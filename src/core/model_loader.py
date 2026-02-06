'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Model loading helpers for Hugging Face or local paths, with optional adapter support."
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

from src.core.errors import log_and_raise_missing_path, log_and_raise_pipeline_error
from src.utils.logging_utils import get_logger

## Initialize module-level logger
LOGGER = get_logger(__name__)


def resolve_model_source(
    model_name_or_path: str,
) -> str:
    """
        Resolve a model source to a valid Hugging Face id or local path

        Args:
            model_name_or_path: HF repo id or local filesystem path

        Returns:
            The resolved HF repo id or absolute local path
    """
    candidate = Path(model_name_or_path).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return model_name_or_path


def load_hf_model_and_tokenizer(
    model_name_or_path: str,
    revision: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
        Load a Hugging Face causal language model and its tokenizer

        Notes:
            - Import is done locally to keep optional dependencies isolated
            - The caller decides device placement and quantization strategy

        Args:
            model_name_or_path: HF repo id or local path
            revision: Optional HF revision or branch

        Returns:
            A tuple (model, tokenizer)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        log_and_raise_pipeline_error(
            step="hf_import",
            reason=f"transformers is required to load HF models: {exc}",
        )

    resolved = resolve_model_source(model_name_or_path)
    LOGGER.info("Loading HF model from source=%s", resolved)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            resolved,
            revision=revision,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            resolved,
            revision=revision,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        return model, tokenizer
    except Exception as exc:
        log_and_raise_pipeline_error(
            step="hf_load",
            reason=str(exc),
        )
        raise


def validate_optional_adapter(adapter_path: Optional[Path]) -> None:
    """
        Validate that an optional adapter path exists

        Args:
            adapter_path: Optional LoRA adapter path

        Returns:
            None
    """
    if adapter_path is None:
        return

    if not adapter_path.exists():
        log_and_raise_missing_path(adapter_path, context="adapter_path")
