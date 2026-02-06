'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unit tests for core utilities, settings builder, and error helpers."
'''

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config.settings import build_pipeline_config
from src.core.errors import (
    ConfigurationError,
    DataError,
    log_and_raise_missing_env,
    log_and_raise_missing_path,
)
from src.utils.utils import get_env_bool, get_env_int, get_env_str


def test_get_env_str_returns_default_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """
        Validate get_env_str returns default when env variable is missing

        Args:
            monkeypatch: Pytest monkeypatch fixture

        Returns:
            None
    """
    monkeypatch.delenv("TEST_ENV_STR", raising=False)
    assert get_env_str("TEST_ENV_STR", "default") == "default"


def test_get_env_int_returns_default_on_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """
        Validate get_env_int returns default when env value is invalid

        Args:
            monkeypatch: Pytest monkeypatch fixture

        Returns:
            None
    """
    monkeypatch.setenv("TEST_ENV_INT", "not_an_int")
    assert get_env_int("TEST_ENV_INT", 123) == 123


def test_get_env_bool_parses_true_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """
        Validate get_env_bool parses common boolean values

        Args:
            monkeypatch: Pytest monkeypatch fixture

        Returns:
            None
    """
    monkeypatch.setenv("TEST_ENV_BOOL", "true")
    assert get_env_bool("TEST_ENV_BOOL", False) is True

    monkeypatch.setenv("TEST_ENV_BOOL", "false")
    assert get_env_bool("TEST_ENV_BOOL", True) is False


def test_log_and_raise_missing_env_raises_configuration_error() -> None:
    """
        Validate missing env helper raises ConfigurationError

        Args:
            None

        Returns:
            None
    """
    with pytest.raises(ConfigurationError):
        log_and_raise_missing_env("SOME_MISSING_ENV", reason="unit_test")


def test_log_and_raise_missing_path_raises_data_error(tmp_path: Path) -> None:
    """
        Validate missing path helper raises DataError

        Args:
            tmp_path: Pytest tmp_path fixture

        Returns:
            None
    """
    missing_path = tmp_path / "does_not_exist.txt"
    with pytest.raises(DataError):
        log_and_raise_missing_path(missing_path, context="unit_test")


def test_build_pipeline_config_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
        Validate build_pipeline_config returns a PipelineConfig with minimal env

        Args:
            monkeypatch: Pytest monkeypatch fixture
            tmp_path: Pytest tmp_path fixture

        Returns:
            None
    """
    ## Minimal required env
    monkeypatch.setenv("PIPELINE_MODE", "quantize")
    monkeypatch.setenv("MODEL_NAME_OR_PATH", "mistralai/Mistral-7B-Instruct-v0.3")
    monkeypatch.setenv("QUANT_BACKEND", "bnb_nf4")
    monkeypatch.setenv("QUANT_BITS", "4")

    ## Optional env for export/benchmark left empty
    monkeypatch.setenv("EXPORT_OUTPUT_DIR", "")
    monkeypatch.setenv("BENCHMARK_PROMPTS", "")

    config = build_pipeline_config()

    assert config.mode == "quantize"
    assert config.model.model_name_or_path != ""
    assert config.quantization.backend == "bnb_nf4"
    assert config.quantization.bits == 4


def test_build_pipeline_config_missing_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
        Validate build_pipeline_config raises when required env is missing

        Args:
            monkeypatch: Pytest monkeypatch fixture

        Returns:
            None
    """
    ## Ensure required env is missing
    monkeypatch.delenv("PIPELINE_MODE", raising=False)
    monkeypatch.delenv("MODEL_NAME_OR_PATH", raising=False)
    monkeypatch.delenv("QUANT_BACKEND", raising=False)

    with pytest.raises(ConfigurationError):
        build_pipeline_config()
