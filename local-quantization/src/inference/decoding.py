'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Deterministic decoding configuration defaults for inference and benchmarking."
'''

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodingConfig:
    """
        Configuration describing decoding parameters for inference

        Design choice:
            - Defaults are deterministic to make benchmarking reproducible
            - Sampling is disabled unless explicitly enabled

        Args:
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to enable sampling
    """

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
