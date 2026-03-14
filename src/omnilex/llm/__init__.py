"""LLM loading and prompt utilities."""

from .loader import count_tokens, generate, get_device_info, has_cuda_support, is_kaggle_env, load_model
from .prompts import AGENT_SYSTEM_PROMPT, DIRECT_GENERATION_PROMPT

__all__ = [
    "load_model",
    "generate",
    "count_tokens",
    "is_kaggle_env",
    "has_cuda_support",
    "get_device_info",
    "DIRECT_GENERATION_PROMPT",
    "AGENT_SYSTEM_PROMPT",
]
