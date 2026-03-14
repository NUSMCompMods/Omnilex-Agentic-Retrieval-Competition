"""LLM loading utilities for local and Modal-backed inference."""

import os
from pathlib import Path
from typing import Any

# Type hint for Llama (actual import happens at runtime)
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore


def is_kaggle_env() -> bool:
    """Check if running in Kaggle notebook environment.

    Returns:
        True if running on Kaggle, False otherwise
    """
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def get_default_model_path() -> Path:
    """Get the default model path based on environment.

    Returns:
        Path to model directory
    """
    if is_kaggle_env():
        # Kaggle input dataset path
        return Path("/kaggle/input/llama-model")
    else:
        # Local development path
        return Path(__file__).parent.parent.parent.parent / "models"


def find_model_file(
    model_dir: Path,
    pattern: str = "*.gguf",
) -> Path | None:
    """Find a model file in a directory.

    Args:
        model_dir: Directory to search
        pattern: Glob pattern for model files

    Returns:
        Path to first matching model file, or None
    """
    model_dir = Path(model_dir)

    if not model_dir.exists():
        return None

    # If model_dir is actually a file, return it
    if model_dir.is_file():
        return model_dir

    # Search for model files
    matches = list(model_dir.glob(pattern))
    if matches:
        return matches[0]

    # Try recursive search
    matches = list(model_dir.rglob(pattern))
    if matches:
        return matches[0]

    return None


def has_cuda_support() -> bool:
    """Check if llama-cpp-python was built with CUDA support.

    Returns:
        True if CUDA support is available, False otherwise
    """
    if Llama is None:
        return False

    try:
        import importlib.util

        spec = importlib.util.find_spec("llama_cpp")
        if spec and spec.origin:
            lib_dir = Path(spec.origin).parent
            # Check for CUDA shared libraries in main dir and lib/ subdirectory
            cuda_libs = (
                list(lib_dir.glob("*cuda*"))
                + list(lib_dir.glob("*cublas*"))
                + list((lib_dir / "lib").glob("*cuda*"))
                + list((lib_dir / "lib").glob("*cublas*"))
            )
            if cuda_libs:
                return True
        return False
    except Exception:
        return False


def get_device_info(n_gpu_layers: int) -> str:
    """Get human-readable device info string.

    Args:
        n_gpu_layers: Number of GPU layers configured

    Returns:
        String describing the compute device
    """
    if n_gpu_layers == -1:
        return "GPU (all layers offloaded)"
    elif n_gpu_layers > 0:
        return f"GPU ({n_gpu_layers} layers offloaded)"
    else:
        return "CPU"


def load_model(
    model_path: Path | str | None = None,
    n_ctx: int = 4096,
    n_threads: int | None = None,
    n_gpu_layers: int | None = None,
    verbose: bool = False,
    backend: str | None = None,
    modal_app_name: str | None = None,
    modal_class_name: str | None = None,
    **kwargs,
) -> Any:
    """Load a GGUF model using llama-cpp-python.

    Args:
        model_path: Path to model file or directory containing model.
                   If None, uses default path based on environment.
        n_ctx: Context window size (max tokens)
        n_threads: Number of CPU threads (None = auto-detect)
        n_gpu_layers: Number of layers to offload to GPU.
                     None = auto-detect (GPU if available, else CPU)
                     -1 = offload all layers to GPU
                     0 = CPU only
        verbose: Whether to print llama.cpp logs
        backend: Execution backend. `local` uses llama-cpp-python in-process.
                 `modal` forwards inference to a deployed Modal L40S container.
                 If None, reads `OMNILEX_LLM_BACKEND` and falls back to `local`.
        modal_app_name: Override the deployed Modal app name.
        modal_class_name: Override the deployed Modal class name.
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        Loaded local model instance or Modal proxy object

    Raises:
        ImportError: If llama-cpp-python is not installed
        FileNotFoundError: If model file not found
    """
    selected_backend = (backend or os.environ.get("OMNILEX_LLM_BACKEND", "local")).lower()

    if selected_backend == "modal":
        from .modal_backend import ModalLlamaProxy

        resolved_n_gpu_layers = -1 if n_gpu_layers is None else n_gpu_layers
        return ModalLlamaProxy(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=resolved_n_gpu_layers,
            verbose=verbose,
            app_name=modal_app_name,
            class_name=modal_class_name,
            model_kwargs=kwargs,
        )

    if selected_backend != "local":
        raise ValueError(
            f"Unsupported backend {selected_backend!r}. Expected one of: local, modal."
        )

    if Llama is None:
        raise ImportError(
            "llama-cpp-python is required. Install with: pip install llama-cpp-python"
        )

    # Determine model path
    if model_path is None:
        model_dir = get_default_model_path()
        model_file = find_model_file(model_dir)
        if model_file is None:
            raise FileNotFoundError(
                f"No GGUF model found in {model_dir}. "
                "Please download a model or specify model_path."
            )
        model_path = model_file
    else:
        model_path = Path(model_path)
        if model_path.is_dir():
            model_file = find_model_file(model_path)
            if model_file is None:
                raise FileNotFoundError(f"No GGUF model found in {model_path}")
            model_path = model_file

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Auto-detect threads if not specified
    if n_threads is None:
        n_threads = min(os.cpu_count() or 4, 8)

    # Auto-detect GPU: use GPU if CUDA support available, else CPU
    if n_gpu_layers is None:
        n_gpu_layers = -1 if has_cuda_support() else 0

    # Load model
    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
        **kwargs,
    )

    return llm


def generate(
    llm: Any,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    stop: list[str] | None = None,
    **kwargs,
) -> str:
    """Generate text from prompt.

    Args:
        llm: Loaded Llama model
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = deterministic)
        stop: Stop sequences
        **kwargs: Additional generation arguments

    Returns:
        Generated text
    """
    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        **kwargs,
    )

    return response["choices"][0]["text"]


def count_tokens(llm: Any, text: str) -> int:
    """Count tokens in a text string.

    Args:
        llm: Loaded Llama model or Modal proxy (for tokenizer)
        text: Text to tokenize

    Returns:
        Number of tokens
    """
    tokens = llm.tokenize(text.encode("utf-8"))
    return len(tokens)
