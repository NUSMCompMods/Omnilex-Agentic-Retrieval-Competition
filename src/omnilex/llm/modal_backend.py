"""Modal-backed remote GGUF inference for Omnilex.

Deploy the backend once:

    set -a && source .env && set +a
    python scripts/upload_modal_model.py models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
    modal deploy src/omnilex/llm/modal_backend.py

Then switch local code to the remote backend with either:

    export OMNILEX_LLM_BACKEND=modal

or:

    load_model(..., backend="modal")
"""

import os
import json
from pathlib import Path
from typing import Any

try:
    import modal
except ImportError:  # pragma: no cover - exercised via loader fallback
    modal = None  # type: ignore[assignment]


MODAL_APP_NAME = "omnilex-remote-llm"
MODAL_CLASS_NAME = "RemoteLlama"
MODAL_MODEL_VOLUME_NAME = "omnilex-llm-models"
MODAL_MODEL_DIR = "/models"
MODAL_GPU = "L40S"
MODAL_IMAGE_TAG = "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04"
LLAMA_CPP_VERSION = "0.3.16"
LLAMA_CPP_CUDA_WHEEL = "https://abetlen.github.io/llama-cpp-python/whl/cu124"

REPO_ROOT = Path(__file__).resolve().parents[3]


def _find_model_file(model_dir: Path, pattern: str = "*.gguf") -> Path | None:
    if not model_dir.exists():
        return None

    if model_dir.is_file():
        return model_dir

    matches = list(model_dir.glob(pattern))
    if matches:
        return matches[0]

    matches = list(model_dir.rglob(pattern))
    if matches:
        return matches[0]

    return None


def _default_local_model_file() -> Path | None:
    return _find_model_file(REPO_ROOT / "models")


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}

    if not path.exists():
        return values

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        values[key] = value

    return values


def ensure_modal_credentials() -> None:
    """Load Modal credentials from `.env` into the current Python process if needed."""

    required_keys = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
    if all(os.environ.get(key) for key in required_keys):
        return

    candidates = [Path.cwd() / ".env", REPO_ROOT / ".env"]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)

        for key, value in _parse_dotenv(candidate).items():
            if key in required_keys and not os.environ.get(key):
                os.environ[key] = value

        if all(os.environ.get(key) for key in required_keys):
            return


def resolve_modal_model_path(model_path: str | Path | None) -> str:
    """Map a local GGUF path to the mounted Modal volume path."""

    if model_path is None:
        default_model = _default_local_model_file()
        if default_model is None:
            raise FileNotFoundError(
                "No GGUF model found in the local models directory. "
                "Upload a model with scripts/upload_modal_model.py and "
                "pass model_path='your-model.gguf' or '/models/your-model.gguf'."
            )
        return f"{MODAL_MODEL_DIR}/{default_model.name}"

    model_path_str = str(model_path)
    if model_path_str.startswith(f"{MODAL_MODEL_DIR}/"):
        return model_path_str

    candidate = Path(model_path)
    if candidate.exists():
        if candidate.is_dir():
            model_file = _find_model_file(candidate)
            if model_file is None:
                raise FileNotFoundError(f"No GGUF model found in {candidate}")
            return f"{MODAL_MODEL_DIR}/{model_file.name}"
        return f"{MODAL_MODEL_DIR}/{candidate.name}"

    if candidate.suffix == ".gguf":
        return f"{MODAL_MODEL_DIR}/{candidate.name}"

    raise FileNotFoundError(
        f"Modal backend could not resolve model path {model_path!r}. "
        "Pass a GGUF filename that already exists in the Modal volume, an absolute "
        "remote path under /models, or a local GGUF file path."
    )


class ModalLlamaProxy:
    """Local proxy object that forwards `llama_cpp.Llama`-style calls to Modal."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        *,
        app_name: str | None = None,
        class_name: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if modal is None:
            raise ImportError("Modal backend requires `modal`. Install with: pip install modal")

        ensure_modal_credentials()

        self.model_path = resolve_modal_model_path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads if n_threads is not None else min(os.cpu_count() or 4, 8)
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.model_kwargs = model_kwargs or {}
        self.model_kwargs_json = json.dumps(self.model_kwargs)
        self.app_name = app_name or MODAL_APP_NAME
        self.class_name = class_name or MODAL_CLASS_NAME

        remote_cls = modal.Cls.from_name(self.app_name, self.class_name)
        self._remote = remote_cls(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
            model_kwargs_json=self.model_kwargs_json,
        )

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._remote.completion.remote(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **kwargs,
        )

    def tokenize(self, text: bytes) -> list[int]:
        return self._remote.tokenize.remote(text)

    def metadata(self) -> dict[str, Any]:
        return self._remote.metadata.remote()


if modal is not None:  # pragma: no branch
    app = modal.App(MODAL_APP_NAME)

    image = (
        modal.Image.from_registry(MODAL_IMAGE_TAG, add_python="3.10")
        .apt_install("libgomp1")
        .pip_install(
            f"llama-cpp-python=={LLAMA_CPP_VERSION}",
            extra_index_url=LLAMA_CPP_CUDA_WHEEL,
        )
    )

    model_volume = modal.Volume.from_name(MODAL_MODEL_VOLUME_NAME, create_if_missing=True)

    @app.cls(
        gpu=MODAL_GPU,
        image=image,
        volumes={MODAL_MODEL_DIR: model_volume},
        cpu=8.0,
        memory=32768,
        timeout=3600,
        startup_timeout=1800,
        scaledown_window=300,
    )
    class RemoteLlama:
        """GPU-backed `llama_cpp.Llama` hosted on Modal."""

        model_path: str = modal.parameter()
        n_ctx: int = modal.parameter(default=4096)
        n_threads: int = modal.parameter(default=8)
        n_gpu_layers: int = modal.parameter(default=-1)
        verbose: bool = modal.parameter(default=False)
        model_kwargs_json: str = modal.parameter(default="{}")

        @modal.enter()
        def load(self) -> None:
            from llama_cpp import Llama

            model_path = Path(self.model_path)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found in Modal volume: {self.model_path}. "
                    "Upload it first with scripts/upload_modal_model.py."
                )

            model_kwargs = json.loads(self.model_kwargs_json or "{}")

            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
                **model_kwargs,
            )

        @modal.method()
        def completion(
            self,
            prompt: str,
            max_tokens: int = 512,
            temperature: float = 0.1,
            stop: list[str] | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            return self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                **kwargs,
            )

        @modal.method()
        def tokenize(self, text: bytes) -> list[int]:
            return list(self.llm.tokenize(text))

        @modal.method()
        def metadata(self) -> dict[str, Any]:
            return {
                "backend": "modal",
                "gpu": MODAL_GPU,
                "model_path": self.model_path,
                "n_ctx": self.n_ctx,
                "n_threads": self.n_threads,
                "n_gpu_layers": self.n_gpu_layers,
            }
