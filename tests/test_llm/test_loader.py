"""Tests for LLM loader module."""

import json
import types

import pytest

from omnilex.llm import count_tokens, generate, get_device_info, has_cuda_support, is_kaggle_env


class TestDeviceInfo:
    """Test device info functions."""

    def test_get_device_info_cpu(self):
        """Test device info for CPU mode."""
        assert get_device_info(0) == "CPU"

    def test_get_device_info_gpu_partial(self):
        """Test device info for partial GPU offload."""
        assert get_device_info(10) == "GPU (10 layers offloaded)"

    def test_get_device_info_gpu_all(self):
        """Test device info for full GPU offload."""
        assert get_device_info(-1) == "GPU (all layers offloaded)"


class TestCudaSupport:
    """Test CUDA support detection."""

    def test_has_cuda_support_returns_bool(self):
        """Test that has_cuda_support returns a boolean."""
        result = has_cuda_support()
        assert isinstance(result, bool)


class TestEnvironmentDetection:
    """Test environment detection functions."""

    def test_is_kaggle_env_returns_bool(self):
        """Test that is_kaggle_env returns a boolean."""
        result = is_kaggle_env()
        assert isinstance(result, bool)

    def test_is_kaggle_env_false_locally(self):
        """Test that is_kaggle_env returns False in local environment."""
        # This test runs in CI/local, not Kaggle
        assert is_kaggle_env() is False


class TestLlamaImport:
    """Test llama-cpp-python import."""

    def test_llama_cpp_importable(self):
        """Test that llama-cpp-python can be imported."""
        try:
            from llama_cpp import Llama

            assert Llama is not None
        except ImportError:
            pytest.skip("llama-cpp-python not installed")

    def test_load_model_import_error_without_llama(self):
        """Test that load_model raises ImportError when llama_cpp unavailable."""
        # This test verifies the error handling works
        # In CI, llama_cpp should be installed
        from omnilex.llm.loader import Llama

        if Llama is None:
            from omnilex.llm import load_model

            with pytest.raises(ImportError):
                load_model(model_path="/nonexistent/path")


class TestModalBackend:
    """Test Modal dispatch logic without calling the real API."""

    def test_resolve_repo_root_handles_src_layout(self, tmp_path):
        from omnilex.llm import modal_backend

        repo_root = tmp_path / "repo"
        module_path = repo_root / "src" / "omnilex" / "llm" / "modal_backend.py"
        module_path.parent.mkdir(parents=True)
        module_path.write_text("# stub")
        (repo_root / "pyproject.toml").write_text("[project]\nname='stub'\n")

        assert modal_backend._resolve_repo_root(module_path) == repo_root

    def test_resolve_repo_root_handles_modal_flattened_file(self, tmp_path):
        from omnilex.llm import modal_backend

        module_path = tmp_path / "modal_backend.py"
        module_path.write_text("# stub")

        assert modal_backend._resolve_repo_root(module_path) == tmp_path

    def test_load_model_modal_returns_proxy(self, monkeypatch, tmp_path):
        from omnilex.llm import load_model
        from omnilex.llm import modal_backend

        captured: dict[str, object] = {}

        class FakeRemote:
            def __init__(self):
                self.completion = types.SimpleNamespace(remote=self._completion_remote)
                self.tokenize = types.SimpleNamespace(remote=self._tokenize_remote)
                self.metadata = types.SimpleNamespace(remote=self._metadata_remote)

            def _completion_remote(self, prompt, **kwargs):
                captured["prompt"] = prompt
                captured["completion_kwargs"] = kwargs
                return {"choices": [{"text": "remote text"}]}

            def _tokenize_remote(self, text):
                captured["tokenize_payload"] = text
                return [1, 2, 3]

            def _metadata_remote(self):
                return {"backend": "modal"}

        class FakeClsFactory:
            def __call__(self, **kwargs):
                captured["instance_kwargs"] = kwargs
                return FakeRemote()

        class FakeClsNamespace:
            @staticmethod
            def from_name(app_name, class_name):
                captured["app_name"] = app_name
                captured["class_name"] = class_name
                return FakeClsFactory()

        fake_modal = types.SimpleNamespace(Cls=FakeClsNamespace)

        monkeypatch.setattr(modal_backend, "modal", fake_modal)
        monkeypatch.setattr(modal_backend, "ensure_modal_credentials", lambda: None)

        model_file = tmp_path / "model.gguf"
        model_file.write_text("stub")

        llm = load_model(
            model_path=model_file,
            backend="modal",
            modal_app_name="test-app",
            modal_class_name="TestRemoteLlama",
            n_ctx=2048,
            chat_format="mistral-instruct",
        )

        assert captured["app_name"] == "test-app"
        assert captured["class_name"] == "TestRemoteLlama"
        assert captured["instance_kwargs"] == {
            "model_path": "/models/model.gguf",
            "n_ctx": 2048,
            "n_threads": 8,
            "n_gpu_layers": -1,
            "verbose": False,
            "model_kwargs_json": json.dumps({"chat_format": "mistral-instruct"}),
        }

        assert generate(llm, "hello") == "remote text"
        assert count_tokens(llm, "hello") == 3
        assert llm.metadata()["app_name"] == "test-app"
        assert llm.metadata()["gpu"] == modal_backend.MODAL_GPU
        assert captured["prompt"] == "hello"
        assert captured["tokenize_payload"] == b"hello"

    def test_load_model_uses_backend_env(self, monkeypatch):
        from omnilex.llm import load_model
        from omnilex.llm import modal_backend

        captured: dict[str, object] = {}

        class FakeRemote:
            def __init__(self):
                self.completion = types.SimpleNamespace(
                    remote=lambda prompt, **kwargs: {"choices": [{"text": prompt}]}
                )
                self.tokenize = types.SimpleNamespace(remote=lambda text: [1])
                self.metadata = types.SimpleNamespace(remote=lambda: {"backend": "modal"})

        class FakeClsFactory:
            def __call__(self, **kwargs):
                captured["instance_kwargs"] = kwargs
                return FakeRemote()

        class FakeClsNamespace:
            @staticmethod
            def from_name(app_name, class_name):
                captured["app_name"] = app_name
                captured["class_name"] = class_name
                return FakeClsFactory()

        fake_modal = types.SimpleNamespace(Cls=FakeClsNamespace)

        monkeypatch.setenv("OMNILEX_LLM_BACKEND", "modal")
        monkeypatch.setattr(modal_backend, "modal", fake_modal)
        monkeypatch.setattr(modal_backend, "ensure_modal_credentials", lambda: None)

        llm = load_model(model_path="uploaded-model.gguf")

        assert captured["app_name"] == modal_backend.MODAL_APP_NAME
        assert captured["class_name"] == modal_backend.MODAL_CLASS_NAME
        assert captured["instance_kwargs"]["model_path"] == "/models/uploaded-model.gguf"
        assert generate(llm, "ping") == "ping"
