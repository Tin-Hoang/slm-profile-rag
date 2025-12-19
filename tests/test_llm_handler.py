"""Tests for LLM handler module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.llm_handler import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_OLLAMA_MODEL,
    GROQ_MODELS,
    OLLAMA_MODELS,
    LLMHandler,
    detect_default_provider,
    get_available_groq_models,
    get_available_ollama_models,
    get_default_groq_model,
    get_default_ollama_model,
    get_llm_handler,
)

# --- LLMHandler Tests ---


def test_init_with_default_provider_no_groq_key():
    """Test handler initialization without GROQ_API_KEY defaults to ollama."""
    with patch.dict(os.environ, {}, clear=True), patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.return_value = "ollama"
        mock_config.return_value.get_env.return_value = None

        handler = LLMHandler()

        assert handler.provider == "ollama"


def test_init_with_groq_api_key_auto_detects_groq():
    """Test handler auto-detects groq provider when API key is present."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.return_value = "ollama"
        mock_config.return_value.get_env.return_value = "test-groq-api-key"

        handler = LLMHandler()

        assert handler.provider == "groq"


def test_init_with_provider_override():
    """Test handler respects provider override."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.return_value = "ollama"
        mock_config.return_value.get_env.return_value = "test-groq-api-key"

        handler = LLMHandler(provider_override="ollama")

        assert handler.provider == "ollama"


def test_init_with_api_key_override():
    """Test handler uses API key override."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.return_value = "ollama"
        mock_config.return_value.get_env.return_value = None

        handler = LLMHandler(api_key_override="custom-api-key")

        # Should detect groq because we provided an API key override
        assert handler.provider == "groq"


def test_get_groq_api_key_from_override():
    """Test getting Groq API key from override."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.return_value = "groq"
        mock_config.return_value.get_env.return_value = None

        handler = LLMHandler(api_key_override="override-key")

        assert handler._get_groq_api_key() == "override-key"


def test_get_groq_api_key_from_env():
    """Test getting Groq API key from environment."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.return_value = "groq"
        mock_config.return_value.get_env.return_value = "env-groq-key"

        handler = LLMHandler(provider_override="groq")

        assert handler._get_groq_api_key() == "env-groq-key"


def test_get_groq_llm_without_api_key_raises():
    """Test get_groq_llm raises ValueError without API key."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.return_value = "groq"
        mock_config.return_value.get_env.return_value = None

        handler = LLMHandler(provider_override="groq")

        with pytest.raises(ValueError, match="GROQ_API_KEY not found"):
            handler.get_groq_llm()


def test_get_groq_llm_with_api_key():
    """Test get_groq_llm initializes with API key."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "llm.provider": "groq",
            "llm.groq_model": "llama-3.3-70b-versatile",
            "llm.temperature": 0.7,
            "llm.max_tokens": 800,
        }.get(key, default)
        mock_config.return_value.get_env.return_value = None

        handler = LLMHandler(provider_override="groq", api_key_override="test-api-key")

        # ChatGroq is imported inside the function, so patch in langchain_groq
        with patch("langchain_groq.ChatGroq") as mock_groq:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content="Test response")
            mock_groq.return_value = mock_llm

            llm = handler.get_groq_llm()

            assert llm is not None
            mock_groq.assert_called_once()


def test_get_groq_llm_connection_error():
    """Test get_groq_llm handles connection errors."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "llm.provider": "groq",
            "llm.groq_model": "llama-3.3-70b-versatile",
            "llm.temperature": 0.7,
            "llm.max_tokens": 800,
        }.get(key, default)
        mock_config.return_value.get_env.return_value = None

        handler = LLMHandler(provider_override="groq", api_key_override="test-api-key")

        # ChatGroq is imported inside the function
        with patch("langchain_groq.ChatGroq") as mock_groq:
            mock_groq.side_effect = Exception("Connection error")

            with pytest.raises(Exception, match="Connection error"):
                handler.get_groq_llm()


def test_get_ollama_llm():
    """Test get_ollama_llm initialization."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "llm.provider": "ollama",
            "llm.model": "llama3.2:3b",
            "llm.temperature": 0.7,
            "llm.max_tokens": 512,
            "llm.top_p": 0.9,
        }.get(key, default)
        mock_config.return_value.get_env.side_effect = lambda key, default=None: {
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }.get(key, default)

        handler = LLMHandler(provider_override="ollama")

        with patch("src.llm_handler.OllamaLLM") as mock_ollama:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = "Test response from Ollama"
            mock_ollama.return_value = mock_llm

            llm = handler.get_ollama_llm()

            assert llm is not None
            mock_ollama.assert_called_once()


def test_get_ollama_llm_model_not_found():
    """Test get_ollama_llm raises helpful error for missing model."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "llm.provider": "ollama",
            "llm.model": "nonexistent-model",
            "llm.temperature": 0.7,
            "llm.max_tokens": 512,
            "llm.top_p": 0.9,
        }.get(key, default)
        mock_config.return_value.get_env.side_effect = lambda key, default=None: {
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }.get(key, default)

        handler = LLMHandler(provider_override="ollama")

        with patch("src.llm_handler.OllamaLLM") as mock_ollama:
            mock_ollama.return_value.invoke.side_effect = Exception(
                "model not found: nonexistent-model"
            )

            with pytest.raises(ValueError, match="not found in Ollama"):
                handler.get_ollama_llm()


def test_get_ollama_llm_other_error():
    """Test get_ollama_llm re-raises non-model-not-found errors."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "llm.provider": "ollama",
            "llm.model": "llama3.2:3b",
            "llm.temperature": 0.7,
            "llm.max_tokens": 512,
            "llm.top_p": 0.9,
        }.get(key, default)
        mock_config.return_value.get_env.side_effect = lambda key, default=None: {
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }.get(key, default)

        handler = LLMHandler(provider_override="ollama")

        with patch("src.llm_handler.OllamaLLM") as mock_ollama:
            mock_ollama.return_value.invoke.side_effect = Exception("Connection refused")

            with pytest.raises(Exception, match="Connection refused"):
                handler.get_ollama_llm()


def test_get_llm_returns_cached_instance():
    """Test get_llm returns cached LLM instance."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.return_value = "ollama"
        mock_config.return_value.get_env.return_value = None

        handler = LLMHandler(provider_override="ollama")

        # Set a pre-cached LLM
        mock_llm = MagicMock()
        handler.llm = mock_llm

        result = handler.get_llm()

        assert result is mock_llm


def test_get_llm_unsupported_provider_raises():
    """Test get_llm raises for unsupported provider."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.return_value = "unsupported"
        mock_config.return_value.get_env.return_value = None

        handler = LLMHandler(provider_override="unsupported")

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            handler.get_llm()


def test_get_system_prompt():
    """Test get_system_prompt returns formatted prompt."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "llm.provider": "ollama",
            "llm.system_prompt": "Hello {name}, you are a {title}",
        }.get(key, default)
        mock_config.return_value.get_env.return_value = None
        mock_config.return_value.format_template.return_value = "Hello Tin, you are a Engineer"

        handler = LLMHandler(provider_override="ollama")
        prompt = handler.get_system_prompt()

        assert prompt == "Hello Tin, you are a Engineer"


def test_get_provider():
    """Test get_provider returns current provider."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.return_value = "ollama"
        mock_config.return_value.get_env.return_value = None

        handler = LLMHandler(provider_override="groq")

        assert handler.get_provider() == "groq"


def test_get_model_with_override():
    """Test get_model returns override when provided."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.return_value = "ollama"
        mock_config.return_value.get_env.return_value = None

        handler = LLMHandler(provider_override="ollama", model_override="custom-model")

        assert handler.get_model() == "custom-model"


def test_get_model_groq_provider():
    """Test get_model returns groq model for groq provider."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "llm.provider": "groq",
            "llm.groq_model": "llama-3.3-70b-versatile",
        }.get(key, default)
        mock_config.return_value.get_env.return_value = "test-key"

        handler = LLMHandler(provider_override="groq")

        assert handler.get_model() == "llama-3.3-70b-versatile"


def test_get_model_ollama_provider():
    """Test get_model returns ollama model for ollama provider."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "llm.provider": "ollama",
            "llm.model": "llama3.2:3b",
        }.get(key, default)
        mock_config.return_value.get_env.return_value = None

        handler = LLMHandler(provider_override="ollama")

        assert handler.get_model() == "llama3.2:3b"


# --- Helper Functions Tests ---


def test_get_llm_handler():
    """Test get_llm_handler factory function."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get.return_value = "ollama"
        mock_config.return_value.get_env.return_value = None

        handler = get_llm_handler(provider_override="ollama")

        assert isinstance(handler, LLMHandler)


def test_get_available_groq_models():
    """Test get_available_groq_models returns list copy."""
    models = get_available_groq_models()

    assert isinstance(models, list)
    assert len(models) > 0
    assert models == GROQ_MODELS
    # Verify it's a copy
    models.append("test-model")
    assert "test-model" not in GROQ_MODELS


def test_get_default_groq_model():
    """Test get_default_groq_model returns default."""
    model = get_default_groq_model()

    assert model == DEFAULT_GROQ_MODEL


def test_get_available_ollama_models_success():
    """Test get_available_ollama_models fetches from API."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get_env.return_value = "http://localhost:11434"

        # ollama is imported inside the function
        with patch.dict("sys.modules", {"ollama": MagicMock()}):
            import sys

            mock_ollama = sys.modules["ollama"]
            mock_model1 = MagicMock()
            mock_model1.model = "llama3.2:3b"
            mock_model2 = MagicMock()
            mock_model2.model = "phi3:mini"

            mock_response = MagicMock()
            mock_response.models = [mock_model1, mock_model2]
            mock_ollama.Client.return_value.list.return_value = mock_response

            models = get_available_ollama_models()

            assert "llama3.2:3b" in models
            assert "phi3:mini" in models


def test_get_available_ollama_models_empty():
    """Test get_available_ollama_models returns empty list when no models."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get_env.return_value = "http://localhost:11434"

        # ollama is imported inside the function
        with patch.dict("sys.modules", {"ollama": MagicMock()}):
            import sys

            mock_ollama = sys.modules["ollama"]
            mock_response = MagicMock()
            mock_response.models = []
            mock_ollama.Client.return_value.list.return_value = mock_response

            models = get_available_ollama_models()

            assert models == []


def test_get_available_ollama_models_fallback_on_error():
    """Test get_available_ollama_models falls back to static list on error."""
    with patch("src.llm_handler.get_config") as mock_config:
        mock_config.return_value.get_env.return_value = "http://localhost:11434"

        # ollama is imported inside the function
        with patch.dict("sys.modules", {"ollama": MagicMock()}):
            import sys

            mock_ollama = sys.modules["ollama"]
            mock_ollama.Client.side_effect = Exception("Connection failed")

            models = get_available_ollama_models()

            assert models == OLLAMA_MODELS


def test_get_default_ollama_model():
    """Test get_default_ollama_model returns default."""
    model = get_default_ollama_model()

    assert model == DEFAULT_OLLAMA_MODEL


def test_detect_default_provider_with_groq_key():
    """Test detect_default_provider returns groq when key present."""
    # Path is imported from pathlib inside the function
    with patch("pathlib.Path.exists", return_value=False):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            provider = detect_default_provider()

            assert provider == "groq"


def test_detect_default_provider_without_groq_key():
    """Test detect_default_provider returns ollama when no key."""
    # Path is imported from pathlib inside the function
    with patch("pathlib.Path.exists", return_value=False):
        with patch.dict(os.environ, {}, clear=True):
            # Remove GROQ_API_KEY if it exists
            os.environ.pop("GROQ_API_KEY", None)
            provider = detect_default_provider()

            assert provider == "ollama"
