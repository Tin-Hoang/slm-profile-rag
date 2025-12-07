"""LLM handler for Ollama and Groq providers."""

import logging
import os

from langchain_core.language_models.base import BaseLanguageModel
from langchain_ollama import OllamaLLM

from .config_loader import get_config

logger = logging.getLogger(__name__)

# Available Ollama models (common options)
OLLAMA_MODELS = [
    "llama3.2:3b",
    "llama3.2:1b",
    "llama3.1:8b",
    "phi3:mini",
    "gemma2:2b",
    "mistral:7b",
    "qwen2.5:3b",
]

DEFAULT_OLLAMA_MODEL = "llama3.2:3b"

# Available Groq models (free tier)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


class LLMHandler:
    """Handle LLM initialization and configuration."""

    def __init__(
        self,
        provider_override: str | None = None,
        api_key_override: str | None = None,
        model_override: str | None = None,
    ):
        """Initialize LLM handler with configuration.

        Args:
            provider_override: Override provider from config (ollama/groq)
            api_key_override: Override API key from environment
            model_override: Override model from config
        """
        self.config = get_config()
        self._api_key_override = api_key_override
        self._model_override = model_override
        self.llm: BaseLanguageModel | None = None

        # Auto-detect provider based on GROQ_API_KEY presence
        if provider_override:
            self.provider = provider_override
        else:
            groq_api_key = self._get_groq_api_key()
            if groq_api_key:
                self.provider = "groq"
                logger.info("GROQ_API_KEY detected, using Groq as default provider")
            else:
                self.provider = self.config.get("llm.provider", "ollama")

    def _get_groq_api_key(self) -> str | None:
        """Get Groq API key from override or environment."""
        if self._api_key_override:
            return self._api_key_override
        return self.config.get_env("GROQ_API_KEY")

    def get_ollama_llm(self) -> OllamaLLM:
        """Initialize Ollama LLM.

        Returns:
            OllamaLLM instance
        """
        model = self._model_override or self.config.get("llm.model", "llama3.2:3b")
        temperature = self.config.get("llm.temperature", 0.7)
        base_url = self.config.get_env("OLLAMA_BASE_URL", "http://localhost:11434")

        logger.info(f"Initializing Ollama with model: {model}")

        try:
            llm = OllamaLLM(
                model=model,
                temperature=temperature,
                base_url=base_url,
                num_predict=self.config.get("llm.max_tokens", 512),
                top_p=self.config.get("llm.top_p", 0.9),
            )

            # Test the connection
            logger.debug("Testing Ollama connection...")
            test_response = llm.invoke("Hi")
            logger.debug(f"Ollama test response: {test_response[:50]}...")

            return llm

        except Exception as e:
            # Check for "model not found" or 404 error
            error_str = str(e).lower()
            if "not found" in error_str or "404" in error_str:
                logger.warning(f"Ollama model '{model}' not found.")
                # We raise a ValueError with a helpful message that app.py can display nicely
                msg = (
                    f"Model '{model}' not found in Ollama.\n"
                    f"Please run this command in your terminal:\n"
                    f"ollama pull {model}"
                )
                raise ValueError(msg) from e

            logger.error(f"Error initializing Ollama: {e}")
            logger.error(
                f"Make sure Ollama is running and the model is pulled. Try: ollama pull {model}"
            )
            # Re-raise the original exception if it's not a missing model issue
            raise

    def get_groq_llm(self) -> BaseLanguageModel:
        """Initialize Groq LLM.

        Returns:
            ChatGroq instance
        """
        from langchain_groq import ChatGroq

        api_key = self._get_groq_api_key()
        if not api_key:
            msg = "GROQ_API_KEY not found. Set it in .env or provide via UI."
            raise ValueError(msg)

        model = self._model_override or self.config.get("llm.groq_model", DEFAULT_GROQ_MODEL)
        temperature = self.config.get("llm.temperature", 0.7)
        max_tokens = self.config.get("llm.max_tokens", 800)

        logger.info(f"Initializing Groq with model: {model}")

        try:
            llm = ChatGroq(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Test the connection
            logger.debug("Testing Groq connection...")
            test_response = llm.invoke("Hi")
            logger.debug(f"Groq test response: {str(test_response.content)[:50]}...")

            return llm

        except Exception as e:
            logger.error(f"Error initializing Groq: {e}")
            raise

    def get_llm(self) -> BaseLanguageModel:
        """Get LLM instance based on configured provider.

        Returns:
            LLM instance
        """
        if self.llm is not None:
            return self.llm

        if self.provider == "ollama":
            self.llm = self.get_ollama_llm()
        elif self.provider == "groq":
            self.llm = self.get_groq_llm()
        else:
            msg = f"Unsupported LLM provider: {self.provider}"
            raise ValueError(msg)

        return self.llm

    def get_system_prompt(self) -> str:
        """Get formatted system prompt from configuration.

        Returns:
            Formatted system prompt
        """
        template = self.config.get(
            "llm.system_prompt",
            "You are a helpful AI assistant. Answer questions based on the provided context.",
        )
        return self.config.format_template(template)

    def get_provider(self) -> str:
        """Get the current provider name."""
        return self.provider

    def get_model(self) -> str:
        """Get the current model name."""
        if self._model_override:
            return self._model_override
        if self.provider == "groq":
            return self.config.get("llm.groq_model", DEFAULT_GROQ_MODEL)
        return self.config.get("llm.model", "llama3.2:3b")


def get_llm_handler(
    provider_override: str | None = None,
    api_key_override: str | None = None,
    model_override: str | None = None,
) -> LLMHandler:
    """Get LLM handler instance with optional overrides."""
    return LLMHandler(
        provider_override=provider_override,
        api_key_override=api_key_override,
        model_override=model_override,
    )


def get_available_groq_models() -> list[str]:
    """Get list of available Groq models."""
    return GROQ_MODELS.copy()


def get_default_groq_model() -> str:
    """Get default Groq model."""
    return DEFAULT_GROQ_MODEL


def get_available_ollama_models() -> list[str]:
    """Get list of currently pulled Ollama models.

    Fetches the list from Ollama API. Falls back to static list if unavailable.
    """
    try:
        import ollama

        config = get_config()
        base_url = config.get_env("OLLAMA_BASE_URL", "http://localhost:11434")

        # Create client with configured host
        client = ollama.Client(host=base_url)
        response = client.list()

        # Extract model names - response.models is a list of Model objects
        models = []
        for model in response.models:
            # Each model has a .model attribute with the name
            name = getattr(model, "model", None) or getattr(model, "name", None)
            if name:
                models.append(name)

        if models:
            logger.debug(f"Found {len(models)} pulled Ollama models: {models}")
            return models

        # No models pulled - return empty list
        logger.warning("No Ollama models found. Please pull a model first.")
        return []

    except Exception as e:
        logger.warning(f"Could not fetch Ollama models: {e}. Using static list.")
        return OLLAMA_MODELS.copy()


def get_default_ollama_model() -> str:
    """Get default Ollama model."""
    return DEFAULT_OLLAMA_MODEL


def detect_default_provider() -> str:
    """Detect default provider based on environment.

    Loads .env file first to ensure environment variables are available.
    """
    from pathlib import Path

    from dotenv import load_dotenv

    # Load .env file to ensure GROQ_API_KEY is available
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)

    if os.getenv("GROQ_API_KEY"):
        return "groq"
    return "ollama"
