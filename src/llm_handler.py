"""LLM handler for Ollama and other providers."""

import logging

from langchain_community.llms import Ollama
from langchain_core.language_models.base import BaseLanguageModel

from .config_loader import get_config

logger = logging.getLogger(__name__)


class LLMHandler:
    """Handle LLM initialization and configuration."""

    def __init__(self):
        """Initialize LLM handler with configuration."""
        self.config = get_config()
        self.provider = self.config.get("llm.provider", "ollama")
        self.llm: BaseLanguageModel | None = None

    def get_ollama_llm(self) -> Ollama:
        """Initialize Ollama LLM.

        Returns:
            Ollama LLM instance
        """
        model = self.config.get("llm.model", "llama3.2:3b")
        temperature = self.config.get("llm.temperature", 0.7)
        base_url = self.config.get_env("OLLAMA_BASE_URL", "http://localhost:11434")

        logger.info(f"Initializing Ollama with model: {model}")

        try:
            llm = Ollama(
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
            logger.error(f"Error initializing Ollama: {e}")
            logger.error(
                f"Make sure Ollama is running and the model is pulled. Try: ollama pull {model}"
            )
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


def get_llm_handler() -> LLMHandler:
    """Get LLM handler instance."""
    return LLMHandler()
