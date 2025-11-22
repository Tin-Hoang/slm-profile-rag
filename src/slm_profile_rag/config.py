"""Configuration loader for the RAG chatbot."""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class Config:
    """Configuration manager for the RAG chatbot."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration.

        Args:
            config_path: Path to the YAML configuration file.
        """
        load_dotenv()

        self.config_path = Path(config_path)
        self._config = self._load_config()

        # Environment variables override
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv(
            "OLLAMA_MODEL", self._config.get("model", {}).get("name", "llama2")
        )
        self.profile_docs_path = os.getenv("PROFILE_DOCS_PATH", "./profile_docs")
        self.vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        # Setup logging
        setup_logging(self.log_level)

        # YAML config values with env override
        chunk_size = os.getenv("CHUNK_SIZE")
        chunk_overlap = os.getenv("CHUNK_OVERLAP")

        self.chunk_size = (
            int(chunk_size)
            if chunk_size
            else self._config.get("documents", {}).get("chunk_size", 1000)
        )
        self.chunk_overlap = (
            int(chunk_overlap)
            if chunk_overlap
            else self._config.get("documents", {}).get("chunk_overlap", 200)
        )

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'model.name').
            default: Default value if key is not found.

        Returns:
            Configuration value or default.
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value
