"""Configuration loader for YAML settings and environment variables."""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration from YAML and environment variables."""

    def __init__(self, config_path: str = "config.yaml", env_path: str = ".env"):
        """Initialize configuration loader.

        Args:
            config_path: Path to YAML configuration file
            env_path: Path to .env file
        """
        self.config_path = Path(config_path)
        self.env_path = Path(env_path)
        self._config: dict[str, Any] = {}

        # Load environment variables first
        self._load_env()

        # Load YAML configuration
        self._load_yaml()

        # Setup logging
        self._setup_logging()

    def _load_env(self) -> None:
        """Load environment variables from .env file."""
        if self.env_path.exists():
            load_dotenv(self.env_path)
            logger.debug(f"Loaded environment variables from {self.env_path}")
        else:
            logger.warning(f".env file not found at {self.env_path}, using env.template defaults")
            # Try to load from env.template as fallback
            env_template = Path("env.template")
            if env_template.exists():
                load_dotenv(env_template)

    def _load_yaml(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            msg = f"Configuration file not found: {self.config_path}"
            raise FileNotFoundError(msg)

        with open(self.config_path) as f:
            self._config = yaml.safe_load(f)

        logger.debug(f"Loaded configuration from {self.config_path}")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self._config.get("logging", {})
        log_level = os.getenv("LOG_LEVEL", log_config.get("level", "INFO"))

        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dot notation).

        Args:
            key: Configuration key (e.g., 'llm.model' for nested access)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_env(self, key: str, default: str | None = None) -> str | None:
        """Get environment variable.

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Environment variable value
        """
        return os.getenv(key, default)

    @property
    def config(self) -> dict[str, Any]:
        """Get the entire configuration dictionary."""
        return self._config

    def format_template(self, template: str) -> str:
        """Format template string with profile information.

        Args:
            template: Template string with {name}, {title} placeholders

        Returns:
            Formatted string
        """
        profile = self._config.get("profile", {})
        return template.format(
            name=profile.get("name", "the candidate"),
            title=profile.get("title", "professional"),
        )


# Singleton instance
_config_instance: ConfigLoader | None = None


def get_config() -> ConfigLoader:
    """Get singleton configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader()
    return _config_instance


def reload_config() -> ConfigLoader:
    """Reload configuration (useful for testing or config changes)."""
    global _config_instance
    _config_instance = ConfigLoader()
    return _config_instance
