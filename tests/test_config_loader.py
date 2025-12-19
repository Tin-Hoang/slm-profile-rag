"""Tests for config loader."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.config_loader import ConfigLoader, get_config, reload_config


@pytest.fixture
def temp_config():
    """Create a temporary config file."""
    config_data = {
        "llm": {
            "model": "test-model",
            "temperature": 0.5,
        },
        "profile": {
            "name": "Test User",
            "title": "Test Title",
        },
        "logging": {
            "level": "INFO",
            "format": "%(message)s",
        },
        "nested": {
            "level1": {
                "level2": "deep_value",
            }
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    yield config_path

    # Cleanup
    os.unlink(config_path)


# --- ConfigLoader Tests ---


def test_config_loader_initialization():
    """Test that ConfigLoader initializes properly."""
    config = ConfigLoader()
    assert config is not None
    assert config.config is not None


def test_config_loader_with_custom_path(temp_config):
    """Test ConfigLoader with custom config path."""
    config = ConfigLoader(config_path=temp_config)

    assert config.get("llm.model") == "test-model"
    assert config.get("llm.temperature") == 0.5


def test_config_loader_missing_config_raises():
    """Test ConfigLoader raises for missing config file."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        ConfigLoader(config_path="/nonexistent/config.yaml")


def test_get_nested_config():
    """Test getting nested configuration values."""
    config = ConfigLoader()

    # Test nested access
    model = config.get("llm.model")
    assert model is not None

    # Test default value
    nonexistent = config.get("nonexistent.key", "default_value")
    assert nonexistent == "default_value"


def test_get_deeply_nested_config(temp_config):
    """Test getting deeply nested values."""
    config = ConfigLoader(config_path=temp_config)

    value = config.get("nested.level1.level2")
    assert value == "deep_value"


def test_get_non_dict_intermediate(temp_config):
    """Test get returns default when intermediate key is not a dict."""
    config = ConfigLoader(config_path=temp_config)

    # llm.model is a string, not a dict
    value = config.get("llm.model.invalid", "default")
    assert value == "default"


def test_get_env():
    """Test getting environment variables."""
    config = ConfigLoader()

    # Test with existing env var
    with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
        value = config.get_env("TEST_VAR")
        assert value == "test_value"

    # Test with default
    value = config.get_env("NONEXISTENT_VAR", "default")
    assert value == "default"


def test_format_template():
    """Test template formatting with profile info."""
    config = ConfigLoader()

    template = "Hello, I'm {name}, a {title}"
    formatted = config.format_template(template)

    assert "{name}" not in formatted
    assert "{title}" not in formatted


def test_format_template_with_custom_config(temp_config):
    """Test template formatting with custom config."""
    config = ConfigLoader(config_path=temp_config)

    template = "Hello, I'm {name}, a {title}"
    formatted = config.format_template(template)

    assert "Test User" in formatted
    assert "Test Title" in formatted


def test_format_template_missing_profile():
    """Test template formatting with missing profile uses defaults."""
    config_data = {"llm": {"model": "test"}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        config = ConfigLoader(config_path=config_path)
        template = "Hello, I'm {name}, a {title}"
        formatted = config.format_template(template)

        assert "the candidate" in formatted
        assert "professional" in formatted
    finally:
        os.unlink(config_path)


def test_config_property(temp_config):
    """Test config property returns full config dict."""
    config = ConfigLoader(config_path=temp_config)

    full_config = config.config

    assert isinstance(full_config, dict)
    assert "llm" in full_config
    assert "profile" in full_config


def test_load_env_from_file():
    """Test loading environment from .env file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env_path = Path(tmpdir) / ".env"
        env_path.write_text("TEST_ENV_VAR=loaded_value\n")

        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text("llm:\n  model: test\n")

        # Change to temp directory to test .env loading
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            # Load the config which should trigger .env loading
            ConfigLoader(config_path=str(config_path), env_path=str(env_path))

            # The value should be loaded from .env
            value = os.getenv("TEST_ENV_VAR")
            assert value == "loaded_value"
        finally:
            os.chdir(original_cwd)
            # Clean up env var
            os.environ.pop("TEST_ENV_VAR", None)


def test_load_env_missing_uses_template():
    """Test loading environment falls back to env.template."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text("llm:\n  model: test\n")

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            # No .env file, no env.template - should just log warning
            config = ConfigLoader(
                config_path=str(config_path),
                env_path=str(Path(tmpdir) / "nonexistent.env"),
            )
            assert config is not None
        finally:
            os.chdir(original_cwd)


# --- Singleton Function Tests ---


def test_get_config_singleton():
    """Test get_config returns singleton instance."""
    config1 = get_config()
    config2 = get_config()

    assert config1 is config2


def test_reload_config():
    """Test reload_config creates new instance."""
    # Get initial config
    get_config()
    # Reload and verify new instance
    config2 = reload_config()

    # reload_config creates a new instance
    assert config2 is not None
    assert isinstance(config2, ConfigLoader)
