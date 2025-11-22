"""Tests for config loader."""

from src.config_loader import ConfigLoader


def test_config_loader_initialization():
    """Test that ConfigLoader initializes properly."""
    config = ConfigLoader()
    assert config is not None
    assert config.config is not None


def test_get_nested_config():
    """Test getting nested configuration values."""
    config = ConfigLoader()

    # Test nested access
    model = config.get("llm.model")
    assert model is not None

    # Test default value
    nonexistent = config.get("nonexistent.key", "default_value")
    assert nonexistent == "default_value"


def test_format_template():
    """Test template formatting with profile info."""
    config = ConfigLoader()

    template = "Hello, I'm {name}, a {title}"
    formatted = config.format_template(template)

    assert "{name}" not in formatted
    assert "{title}" not in formatted
