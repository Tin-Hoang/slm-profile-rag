"""Tests for configuration module."""

import os
import tempfile

import pytest
import yaml

from slm_profile_rag.config import Config


def test_config_loads_yaml():
    """Test that config loads YAML file correctly."""
    config_data = {
        "model": {"name": "test-model", "temperature": 0.5},
        "documents": {"chunk_size": 500, "chunk_overlap": 100},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        config = Config(config_path)
        assert config.get("model.name") == "test-model"
        assert config.get("model.temperature") == 0.5
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
    finally:
        os.unlink(config_path)


def test_config_env_override():
    """Test that environment variables override config values."""
    config_data = {
        "model": {"name": "yaml-model"},
        "documents": {"chunk_size": 1000, "chunk_overlap": 200},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        os.environ["OLLAMA_MODEL"] = "env-model"
        os.environ["CHUNK_SIZE"] = "2000"

        config = Config(config_path)
        assert config.ollama_model == "env-model"
        assert config.chunk_size == 2000
    finally:
        os.unlink(config_path)
        os.environ.pop("OLLAMA_MODEL", None)
        os.environ.pop("CHUNK_SIZE", None)


def test_config_get_nested():
    """Test getting nested configuration values."""
    config_data = {
        "model": {"name": "test", "params": {"temperature": 0.7, "max_tokens": 100}},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        config = Config(config_path)
        assert config.get("model.params.temperature") == 0.7
        assert config.get("model.params.max_tokens") == 100
        assert config.get("nonexistent.key", "default") == "default"
    finally:
        os.unlink(config_path)


def test_config_file_not_found():
    """Test that missing config file raises error."""
    with pytest.raises(FileNotFoundError):
        Config("nonexistent_config.yaml")
