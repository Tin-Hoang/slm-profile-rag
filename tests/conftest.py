"""Pytest configuration and fixtures for all tests.

This module provides global test fixtures that are automatically applied
to all tests in the test suite.

Environment Variables:
    SKIP_LLM_MOCK: Set to "true" to disable LLM mocking for integration tests
                   that require actual Ollama connection.

Example:
    # Run tests with actual Ollama (requires Ollama to be running)
    SKIP_LLM_MOCK=true pytest tests/

    # Run tests with mocked LLM (default, works in CI/CD)
    pytest tests/
"""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_llm_handler():
    """Mock LLM handler to avoid requiring Ollama in tests.

    This fixture is automatically used for all tests unless explicitly disabled
    via the SKIP_LLM_MOCK environment variable. It prevents tests from trying
    to connect to Ollama, which may not be available in CI/CD environments
    like GitHub Actions.

    The mock provides:
    - A mock LLM that returns test responses
    - A mock system prompt
    - All necessary methods to simulate LLM behavior

    This allows tests to focus on business logic without requiring a running
    Ollama instance.
    """
    # Check if we should skip mocking (e.g., for integration tests)
    if os.environ.get("SKIP_LLM_MOCK") == "true":
        yield
        return

    with patch("src.rag_pipeline.get_llm_handler") as mock_get_handler:
        # Create a mock LLM handler
        mock_handler = MagicMock()

        # Create a mock LLM that returns simple responses
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "This is a test response from the mocked LLM."
        mock_llm.ainvoke.return_value = "This is a test response from the mocked LLM."

        # Configure the mock handler to return the mock LLM
        mock_handler.get_llm.return_value = mock_llm
        mock_handler.get_system_prompt.return_value = (
            "You are a helpful AI assistant. Answer questions based on the provided context."
        )

        # Make get_llm_handler return our mock
        mock_get_handler.return_value = mock_handler

        yield mock_handler
