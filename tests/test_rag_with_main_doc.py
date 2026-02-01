"""Integration tests for RAG pipeline with main document."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config_loader import get_config
from src.main_document_loader import reload_main_document_loader
from src.rag_pipeline import RAGPipeline


@pytest.fixture(autouse=True)
def ensure_fail_silently():
    """Ensure fail_silently is always set in config for all tests in this module."""
    import copy

    config = get_config()

    # Reload singleton at the start to ensure clean state
    reload_main_document_loader()

    # Save original config with deep copy to avoid reference issues
    original_main_doc_config = copy.deepcopy(config._config.get("main_document", {}))

    # Ensure fail_silently is set
    if "main_document" not in config._config:
        config._config["main_document"] = {}
    if "fail_silently" not in config._config["main_document"]:
        config._config["main_document"]["fail_silently"] = True

    yield

    # Restore original config after test
    if original_main_doc_config:
        config._config["main_document"] = original_main_doc_config
    elif "main_document" in config._config:
        # If there was no original config, remove what we added
        del config._config["main_document"]

    # Also reload the singleton to ensure it picks up the restored config
    reload_main_document_loader()


class TestRAGWithMainDocument:
    """Test RAG pipeline with main document integration."""

    def test_rag_pipeline_initialization(self):
        """Test that RAG pipeline initializes with main document loader."""
        pipeline = RAGPipeline()

        # Should have main_doc_loader attribute
        assert hasattr(pipeline, "main_doc_loader")
        assert hasattr(pipeline, "main_doc_content")

    def test_main_doc_info_disabled(self):
        """Test main document info when feature is disabled."""
        # Temporarily disable main document
        config = get_config()
        # Save the entire main_document config section
        original_main_doc_config = config._config.get("main_document", {}).copy()

        try:
            # Mock disabled state but preserve other config values
            config._config["main_document"] = original_main_doc_config.copy()
            config._config["main_document"]["enabled"] = False

            pipeline = RAGPipeline()
            info = pipeline.get_main_document_info()

            assert "enabled" in info
            assert "loaded" in info
            assert info["enabled"] is False
            assert info["loaded"] is False

        finally:
            # Restore original config completely
            if original_main_doc_config:
                config._config["main_document"] = original_main_doc_config

    def test_main_doc_info_enabled(self):
        """Test main document info when feature is enabled and loaded."""
        config = get_config()

        # Only run if main document is actually configured and exists
        main_doc_path = config.get("main_document.path", "")
        if not main_doc_path or not Path(main_doc_path).exists():
            pytest.skip("Main document not configured or doesn't exist")

        pipeline = RAGPipeline()

        # If main document is loaded
        if pipeline.main_doc_content:
            info = pipeline.get_main_document_info()

            assert info["enabled"] is True
            assert info["loaded"] is True
            assert "tokens" in info
            assert "path" in info
            assert "size_bytes" in info
            assert info["tokens"] > 0
            assert info["size_bytes"] > 0

    def test_token_budget_calculation(self):
        """Test token budget calculation."""
        pipeline = RAGPipeline()

        budget = pipeline._calculate_context_budget()

        # Check all required keys are present
        assert "model_context_window" in budget
        assert "main_doc_tokens" in budget
        assert "max_output_tokens" in budget
        assert "buffer_tokens" in budget
        assert "estimated_chat_history_tokens" in budget
        assert "available_for_retrieval" in budget
        assert "total_input_budget" in budget

        # Check reasonable values
        assert budget["model_context_window"] > 0
        assert budget["main_doc_tokens"] >= 0
        assert budget["max_output_tokens"] > 0
        assert budget["buffer_tokens"] >= 0
        assert budget["estimated_chat_history_tokens"] >= 0
        assert budget["available_for_retrieval"] >= 0

        # Check math (must account for estimated chat history tokens)
        expected_available = (
            budget["model_context_window"]
            - budget["main_doc_tokens"]
            - budget["max_output_tokens"]
            - budget["buffer_tokens"]
            - budget["estimated_chat_history_tokens"]
        )
        assert budget["available_for_retrieval"] == max(0, expected_available)

    def test_format_main_doc_empty(self):
        """Test _format_main_doc returns empty string when no content."""
        pipeline = RAGPipeline()
        pipeline.main_doc_content = ""

        formatted = pipeline._format_main_doc()
        assert formatted == ""

    def test_format_main_doc_with_content(self):
        """Test _format_main_doc formats content correctly."""
        pipeline = RAGPipeline()
        pipeline.main_doc_content = "Test profile content"

        formatted = pipeline._format_main_doc()

        assert "ESSENTIAL PROFILE INFORMATION" in formatted
        assert "Test profile content" in formatted
        assert "ADDITIONAL CONTEXT FROM DOCUMENTS" in formatted

    def test_reload_main_document(self, tmp_path):
        """Test runtime reloading of main document."""
        # Create a test main document
        test_doc = tmp_path / "test_profile.md"
        test_doc.write_text("# Test Profile\n\nOriginal content")

        # Temporarily override config
        config = get_config()
        # Save the entire main_document config section
        original_main_doc_config = config._config.get("main_document", {}).copy()

        try:
            config._config["main_document"] = {
                "enabled": True,
                "path": str(test_doc),
                "max_tokens": 10000,
                "summarize_if_exceeds": True,
                "cache_enabled": True,
                "cache_check_interval": 60,
                "fail_silently": True,  # Add missing key
                "fallback_to_vectordb_only": True,
            }

            # Reload the singleton to pick up new config
            reload_main_document_loader()

            pipeline = RAGPipeline()

            # Initial content
            assert "Original content" in pipeline.main_doc_content

            # Modify file
            test_doc.write_text("# Test Profile\n\nModified content")

            # Reload
            success = pipeline.reload_main_document()
            assert success is True
            assert "Modified content" in pipeline.main_doc_content

        finally:
            # Restore original config completely
            if original_main_doc_config:
                config._config["main_document"] = original_main_doc_config

    def test_reload_main_document_error_handling(self):
        """Test reload_main_document handles errors gracefully."""
        pipeline = RAGPipeline()

        # Set invalid path
        pipeline.main_doc_loader.path = Path("/nonexistent/path/file.md")

        # Reload should succeed (with empty content) when fail_silently is True
        success = pipeline.reload_main_document()
        assert success is True
        # But content should be empty since file doesn't exist
        assert pipeline.main_doc_content == ""

    def test_query_with_main_doc(self):
        """Test that query works with main document enabled."""
        config = get_config()

        # Only run if main document is configured
        main_doc_enabled = config.get("main_document.enabled", False)
        if not main_doc_enabled:
            pytest.skip("Main document not enabled in config")

        pipeline = RAGPipeline()

        # Simple test query
        response = pipeline.query("What is the name?")

        assert "result" in response
        assert isinstance(response["result"], str)
        assert len(response["result"]) > 0

    def test_format_docs_empty_list(self):
        """Test _format_docs handles empty list."""
        pipeline = RAGPipeline()

        formatted = pipeline._format_docs([])
        assert formatted == ""

    def test_context_budget_warning_large_main_doc(self, caplog):
        """Test that warning is logged when main doc uses >50% of context."""
        pipeline = RAGPipeline()

        # Simulate large main document
        original_content = pipeline.main_doc_content
        pipeline.main_doc_content = "Test " * 10000  # Very large content

        try:
            import logging

            with caplog.at_level(logging.WARNING):
                budget = pipeline._calculate_context_budget()

            # Check if warning was logged (only if main doc is actually large)
            if budget["main_doc_tokens"] > budget["model_context_window"] * 0.5:
                assert any("uses" in record.message for record in caplog.records)

        finally:
            # Restore original content
            pipeline.main_doc_content = original_content

    def test_model_context_window_detection(self):
        """Test that different models get correct context windows."""
        config = get_config()
        original_model = config.get("llm.model", "")

        test_cases = {
            "llama3.2:3b": 8192,
            "llama3.1:8b": 128000,
            "phi3:mini": 4096,
            "gemma2:2b": 8192,
            "unknown:model": 8192,  # Default
        }

        # Mock LLM handler to avoid actual LLM initialization
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_handler.get_llm.return_value = MagicMock()
            mock_llm_handler.return_value = mock_handler

            for model_name, expected_window in test_cases.items():
                config._config["llm"]["model"] = model_name

                pipeline = RAGPipeline()
                budget = pipeline._calculate_context_budget()

                assert budget["model_context_window"] == expected_window

        # Restore original model
        config._config["llm"]["model"] = original_model


class TestMainDocumentContextInsertion:
    """Test that main document content is properly inserted into prompts."""

    def test_prompt_template_structure(self):
        """Test that prompt template includes main document placeholder."""
        # Mock LLM handler to avoid actual LLM initialization
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_handler.get_llm.return_value = MagicMock()
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()

            template_str = pipeline.prompt_template.messages[0].prompt.template

            # Should contain main_document_section placeholder
            assert "{main_document_section}" in template_str
            assert "{context}" in template_str
            assert "{question}" in template_str

    def test_main_doc_position_before_context(self):
        """Test that main document appears BEFORE vectordb context in template."""
        # Mock LLM handler to avoid actual LLM initialization
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_handler.get_llm.return_value = MagicMock()
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()

            template_str = pipeline.prompt_template.messages[0].prompt.template

            main_doc_pos = template_str.find("{main_document_section}")
            context_pos = template_str.find("{context}")

            # Main document should come before context
            assert main_doc_pos < context_pos
