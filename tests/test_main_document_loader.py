"""Tests for main document loader."""

import time
from pathlib import Path

import pytest

from src.main_document_loader import MainDocumentLoader, get_main_document_loader


class TestMainDocumentLoader:
    """Test main document loading functionality."""

    def test_singleton_instance(self):
        """Test that get_main_document_loader returns singleton."""
        loader1 = get_main_document_loader()
        loader2 = get_main_document_loader()
        assert loader1 is loader2

    def test_load_markdown(self, tmp_path):
        """Test loading markdown file."""
        # Create test file
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test Profile\n\nThis is a test profile.")

        loader = MainDocumentLoader()
        loader.path = md_file
        loader.enabled = True

        content = loader.load_main_document()
        assert "Test Profile" in content
        assert "test profile" in content.lower()

    def test_load_text(self, tmp_path):
        """Test loading plain text file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Plain text profile\nWith multiple lines")

        loader = MainDocumentLoader()
        loader.path = txt_file
        loader.enabled = True

        content = loader.load_main_document()
        assert "Plain text profile" in content
        assert "multiple lines" in content

    def test_token_counting(self):
        """Test token counting functionality."""
        loader = MainDocumentLoader()

        text = "Hello world! " * 100
        tokens = loader.count_tokens(text)

        assert tokens > 0
        # Tokens should be less than characters
        assert tokens < len(text)
        # Rough sanity check (should be around 200-300 tokens)
        assert 100 < tokens < 500

    def test_truncation(self):
        """Test token truncation."""
        loader = MainDocumentLoader()

        text = "Test " * 10000
        truncated = loader.truncate_to_tokens(text, 100)

        truncated_tokens = loader.count_tokens(truncated)
        assert truncated_tokens <= 110  # Allow small margin
        assert "[... content truncated ...]" in truncated

    def test_truncation_no_change(self):
        """Test that short text is not truncated."""
        loader = MainDocumentLoader()

        text = "Short text"
        truncated = loader.truncate_to_tokens(text, 100)

        assert truncated == text
        assert "[... content truncated ...]" not in truncated

    def test_caching_enabled(self, tmp_path):
        """Test content caching functionality."""
        md_file = tmp_path / "test.md"
        md_file.write_text("Original content")

        loader = MainDocumentLoader()
        loader.path = md_file
        loader.enabled = True
        loader.cache_enabled = True

        # First load
        content1 = loader.load_main_document()
        assert content1 == "Original content"
        assert loader._cached_content is not None

        # Modify file but should still get cached version
        md_file.write_text("Modified content")

        # Second load (should use cache)
        content2 = loader.load_main_document()
        assert content2 == "Original content"  # Still cached
        assert loader._cached_content is not None

    def test_cache_invalidation_on_file_change(self, tmp_path):
        """Test that cache is invalidated when file changes after interval."""
        md_file = tmp_path / "test.md"
        md_file.write_text("Original content")

        loader = MainDocumentLoader()
        loader.path = md_file
        loader.enabled = True
        loader.cache_enabled = True
        loader.cache_check_interval = 0  # Force immediate check

        # First load
        content1 = loader.load_main_document()
        assert content1 == "Original content"

        # Modify file
        time.sleep(0.1)  # Small delay to ensure file modification time changes
        md_file.write_text("Modified content")

        # Second load (should detect change)
        content2 = loader.load_main_document()
        assert content2 == "Modified content"

    def test_manual_cache_invalidation(self, tmp_path):
        """Test manual cache invalidation."""
        md_file = tmp_path / "test.md"
        md_file.write_text("Original content")

        loader = MainDocumentLoader()
        loader.path = md_file
        loader.enabled = True
        loader.cache_enabled = True

        # Load and cache
        _ = loader.load_main_document()
        assert loader._cached_content is not None

        # Modify file
        md_file.write_text("Modified content")

        # Invalidate cache
        loader.invalidate_cache()
        assert loader._cached_content is None

        # Load again (should get new content)
        content2 = loader.load_main_document()
        assert content2 == "Modified content"

    def test_format_auto_detection(self, tmp_path):
        """Test auto-detection of file formats."""
        formats = {
            "test.md": "# Markdown",
            "test.txt": "Plain text",
        }

        loader = MainDocumentLoader()
        loader.enabled = True

        for filename, content in formats.items():
            file_path = tmp_path / filename
            file_path.write_text(content)
            loader.path = file_path
            loader.invalidate_cache()  # Clear cache between tests

            result = loader.load_main_document()
            assert len(result) > 0
            assert content in result

    def test_fail_silently_missing_file(self):
        """Test graceful failure when file is missing."""
        loader = MainDocumentLoader()
        loader.enabled = True
        loader.path = Path("/nonexistent/file.md")

        # Should not raise, just return empty string
        content = loader.load_main_document()
        assert content == ""

    def test_fail_not_silently(self):
        """Test that error is raised when fail_silently is False."""
        loader = MainDocumentLoader()
        loader.enabled = True
        loader.path = Path("/nonexistent/file.md")

        # Mock config to disable fail_silently
        original_get = loader.config.get

        def mock_get(key, default=None):
            if key == "main_document.fail_silently":
                return False
            return original_get(key, default)

        loader.config.get = mock_get

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            loader.load_main_document()

    def test_disabled_feature(self):
        """Test that disabled feature returns empty string."""
        loader = MainDocumentLoader()
        loader.enabled = False

        content = loader.load_main_document()
        assert content == ""

    def test_file_hash_calculation(self, tmp_path):
        """Test MD5 hash calculation for files."""
        md_file = tmp_path / "test.md"
        md_file.write_text("Test content")

        loader = MainDocumentLoader()
        loader.path = md_file

        hash1 = loader._calculate_file_hash()
        assert hash1 != ""
        assert len(hash1) == 32  # MD5 hash length

        # Same content should produce same hash
        hash2 = loader._calculate_file_hash()
        assert hash1 == hash2

        # Different content should produce different hash
        md_file.write_text("Different content")
        hash3 = loader._calculate_file_hash()
        assert hash3 != hash1

    def test_exceeds_token_limit_truncation(self, tmp_path):
        """Test that content exceeding token limit is truncated when summarization is disabled."""
        md_file = tmp_path / "test.md"
        # Create content that exceeds 100 tokens
        large_content = "This is a test sentence. " * 200
        md_file.write_text(large_content)

        loader = MainDocumentLoader()
        loader.path = md_file
        loader.enabled = True
        loader.max_tokens = 100
        loader.summarize_if_exceeds = False  # Disable summarization

        content = loader.load_main_document()

        # Should be truncated
        assert "[... content truncated ...]" in content
        assert loader.count_tokens(content) <= 110  # Allow small margin

    def test_cache_check_interval(self, tmp_path):
        """Test that cache respects check interval."""
        md_file = tmp_path / "test.md"
        md_file.write_text("Original content")

        loader = MainDocumentLoader()
        loader.path = md_file
        loader.enabled = True
        loader.cache_enabled = True
        loader.cache_check_interval = 3600  # 1 hour

        # First load
        content1 = loader.load_main_document()
        assert content1 == "Original content"

        # Modify file
        md_file.write_text("Modified content")

        # Second load within interval (should use cache)
        content2 = loader.load_main_document()
        assert content2 == "Original content"  # Still cached

    def test_empty_path(self):
        """Test behavior with empty path."""
        loader = MainDocumentLoader()
        loader.enabled = True
        loader.path = None

        content = loader.load_main_document()
        assert content == ""

    def test_unsupported_format_fallback(self, tmp_path):
        """Test that unsupported formats fall back to text loading."""
        unknown_file = tmp_path / "test.xyz"
        unknown_file.write_text("Content in unknown format")

        loader = MainDocumentLoader()
        loader.path = unknown_file
        loader.enabled = True

        content = loader.load_main_document()
        assert "Content in unknown format" in content
