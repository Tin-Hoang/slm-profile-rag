"""Tests for document processor module."""

import tempfile
from pathlib import Path

from slm_profile_rag.document_processor import DocumentProcessor


def test_document_processor_initialization():
    """Test document processor initialization."""
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    assert processor.chunk_size == 500
    assert processor.chunk_overlap == 50


def test_load_text_documents():
    """Test loading text documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample text file
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("This is a test document with some content.")

        processor = DocumentProcessor()
        documents = processor.load_documents(tmpdir)

        assert len(documents) > 0
        assert "test document" in documents[0].page_content.lower()


def test_load_markdown_documents():
    """Test loading markdown documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample markdown file
        test_file = Path(tmpdir) / "test.md"
        test_file.write_text("# Test Header\n\nThis is markdown content.")

        processor = DocumentProcessor()
        documents = processor.load_documents(tmpdir)

        assert len(documents) > 0


def test_split_documents():
    """Test document splitting."""
    from langchain_core.documents import Document

    processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)

    # Create a long document
    long_text = " ".join([f"Sentence {i}." for i in range(100)])
    docs = [Document(page_content=long_text)]

    split_docs = processor.split_documents(docs)

    # Should be split into multiple chunks
    assert len(split_docs) > 1
    assert all(len(doc.page_content) <= 60 for doc in split_docs)  # Some margin for splitting


def test_directory_not_found():
    """Test handling of non-existent directory."""
    processor = DocumentProcessor()

    try:
        processor.load_documents("/nonexistent/directory")
        raise AssertionError("Should raise FileNotFoundError")
    except FileNotFoundError:
        pass
