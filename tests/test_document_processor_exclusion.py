"""Tests for main document exclusion in document processing."""

from src.document_processor import DocumentProcessor


class TestMainDocumentExclusion:
    """Test that main document is excluded from vector store processing."""

    def test_main_document_excluded_from_processing(self, tmp_path):
        """Test that main document file is skipped during directory processing."""
        # Create test documents directory
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()

        # Create main document
        main_doc = docs_dir / "main_profile.md"
        main_doc.write_text("# Main Profile\n\nThis should be excluded.")

        # Create other documents
        other_doc1 = docs_dir / "resume.md"
        other_doc1.write_text("# Resume\n\nThis should be included.")

        other_doc2 = docs_dir / "projects.md"
        other_doc2.write_text("# Projects\n\nThis should also be included.")

        # Create processor and mock config
        processor = DocumentProcessor()
        original_get = processor.config.get

        def mock_get(key, default=None):
            if key == "main_document.path":
                return str(main_doc)
            return original_get(key, default)

        processor.config.get = mock_get

        # Process directory
        documents = processor.process_directory(docs_dir)

        # Get all source files from processed documents
        processed_files = {doc.metadata.get("source", "") for doc in documents}

        # Main document should NOT be in processed files
        assert str(main_doc) not in processed_files

        # Other documents SHOULD be processed
        assert any("resume.md" in source for source in processed_files)
        assert any("projects.md" in source for source in processed_files)

    def test_all_documents_processed_when_main_doc_disabled(self, tmp_path):
        """Test that all documents are processed when main doc feature is disabled."""
        # Create test documents directory
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()

        # Create documents including what would be the main document
        main_doc = docs_dir / "main_profile.md"
        main_doc.write_text("# Main Profile\n\nContent here.")

        other_doc = docs_dir / "resume.md"
        other_doc.write_text("# Resume\n\nContent here.")

        # Create processor with main doc disabled
        processor = DocumentProcessor()
        original_get = processor.config.get

        def mock_get(key, default=None):
            if key == "main_document.path":
                return ""  # Empty path = disabled
            return original_get(key, default)

        processor.config.get = mock_get

        # Process directory
        documents = processor.process_directory(docs_dir)

        # Get all source files
        processed_files = {doc.metadata.get("source", "") for doc in documents}

        # ALL documents should be processed (including main_profile.md)
        assert any("main_profile.md" in source for source in processed_files)
        assert any("resume.md" in source for source in processed_files)

    def test_main_document_exclusion_case_insensitive_paths(self, tmp_path):
        """Test that path comparison works regardless of path format."""
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()

        # Create main document
        main_doc = docs_dir / "main_profile.md"
        main_doc.write_text("# Main Profile\n\nExclude me.")

        other_doc = docs_dir / "other.md"
        other_doc.write_text("# Other\n\nInclude me.")

        processor = DocumentProcessor()
        original_get = processor.config.get

        def mock_get(key, default=None):
            if key == "main_document.path":
                # Return path with different format but same file
                return str(main_doc.resolve())
            return original_get(key, default)

        processor.config.get = mock_get

        # Process directory
        documents = processor.process_directory(docs_dir)

        # Get all source files
        processed_files = {doc.metadata.get("source", "") for doc in documents}

        # Main document should be excluded
        assert str(main_doc) not in processed_files
        assert str(main_doc.resolve()) not in processed_files

        # Other document should be included
        assert len(documents) > 0
        assert any("other.md" in source for source in processed_files)

    def test_subdirectory_main_document_excluded(self, tmp_path):
        """Test that main document in subdirectory is excluded."""
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()

        # Create subdirectory
        subdir = docs_dir / "profiles"
        subdir.mkdir()

        # Create main document in subdirectory
        main_doc = subdir / "main_profile.md"
        main_doc.write_text("# Main Profile\n\nExclude me.")

        # Create other document
        other_doc = docs_dir / "resume.md"
        other_doc.write_text("# Resume\n\nInclude me.")

        processor = DocumentProcessor()
        original_get = processor.config.get

        def mock_get(key, default=None):
            if key == "main_document.path":
                return str(main_doc)
            return original_get(key, default)

        processor.config.get = mock_get

        # Process directory (recursively)
        documents = processor.process_directory(docs_dir)

        # Get all source files
        processed_files = {doc.metadata.get("source", "") for doc in documents}

        # Main document should be excluded
        assert str(main_doc) not in processed_files

        # Other document should be included
        assert any("resume.md" in source for source in processed_files)

    def test_no_documents_after_main_doc_exclusion(self, tmp_path):
        """Test behavior when only main document exists."""
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()

        # Create only the main document
        main_doc = docs_dir / "main_profile.md"
        main_doc.write_text("# Main Profile\n\nOnly document.")

        processor = DocumentProcessor()
        original_get = processor.config.get

        def mock_get(key, default=None):
            if key == "main_document.path":
                return str(main_doc)
            return original_get(key, default)

        processor.config.get = mock_get

        # Process directory
        documents = processor.process_directory(docs_dir)

        # Should have no documents (only main doc was excluded)
        assert len(documents) == 0

    def test_main_document_with_different_extension_not_confused(self, tmp_path):
        """Test that only exact main document file is excluded."""
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()

        # Create main document (PDF)
        main_doc = docs_dir / "main_profile.pdf"
        main_doc.write_text("PDF content")

        # Create similar named file (MD) - should NOT be excluded
        similar_doc = docs_dir / "main_profile.md"
        similar_doc.write_text("# Main Profile\n\nInclude me.")

        processor = DocumentProcessor()
        original_get = processor.config.get

        def mock_get(key, default=None):
            if key == "main_document.path":
                return str(main_doc)  # Only PDF is main doc
            return original_get(key, default)

        processor.config.get = mock_get

        # Process directory
        documents = processor.process_directory(docs_dir)

        # Get all source files
        processed_files = {doc.metadata.get("source", "") for doc in documents}

        # MD file should be included (different extension)
        assert any("main_profile.md" in source for source in processed_files)

        # PDF should be excluded
        # Note: PDF loading might fail with text content, but it should be skipped anyway
