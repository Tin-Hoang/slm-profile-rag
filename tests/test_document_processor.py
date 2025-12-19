"""Tests for document processor module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.document_processor import DocumentProcessor, process_documents


@pytest.fixture
def mock_config():
    """Create a mock config."""
    with patch("src.document_processor.get_config") as mock:
        mock.return_value.get.side_effect = lambda key, default=None: {
            "document_processing.chunk_size": 1000,
            "document_processing.chunk_overlap": 200,
            "document_processing.supported_extensions": [
                ".pdf",
                ".docx",
                ".doc",
                ".html",
                ".htm",
                ".txt",
                ".md",
            ],
            "main_document.path": "",
        }.get(key, default)
        yield mock


@pytest.fixture
def processor(mock_config):
    """Create a DocumentProcessor instance."""
    return DocumentProcessor()


# --- DocumentProcessor Tests ---


def test_processor_init(processor):
    """Test DocumentProcessor initialization."""
    assert processor is not None
    assert processor.chunk_size == 1000
    assert processor.chunk_overlap == 200
    assert len(processor.supported_extensions) > 0


def test_load_pdf(processor):
    """Test loading PDF file."""
    with patch("src.document_processor.PyPDFLoader") as mock_loader:
        mock_docs = [
            Document(page_content="Page 1 content", metadata={"page": 0}),
            Document(page_content="Page 2 content", metadata={"page": 1}),
        ]
        mock_loader.return_value.load.return_value = mock_docs

        result = processor.load_pdf(Path("/test/document.pdf"))

        assert len(result) == 2
        assert result[0].page_content == "Page 1 content"


def test_load_pdf_error(processor):
    """Test loading PDF with error."""
    with patch("src.document_processor.PyPDFLoader") as mock_loader:
        mock_loader.return_value.load.side_effect = Exception("PDF error")

        result = processor.load_pdf(Path("/test/document.pdf"))

        assert result == []


def test_load_docx(processor):
    """Test loading DOCX file."""
    with patch("src.document_processor.DocxDocument") as mock_docx:
        mock_doc = MagicMock()
        mock_paragraph1 = MagicMock()
        mock_paragraph1.text = "First paragraph"
        mock_paragraph2 = MagicMock()
        mock_paragraph2.text = "Second paragraph"
        mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2]
        mock_doc.tables = []
        mock_docx.return_value = mock_doc

        result = processor.load_docx(Path("/test/document.docx"))

        assert len(result) == 1
        assert "First paragraph" in result[0].page_content
        assert "Second paragraph" in result[0].page_content


def test_load_docx_with_tables(processor):
    """Test loading DOCX file with tables."""
    with patch("src.document_processor.DocxDocument") as mock_docx:
        mock_doc = MagicMock()
        mock_paragraph = MagicMock()
        mock_paragraph.text = "Document content"
        mock_doc.paragraphs = [mock_paragraph]

        # Create mock table
        mock_cell1 = MagicMock()
        mock_cell1.text = "Cell 1"
        mock_cell2 = MagicMock()
        mock_cell2.text = "Cell 2"
        mock_row = MagicMock()
        mock_row.cells = [mock_cell1, mock_cell2]
        mock_table = MagicMock()
        mock_table.rows = [mock_row]
        mock_doc.tables = [mock_table]
        mock_docx.return_value = mock_doc

        result = processor.load_docx(Path("/test/document.docx"))

        assert len(result) == 1
        assert "Cell 1" in result[0].page_content
        assert "Cell 2" in result[0].page_content


def test_load_docx_error(processor):
    """Test loading DOCX with error."""
    with patch("src.document_processor.DocxDocument") as mock_docx:
        mock_docx.side_effect = Exception("DOCX error")

        result = processor.load_docx(Path("/test/document.docx"))

        assert result == []


def test_load_html(processor):
    """Test loading HTML file."""
    html_content = """
    <html>
    <head><style>body { color: red; }</style></head>
    <body>
        <script>console.log("test");</script>
        <h1>Title</h1>
        <p>Content paragraph</p>
    </body>
    </html>
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html_content)
        f.flush()

        result = processor.load_html(Path(f.name))

        assert len(result) == 1
        assert "Title" in result[0].page_content
        assert "Content paragraph" in result[0].page_content
        # Script content should be removed
        assert "console.log" not in result[0].page_content


def test_load_html_error(processor):
    """Test loading HTML with error."""
    result = processor.load_html(Path("/nonexistent/file.html"))

    assert result == []


def test_load_text(processor):
    """Test loading text file."""
    with patch("src.document_processor.TextLoader") as mock_loader:
        mock_docs = [Document(page_content="Text content", metadata={"source": "test.txt"})]
        mock_loader.return_value.load.return_value = mock_docs

        result = processor.load_text(Path("/test/document.txt"))

        assert len(result) == 1
        assert result[0].page_content == "Text content"


def test_load_text_error(processor):
    """Test loading text with error."""
    with patch("src.document_processor.TextLoader") as mock_loader:
        mock_loader.return_value.load.side_effect = Exception("Text error")

        result = processor.load_text(Path("/test/document.txt"))

        assert result == []


def test_load_document_pdf(processor):
    """Test load_document routes to correct loader for PDF."""
    with patch.object(processor, "load_pdf") as mock_load:
        mock_load.return_value = [Document(page_content="PDF")]

        result = processor.load_document(Path("/test/file.pdf"))

        mock_load.assert_called_once()
        assert len(result) == 1


def test_load_document_docx(processor):
    """Test load_document routes to correct loader for DOCX."""
    with patch.object(processor, "load_docx") as mock_load:
        mock_load.return_value = [Document(page_content="DOCX")]

        result = processor.load_document(Path("/test/file.docx"))

        mock_load.assert_called_once()
        assert len(result) == 1


def test_load_document_doc(processor):
    """Test load_document routes to correct loader for DOC."""
    with patch.object(processor, "load_docx") as mock_load:
        mock_load.return_value = [Document(page_content="DOC")]

        processor.load_document(Path("/test/file.doc"))

        mock_load.assert_called_once()


def test_load_document_html(processor):
    """Test load_document routes to correct loader for HTML."""
    with patch.object(processor, "load_html") as mock_load:
        mock_load.return_value = [Document(page_content="HTML")]

        processor.load_document(Path("/test/file.html"))

        mock_load.assert_called_once()


def test_load_document_htm(processor):
    """Test load_document routes to correct loader for HTM."""
    with patch.object(processor, "load_html") as mock_load:
        mock_load.return_value = [Document(page_content="HTM")]

        processor.load_document(Path("/test/file.htm"))

        mock_load.assert_called_once()


def test_load_document_txt(processor):
    """Test load_document routes to correct loader for TXT."""
    with patch.object(processor, "load_text") as mock_load:
        mock_load.return_value = [Document(page_content="TXT")]

        processor.load_document(Path("/test/file.txt"))

        mock_load.assert_called_once()


def test_load_document_md(processor):
    """Test load_document routes to correct loader for MD."""
    with patch.object(processor, "load_text") as mock_load:
        mock_load.return_value = [Document(page_content="MD")]

        processor.load_document(Path("/test/file.md"))

        mock_load.assert_called_once()


def test_load_document_unsupported(processor):
    """Test load_document returns empty for unsupported extensions."""
    result = processor.load_document(Path("/test/file.xyz"))

    assert result == []


def test_chunk_documents_empty(processor):
    """Test chunking empty document list."""
    result = processor.chunk_documents([])

    assert result == []


def test_chunk_documents(processor):
    """Test chunking documents."""
    documents = [
        Document(
            page_content="A" * 2000,  # Long content that will be split
            metadata={"source": "test.txt"},
        )
    ]

    result = processor.chunk_documents(documents)

    assert len(result) > 0  # Should be split into multiple chunks


def test_chunk_documents_error(processor):
    """Test chunking with error."""
    with patch.object(processor.text_splitter, "split_documents") as mock_split:
        mock_split.side_effect = Exception("Chunk error")

        result = processor.chunk_documents([Document(page_content="test")])

        assert result == []


def test_process_directory_not_found(processor):
    """Test processing non-existent directory."""
    result = processor.process_directory("/nonexistent/directory")

    assert result == []


def test_process_directory_empty(processor):
    """Test processing empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = processor.process_directory(tmpdir)

        assert result == []


def test_process_directory_with_files(processor):
    """Test processing directory with files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test markdown file
        test_file = Path(tmpdir) / "test.md"
        test_file.write_text("# Test Document\n\nThis is test content.")

        with patch.object(processor, "load_text") as mock_load:
            mock_load.return_value = [
                Document(
                    page_content="# Test Document\n\nThis is test content.",
                    metadata={"source": str(test_file)},
                )
            ]

            processor.process_directory(tmpdir)

            # Should process and chunk the file
            mock_load.assert_called()


def test_process_directory_excludes_main_doc():
    """Test processing directory excludes main document."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        main_doc = Path(tmpdir) / "main_profile.md"
        main_doc.write_text("# Main Profile")
        other_doc = Path(tmpdir) / "other.md"
        other_doc.write_text("# Other Document")

        with patch("src.document_processor.get_config") as mock_config:
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                "document_processing.chunk_size": 1000,
                "document_processing.chunk_overlap": 200,
                "document_processing.supported_extensions": [".md"],
                "main_document.path": str(main_doc),
            }.get(key, default)

            processor = DocumentProcessor()

            with patch.object(processor, "load_text") as mock_load:
                mock_load.return_value = [Document(page_content="Other content", metadata={})]

                processor.process_directory(tmpdir)

                # Should only load other.md, not main_profile.md
                # Verify main_doc was skipped by checking call args
                for call in mock_load.call_args_list:
                    assert str(main_doc) not in str(call)


# --- process_documents Function Tests ---


def test_process_documents_default_path():
    """Test process_documents uses default path from config."""
    with patch("src.document_processor.get_config") as mock_config:
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "document_processing.chunk_size": 1000,
            "document_processing.chunk_overlap": 200,
            "document_processing.supported_extensions": [".md"],
            "main_document.path": "",
        }.get(key, default)
        mock_config.return_value.get_env.return_value = "/default/path"

        with patch("src.document_processor.DocumentProcessor") as mock_processor:
            mock_processor.return_value.process_directory.return_value = []

            process_documents()

            mock_processor.return_value.process_directory.assert_called_once_with("/default/path")


def test_process_documents_custom_path():
    """Test process_documents uses custom path."""
    with patch("src.document_processor.get_config") as mock_config:
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "document_processing.chunk_size": 1000,
            "document_processing.chunk_overlap": 200,
            "document_processing.supported_extensions": [".md"],
            "main_document.path": "",
        }.get(key, default)

        with patch("src.document_processor.DocumentProcessor") as mock_processor:
            mock_processor.return_value.process_directory.return_value = []

            process_documents("/custom/path")

            mock_processor.return_value.process_directory.assert_called_once_with("/custom/path")
