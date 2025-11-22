"""Document processing module for loading and splitting documents."""

from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Process and load documents for the RAG system."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize document processor.

        Args:
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_documents(self, directory_path: str) -> list[Document]:
        """Load documents from a directory.

        Args:
            directory_path: Path to directory containing documents.

        Returns:
            List of loaded documents.
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        documents = []

        # Load PDF files
        pdf_files = list(directory.glob("**/*.pdf"))
        for pdf_file in pdf_files:
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())

        # Load HTML files
        html_files = list(directory.glob("**/*.html"))
        for html_file in html_files:
            loader = UnstructuredHTMLLoader(str(html_file))
            documents.extend(loader.load())

        # Load text files (txt, md)
        text_patterns = ["**/*.txt", "**/*.md"]
        for pattern in text_patterns:
            text_files = list(directory.glob(pattern))
            for text_file in text_files:
                loader = TextLoader(str(text_file))
                documents.extend(loader.load())

        return documents

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks.

        Args:
            documents: List of documents to split.

        Returns:
            List of split document chunks.
        """
        return self.text_splitter.split_documents(documents)

    def process_documents(self, directory_path: str) -> list[Document]:
        """Load and split documents from a directory.

        Args:
            directory_path: Path to directory containing documents.

        Returns:
            List of processed document chunks.
        """
        documents = self.load_documents(directory_path)
        if not documents:
            raise ValueError(f"No documents found in {directory_path}")

        return self.split_documents(documents)
