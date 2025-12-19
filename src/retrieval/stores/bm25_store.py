"""BM25 index storage and retrieval."""

import hashlib
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Store:
    """Storage and retrieval for BM25 index.

    This class manages a BM25 index for lexical/keyword search.
    It handles tokenization, index building, persistence, and retrieval.

    Attributes:
        persist_path: Directory to store the index
        tokenizer: Tokenization method ('simple' or 'nltk')
    """

    def __init__(
        self,
        persist_path: str = "./bm25_index",
        tokenizer: str = "simple",
    ):
        """Initialize BM25 store.

        Args:
            persist_path: Directory to store the index files
            tokenizer: Tokenization method ('simple' or 'nltk')
        """
        self.persist_path = Path(persist_path)
        self.tokenizer_type = tokenizer

        self.bm25: BM25Okapi | None = None
        self.documents: list[Document] = []
        self.tokenized_corpus: list[list[str]] = []
        self._doc_hash: str | None = None

    def _simple_tokenize(self, text: str) -> list[str]:
        """Simple whitespace and punctuation-based tokenization.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase and split on non-alphanumeric characters
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text using configured tokenizer.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        if self.tokenizer_type == "nltk":
            try:
                import nltk

                # Ensure punkt tokenizer is available
                try:
                    nltk.data.find("tokenizers/punkt")
                except LookupError:
                    nltk.download("punkt", quiet=True)

                tokens = nltk.word_tokenize(text.lower())
                return [t for t in tokens if t.isalnum()]
            except ImportError:
                logger.warning("NLTK not installed, falling back to simple tokenizer")
                return self._simple_tokenize(text)
        else:
            return self._simple_tokenize(text)

    def _compute_hash(self, documents: list[Document]) -> str:
        """Compute hash of document contents for change detection.

        Args:
            documents: List of documents

        Returns:
            Hash string
        """
        content = "".join(doc.page_content for doc in documents)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def build_index(self, documents: list[Document]) -> None:
        """Build BM25 index from documents.

        Args:
            documents: List of Document objects to index
        """
        if not documents:
            raise ValueError("No documents provided for indexing")

        logger.info(f"Building BM25 index with {len(documents)} documents...")

        # Store documents
        self.documents = documents

        # Tokenize all documents
        self.tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Compute hash for change detection
        self._doc_hash = self._compute_hash(documents)

        logger.info(f"BM25 index built successfully with {len(documents)} documents")

    def save(self) -> None:
        """Save BM25 index to disk."""
        if self.bm25 is None:
            raise ValueError("No index to save. Build index first.")

        # Create directory if needed
        self.persist_path.mkdir(parents=True, exist_ok=True)

        # Save BM25 index
        index_path = self.persist_path / "bm25_index.pkl"
        with open(index_path, "wb") as f:
            pickle.dump(self.bm25, f)

        # Save documents
        docs_path = self.persist_path / "documents.pkl"
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

        # Save tokenized corpus (for rebuilding if needed)
        corpus_path = self.persist_path / "tokenized_corpus.pkl"
        with open(corpus_path, "wb") as f:
            pickle.dump(self.tokenized_corpus, f)

        # Save metadata
        metadata = {
            "doc_hash": self._doc_hash,
            "num_documents": len(self.documents),
            "tokenizer": self.tokenizer_type,
        }
        metadata_path = self.persist_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"BM25 index saved to {self.persist_path}")

    def load(self) -> bool:
        """Load BM25 index from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = self.persist_path / "bm25_index.pkl"
        docs_path = self.persist_path / "documents.pkl"
        corpus_path = self.persist_path / "tokenized_corpus.pkl"
        metadata_path = self.persist_path / "metadata.json"

        # Check if all required files exist
        if not all(p.exists() for p in [index_path, docs_path, corpus_path, metadata_path]):
            logger.warning(f"BM25 index not found at {self.persist_path}")
            return False

        try:
            # Load BM25 index
            with open(index_path, "rb") as f:
                self.bm25 = pickle.load(f)

            # Load documents
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)

            # Load tokenized corpus
            with open(corpus_path, "rb") as f:
                self.tokenized_corpus = pickle.load(f)

            # Load metadata
            with open(metadata_path) as f:
                metadata = json.load(f)
                self._doc_hash = metadata.get("doc_hash")

            logger.info(
                f"BM25 index loaded from {self.persist_path} ({len(self.documents)} documents)"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading BM25 index: {e}")
            return False

    def search(self, query: str, k: int = 4) -> list[Document]:
        """Search the BM25 index.

        Args:
            query: Search query
            k: Number of documents to return

        Returns:
            List of relevant documents
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built or loaded")

        # Tokenize query
        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            logger.warning("Query tokenization produced no tokens")
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        # Return documents
        results = [self.documents[i] for i in top_indices if scores[i] > 0]

        logger.debug(f"BM25 search returned {len(results)} documents for query: {query[:50]}...")

        return results

    def search_with_scores(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Search the BM25 index and return scores.

        Args:
            query: Search query
            k: Number of documents to return

        Returns:
            List of (document, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built or loaded")

        # Tokenize query
        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top k indices with scores
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        # Return documents with scores
        return [(self.documents[i], scores[i]) for i in top_indices if scores[i] > 0]

    def is_built(self) -> bool:
        """Check if index is built.

        Returns:
            True if index is ready
        """
        return self.bm25 is not None and len(self.documents) > 0

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary with index statistics
        """
        return {
            "num_documents": len(self.documents),
            "tokenizer": self.tokenizer_type,
            "persist_path": str(self.persist_path),
            "doc_hash": self._doc_hash,
            "is_built": self.is_built(),
        }

    def delete(self) -> None:
        """Delete the persisted index."""
        import shutil

        if self.persist_path.exists():
            shutil.rmtree(self.persist_path)
            logger.info(f"Deleted BM25 index at {self.persist_path}")

        self.bm25 = None
        self.documents = []
        self.tokenized_corpus = []
        self._doc_hash = None
