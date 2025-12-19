"""BM25-based retrieval strategy for lexical/keyword search."""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..base import BaseRetrieverStrategy, StrategyRetriever
from ..factory import RetrieverFactory
from ..stores.bm25_store import BM25Store

logger = logging.getLogger(__name__)


@RetrieverFactory.register("bm25")
class BM25Strategy(BaseRetrieverStrategy):
    """BM25-based retrieval using lexical/keyword matching.

    This strategy uses the BM25 algorithm (Best Matching 25) for
    traditional keyword-based document retrieval. It's effective for:
    - Exact term matching
    - Abbreviations and acronyms
    - Proper nouns and technical terms
    - Queries with specific keywords

    Configuration:
        retrieval.bm25.k: Number of documents to retrieve
        retrieval.bm25.persist_path: Path to store BM25 index
        retrieval.bm25.tokenizer: Tokenization method ('simple' or 'nltk')
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize BM25 strategy.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Get BM25-specific config
        bm25_config = config.get("retrieval", {}).get("bm25", {})
        self.k = bm25_config.get("k", 10)
        self.persist_path = bm25_config.get("persist_path", "./bm25_index")
        self.tokenizer = bm25_config.get("tokenizer", "simple")

        # Initialize BM25 store
        self.bm25_store = BM25Store(
            persist_path=self.persist_path,
            tokenizer=self.tokenizer,
        )

    @property
    def name(self) -> str:
        """Return strategy identifier."""
        return "bm25"

    def build_index(self, documents: list[Document]) -> None:
        """Build BM25 index from documents.

        Args:
            documents: List of Document objects to index
        """
        logger.info(f"Building BM25 index with {len(documents)} documents...")
        self.bm25_store.build_index(documents)
        self.bm25_store.save()
        self._is_initialized = True
        logger.info("BM25 index built and saved successfully")

    def load_index(self) -> bool:
        """Load existing BM25 index.

        Returns:
            True if loaded successfully
        """
        if self.bm25_store.load():
            self._is_initialized = True
            return True
        return False

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """Retrieve relevant documents using BM25.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            List of relevant documents
        """
        if not self._is_initialized and not self.load_index():
            raise ValueError("BM25 index not available. Build index first.")

        k = k or self.k
        return self.bm25_store.search(query, k=k)

    def retrieve_with_scores(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with BM25 scores.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            List of (document, score) tuples
        """
        if not self._is_initialized and not self.load_index():
            raise ValueError("BM25 index not available. Build index first.")

        k = k or self.k
        return self.bm25_store.search_with_scores(query, k=k)

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Get LangChain-compatible retriever.

        Args:
            **kwargs: Override search parameters

        Returns:
            BaseRetriever instance
        """
        if not self._is_initialized:
            self.load_index()

        search_kwargs = {"k": kwargs.get("k", self.k)}

        return StrategyRetriever(
            strategy=self,
            search_kwargs=search_kwargs,
        )

    def get_index_stats(self) -> dict[str, Any]:
        """Get BM25 index statistics.

        Returns:
            Dictionary with index statistics
        """
        stats = super().get_index_stats()
        stats.update(self.bm25_store.get_stats())
        return stats

    def delete_index(self) -> None:
        """Delete the persisted BM25 index."""
        self.bm25_store.delete()
        self._is_initialized = False
