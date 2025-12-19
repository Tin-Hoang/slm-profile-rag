"""Base classes for retrieval strategies."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


class BaseRetrieverStrategy(ABC):
    """Abstract base class for all retrieval strategies.

    This provides a common interface for different retrieval approaches:
    - Vector-based (semantic similarity)
    - BM25 (lexical/keyword matching)
    - Hybrid combinations
    - Future strategies (PageIndex, GraphRAG, etc.)

    Subclasses must implement all abstract methods to be usable
    with the RetrieverFactory.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the retrieval strategy.

        Args:
            config: Configuration dictionary for the strategy
        """
        self.config = config
        self._is_initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier for this strategy.

        Returns:
            Strategy name (e.g., 'vector', 'bm25', 'bm25_vector')
        """
        pass

    @abstractmethod
    def build_index(self, documents: list[Document]) -> None:
        """Build or rebuild the index from documents.

        This method should process documents and create the necessary
        index structures for retrieval.

        Args:
            documents: List of Document objects to index
        """
        pass

    @abstractmethod
    def load_index(self) -> bool:
        """Load an existing index from persistent storage.

        Returns:
            True if index was loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query string
            k: Number of documents to retrieve

        Returns:
            List of relevant Document objects
        """
        pass

    @abstractmethod
    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a LangChain-compatible retriever.

        Args:
            **kwargs: Additional arguments for the retriever

        Returns:
            A BaseRetriever instance compatible with LangChain chains
        """
        pass

    def is_index_ready(self) -> bool:
        """Check if the index is built and ready for queries.

        Returns:
            True if index is ready, False otherwise
        """
        return self._is_initialized

    def get_index_stats(self) -> dict[str, Any]:
        """Get statistics about the current index.

        Returns:
            Dictionary with index statistics (implementation-specific)
        """
        return {
            "strategy": self.name,
            "initialized": self._is_initialized,
        }


class StrategyRetriever(BaseRetriever):
    """LangChain-compatible retriever wrapper for BaseRetrieverStrategy.

    This adapter allows any BaseRetrieverStrategy to be used
    in LangChain LCEL chains and pipelines.
    """

    strategy: BaseRetrieverStrategy
    search_kwargs: dict[str, Any] = {}

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,  # noqa: ARG002
    ) -> list[Document]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for
            run_manager: Callback manager for the retriever run

        Returns:
            List of relevant documents
        """
        k = self.search_kwargs.get("k", 4)
        return self.strategy.retrieve(query, k=k)
