"""Vector-based retrieval strategy using embeddings."""

import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..base import BaseRetrieverStrategy
from ..factory import RetrieverFactory

logger = logging.getLogger(__name__)


@RetrieverFactory.register("vector")
class VectorStrategy(BaseRetrieverStrategy):
    """Vector-based retrieval using semantic similarity.

    This strategy uses embedding models to convert documents and queries
    into dense vectors, then retrieves documents based on cosine similarity.
    It wraps the existing VectorStoreManager for backward compatibility.

    Configuration:
        retrieval.vector.search_type: 'similarity' or 'mmr'
        retrieval.vector.k: Number of documents to retrieve
        retrieval.vector.search_kwargs: Additional search parameters
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize vector strategy.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Lazy import to avoid circular dependencies
        from src.vectorstore import VectorStoreManager

        self.vectorstore_manager = VectorStoreManager()
        self._retriever: BaseRetriever | None = None

        # Get vector-specific config
        vector_config = config.get("retrieval", {}).get("vector", {})
        self.search_type = vector_config.get("search_type", "similarity")
        self.k = vector_config.get("k", 4)
        self.search_kwargs = vector_config.get("search_kwargs", {})

    @property
    def name(self) -> str:
        """Return strategy identifier."""
        return "vector"

    def build_index(self, documents: list[Document]) -> None:
        """Build vector index from documents.

        Args:
            documents: List of Document objects to index
        """
        logger.info(f"Building vector index with {len(documents)} documents...")
        self.vectorstore_manager.create_vectorstore(documents)
        self._is_initialized = True
        logger.info("Vector index built successfully")

    def load_index(self) -> bool:
        """Load existing vector index.

        Returns:
            True if loaded successfully
        """
        try:
            persist_dir = Path(self.vectorstore_manager.persist_directory)
            if not persist_dir.exists():
                logger.warning(f"Vector store not found at {persist_dir}")
                return False

            self.vectorstore_manager.load_vectorstore()
            self._is_initialized = True
            logger.info("Vector index loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading vector index: {e}")
            return False

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """Retrieve relevant documents using vector similarity.

        Args:
            query: Search query
            k: Number of documents to retrieve (uses config default if None)

        Returns:
            List of relevant documents
        """
        if not self._is_initialized:
            self.load_index()

        k = k or self.k
        return self.vectorstore_manager.similarity_search(query, k=k)

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Get LangChain-compatible retriever.

        Args:
            **kwargs: Override search parameters

        Returns:
            BaseRetriever instance
        """
        if not self._is_initialized:
            self.load_index()

        # Merge config with kwargs
        search_kwargs = {**self.search_kwargs, "k": self.k}
        search_kwargs.update(kwargs.get("search_kwargs", {}))

        return self.vectorstore_manager.get_retriever(
            search_type=kwargs.get("search_type", self.search_type),
            **search_kwargs,
        )

    def get_index_stats(self) -> dict[str, Any]:
        """Get vector index statistics.

        Returns:
            Dictionary with index statistics
        """
        stats = super().get_index_stats()

        if self._is_initialized and self.vectorstore_manager.vectorstore:
            try:
                collection_count = self.vectorstore_manager.vectorstore._collection.count()
                stats.update(
                    {
                        "num_documents": collection_count,
                        "persist_directory": self.vectorstore_manager.persist_directory,
                        "collection_name": self.vectorstore_manager.collection_name,
                        "search_type": self.search_type,
                    }
                )
            except Exception as e:
                logger.warning(f"Could not get collection stats: {e}")

        return stats
