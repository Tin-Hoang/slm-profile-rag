"""Hybrid BM25 + Vector retrieval strategy."""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..base import BaseRetrieverStrategy
from ..factory import RetrieverFactory
from ..fusion import FusionRetriever, reciprocal_rank_fusion
from .bm25 import BM25Strategy
from .vector import VectorStrategy

logger = logging.getLogger(__name__)


@RetrieverFactory.register("bm25_vector")
class BM25VectorStrategy(BaseRetrieverStrategy):
    """Hybrid retrieval combining BM25 and Vector search.

    This strategy combines lexical (BM25) and semantic (vector) search
    using fusion algorithms like Reciprocal Rank Fusion (RRF).

    Benefits:
    - Better recall: catches both keyword matches and semantic similarities
    - More robust: compensates for weaknesses of individual methods
    - Handles diverse queries: technical terms, natural language, abbreviations

    Configuration:
        retrieval.vector.*: Vector search settings
        retrieval.bm25.*: BM25 search settings
        retrieval.fusion.algorithm: 'rrf' or 'weighted'
        retrieval.fusion.rrf_k: RRF constant (default: 60)
        retrieval.fusion.weights.vector: Weight for vector results
        retrieval.fusion.weights.bm25: Weight for BM25 results
        retrieval.final_k: Final number of documents to return
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize hybrid BM25+Vector strategy.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Initialize sub-strategies
        self.vector_strategy = VectorStrategy(config)
        self.bm25_strategy = BM25Strategy(config)

        # Get fusion config
        fusion_config = config.get("retrieval", {}).get("fusion", {})
        self.fusion_algorithm = fusion_config.get("algorithm", "rrf")
        self.rrf_k = fusion_config.get("rrf_k", 60)

        # Get weights
        weights_config = fusion_config.get("weights", {})
        self.vector_weight = weights_config.get("vector", 0.7)
        self.bm25_weight = weights_config.get("bm25", 0.3)

        # Final k
        self.final_k = config.get("retrieval", {}).get("final_k", 4)

        # Individual retrieval k values
        vector_config = config.get("retrieval", {}).get("vector", {})
        bm25_config = config.get("retrieval", {}).get("bm25", {})
        self.vector_k = vector_config.get("k", 10)
        self.bm25_k = bm25_config.get("k", 10)

    @property
    def name(self) -> str:
        """Return strategy identifier."""
        return "bm25_vector"

    def build_index(self, documents: list[Document]) -> None:
        """Build both BM25 and Vector indexes.

        Args:
            documents: List of Document objects to index
        """
        logger.info(f"Building hybrid index with {len(documents)} documents...")

        # Build both indexes
        self.vector_strategy.build_index(documents)
        self.bm25_strategy.build_index(documents)

        self._is_initialized = True
        logger.info("Hybrid BM25+Vector index built successfully")

    def load_index(self) -> bool:
        """Load both BM25 and Vector indexes.

        Returns:
            True if both loaded successfully
        """
        vector_loaded = self.vector_strategy.load_index()
        bm25_loaded = self.bm25_strategy.load_index()

        if vector_loaded and bm25_loaded:
            self._is_initialized = True
            logger.info("Hybrid index loaded successfully")
            return True
        elif vector_loaded:
            logger.warning("Only vector index loaded, BM25 index missing")
            self._is_initialized = True  # Partial functionality
            return True
        elif bm25_loaded:
            logger.warning("Only BM25 index loaded, vector index missing")
            self._is_initialized = True  # Partial functionality
            return True
        else:
            logger.error("Failed to load either index")
            return False

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """Retrieve documents using hybrid search with fusion.

        Args:
            query: Search query
            k: Number of documents to return (uses final_k if None)

        Returns:
            List of relevant documents
        """
        if not self._is_initialized and not self.load_index():
            raise ValueError("Indexes not available. Build indexes first.")

        k = k or self.final_k

        # Retrieve from both strategies
        vector_results = []
        bm25_results = []

        try:
            if self.vector_strategy.is_index_ready():
                vector_results = self.vector_strategy.retrieve(query, k=self.vector_k)
                logger.debug(f"Vector search returned {len(vector_results)} documents")
        except Exception as e:
            logger.warning(f"Vector retrieval failed: {e}")

        try:
            if self.bm25_strategy.is_index_ready():
                bm25_results = self.bm25_strategy.retrieve(query, k=self.bm25_k)
                logger.debug(f"BM25 search returned {len(bm25_results)} documents")
        except Exception as e:
            logger.warning(f"BM25 retrieval failed: {e}")

        # Handle fallback cases
        if not vector_results and not bm25_results:
            logger.warning("Both retrievers returned no results")
            return []

        if not vector_results:
            return bm25_results[:k]

        if not bm25_results:
            return vector_results[:k]

        # Apply fusion
        weights = [self.vector_weight, self.bm25_weight]
        fused_results = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            k=self.rrf_k,
            weights=weights,
        )

        logger.debug(
            f"Fusion combined {len(vector_results)} vector + {len(bm25_results)} BM25 "
            f"into {len(fused_results)} results, returning top {k}"
        )

        return fused_results[:k]

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Get LangChain-compatible fusion retriever.

        Args:
            **kwargs: Override search parameters

        Returns:
            FusionRetriever instance
        """
        if not self._is_initialized:
            self.load_index()

        # Get sub-retrievers
        retrievers = []
        weights = []

        if self.vector_strategy.is_index_ready():
            retrievers.append(self.vector_strategy.as_retriever(search_kwargs={"k": self.vector_k}))
            weights.append(self.vector_weight)

        if self.bm25_strategy.is_index_ready():
            retrievers.append(self.bm25_strategy.as_retriever(k=self.bm25_k))
            weights.append(self.bm25_weight)

        if not retrievers:
            raise ValueError("No retrievers available")

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        final_k = kwargs.get("k", self.final_k)

        return FusionRetriever(
            retrievers=retrievers,
            weights=weights,
            fusion_algorithm=self.fusion_algorithm,
            rrf_k=self.rrf_k,
            final_k=final_k,
        )

    def get_index_stats(self) -> dict[str, Any]:
        """Get combined index statistics.

        Returns:
            Dictionary with index statistics
        """
        stats = super().get_index_stats()
        stats.update(
            {
                "vector": self.vector_strategy.get_index_stats(),
                "bm25": self.bm25_strategy.get_index_stats(),
                "fusion": {
                    "algorithm": self.fusion_algorithm,
                    "rrf_k": self.rrf_k,
                    "weights": {
                        "vector": self.vector_weight,
                        "bm25": self.bm25_weight,
                    },
                },
                "final_k": self.final_k,
            }
        )
        return stats

    def delete_index(self) -> None:
        """Delete both indexes."""
        self.bm25_strategy.delete_index()
        # Vector store deletion is handled separately
        self._is_initialized = False
