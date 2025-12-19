"""Retrieval module for extensible search strategies.

This module provides a pluggable architecture for different retrieval strategies:
- vector: Semantic similarity search using embeddings (ChromaDB)
- bm25: Lexical/keyword search using BM25 algorithm
- bm25_vector: Hybrid search combining BM25 and vector search with fusion
- page_index: (future) Vision-based document understanding (ColPali)
- graph_vector: (future) Knowledge graph enhanced retrieval
"""

from .base import BaseRetrieverStrategy
from .factory import RetrieverFactory
from .fusion import FusionRetriever, reciprocal_rank_fusion

__all__ = [
    "BaseRetrieverStrategy",
    "RetrieverFactory",
    "FusionRetriever",
    "reciprocal_rank_fusion",
]
