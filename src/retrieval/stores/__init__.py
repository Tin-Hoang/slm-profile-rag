"""Storage backends for retrieval indexes.

Available stores:
- BM25Store: BM25 index storage and retrieval
- (future) GraphStore: Knowledge graph storage
"""

from .bm25_store import BM25Store

__all__ = [
    "BM25Store",
]
