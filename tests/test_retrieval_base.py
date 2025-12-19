"""Tests for retrieval base classes and factory."""

import pytest

from src.retrieval import RetrieverFactory
from src.retrieval.base import BaseRetrieverStrategy


class TestRetrieverFactory:
    """Tests for RetrieverFactory."""

    def test_available_strategies(self):
        """Test that default strategies are registered."""
        strategies = RetrieverFactory.available_strategies()
        assert "vector" in strategies
        assert "bm25" in strategies
        assert "bm25_vector" in strategies

    def test_is_registered(self):
        """Test is_registered method."""
        assert RetrieverFactory.is_registered("vector")
        assert RetrieverFactory.is_registered("bm25")
        assert RetrieverFactory.is_registered("bm25_vector")
        assert not RetrieverFactory.is_registered("nonexistent")

    def test_create_unknown_strategy_raises(self):
        """Test that creating unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown retrieval strategy"):
            RetrieverFactory.create("nonexistent_strategy", {})

    def test_create_vector_strategy(self):
        """Test creating vector strategy."""
        config = {"retrieval": {"vector": {"search_type": "similarity", "k": 4}}}
        strategy = RetrieverFactory.create("vector", config)
        assert strategy.name == "vector"
        assert isinstance(strategy, BaseRetrieverStrategy)

    def test_create_bm25_strategy(self):
        """Test creating BM25 strategy."""
        config = {
            "retrieval": {
                "bm25": {
                    "k": 10,
                    "persist_path": "./test_bm25_index",
                    "tokenizer": "simple",
                }
            }
        }
        strategy = RetrieverFactory.create("bm25", config)
        assert strategy.name == "bm25"
        assert isinstance(strategy, BaseRetrieverStrategy)

    def test_create_bm25_vector_strategy(self):
        """Test creating BM25+Vector strategy."""
        config = {
            "retrieval": {
                "final_k": 4,
                "vector": {"search_type": "similarity", "k": 10},
                "bm25": {"k": 10, "persist_path": "./test_bm25_index"},
                "fusion": {
                    "algorithm": "rrf",
                    "rrf_k": 60,
                    "weights": {"vector": 0.7, "bm25": 0.3},
                },
            }
        }
        strategy = RetrieverFactory.create("bm25_vector", config)
        assert strategy.name == "bm25_vector"
        assert isinstance(strategy, BaseRetrieverStrategy)

    def test_register_custom_strategy(self):
        """Test registering a custom strategy."""

        @RetrieverFactory.register("test_custom")
        class CustomStrategy(BaseRetrieverStrategy):
            @property
            def name(self):
                return "test_custom"

            def build_index(self, documents):
                pass

            def load_index(self):
                return True

            def retrieve(self, query, k=4):  # noqa: ARG002
                return []

            def as_retriever(self, **kwargs):  # noqa: ARG002
                return None

        assert RetrieverFactory.is_registered("test_custom")
        strategy = RetrieverFactory.create("test_custom", {})
        assert strategy.name == "test_custom"
