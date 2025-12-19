"""Tests for retrieval strategies."""

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from src.retrieval.strategies.bm25 import BM25Strategy
from src.retrieval.strategies.bm25_vector import BM25VectorStrategy


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Tin Hoang is an AI Research Engineer with expertise in machine learning.",
            metadata={"source": "profile.md"},
        ),
        Document(
            page_content="He has experience with Python, PyTorch, and TensorFlow frameworks.",
            metadata={"source": "skills.md"},
        ),
        Document(
            page_content="His research focuses on federated learning for medical AI applications.",
            metadata={"source": "research.md"},
        ),
        Document(
            page_content="He developed OCR systems using deep learning techniques.",
            metadata={"source": "projects.md"},
        ),
    ]


@pytest.fixture
def temp_index_path(tmp_path):
    """Create temporary paths for indexes."""
    return {
        "bm25": str(tmp_path / "bm25_test"),
        "vector": str(tmp_path / "vector_test"),
    }


class TestBM25Strategy:
    """Tests for BM25Strategy."""

    def test_init(self, temp_index_path):
        """Test BM25Strategy initialization."""
        config = {
            "retrieval": {
                "bm25": {
                    "k": 5,
                    "persist_path": temp_index_path["bm25"],
                    "tokenizer": "simple",
                }
            }
        }
        strategy = BM25Strategy(config)

        assert strategy.name == "bm25"
        assert strategy.k == 5
        assert not strategy.is_index_ready()

    def test_build_and_retrieve(self, sample_documents, temp_index_path):
        """Test building index and retrieving documents."""
        config = {
            "retrieval": {
                "bm25": {
                    "k": 3,
                    "persist_path": temp_index_path["bm25"],
                    "tokenizer": "simple",
                }
            }
        }
        strategy = BM25Strategy(config)

        # Build index
        strategy.build_index(sample_documents)
        assert strategy.is_index_ready()

        # Retrieve
        results = strategy.retrieve("machine learning AI", k=2)
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)

    def test_load_index(self, sample_documents, temp_index_path):
        """Test loading a saved index."""
        config = {
            "retrieval": {
                "bm25": {
                    "k": 3,
                    "persist_path": temp_index_path["bm25"],
                }
            }
        }

        # Build and save
        strategy1 = BM25Strategy(config)
        strategy1.build_index(sample_documents)

        # Load in new instance
        strategy2 = BM25Strategy(config)
        assert strategy2.load_index()
        assert strategy2.is_index_ready()

        # Should be able to retrieve
        results = strategy2.retrieve("Python", k=2)
        assert len(results) > 0

    def test_as_retriever(self, sample_documents, temp_index_path):
        """Test getting LangChain retriever."""
        config = {
            "retrieval": {
                "bm25": {
                    "k": 2,
                    "persist_path": temp_index_path["bm25"],
                }
            }
        }
        strategy = BM25Strategy(config)
        strategy.build_index(sample_documents)

        retriever = strategy.as_retriever()
        assert retriever is not None

        # Should work with invoke
        results = retriever.invoke("federated learning")
        assert isinstance(results, list)

    def test_get_index_stats(self, sample_documents, temp_index_path):
        """Test getting index statistics."""
        config = {
            "retrieval": {
                "bm25": {
                    "k": 3,
                    "persist_path": temp_index_path["bm25"],
                }
            }
        }
        strategy = BM25Strategy(config)
        strategy.build_index(sample_documents)

        stats = strategy.get_index_stats()
        assert stats["strategy"] == "bm25"
        assert stats["num_documents"] == 4
        assert stats["is_built"]


class TestBM25VectorStrategy:
    """Tests for BM25VectorStrategy."""

    def test_init(self, temp_index_path):
        """Test BM25VectorStrategy initialization."""
        config = {
            "retrieval": {
                "final_k": 4,
                "vector": {"search_type": "similarity", "k": 10},
                "bm25": {"k": 10, "persist_path": temp_index_path["bm25"]},
                "fusion": {
                    "algorithm": "rrf",
                    "rrf_k": 60,
                    "weights": {"vector": 0.7, "bm25": 0.3},
                },
            }
        }
        strategy = BM25VectorStrategy(config)

        assert strategy.name == "bm25_vector"
        assert strategy.vector_weight == 0.7
        assert strategy.bm25_weight == 0.3
        assert strategy.final_k == 4

    @patch("src.retrieval.strategies.vector.VectorStrategy.build_index")
    @patch("src.retrieval.strategies.vector.VectorStrategy.load_index")
    def test_build_index_calls_both(
        self, mock_vector_load, mock_vector_build, sample_documents, temp_index_path
    ):
        """Test that build_index builds both BM25 and Vector indexes."""
        mock_vector_load.return_value = True

        config = {
            "retrieval": {
                "final_k": 4,
                "vector": {"search_type": "similarity", "k": 10},
                "bm25": {"k": 10, "persist_path": temp_index_path["bm25"]},
                "fusion": {
                    "algorithm": "rrf",
                    "rrf_k": 60,
                    "weights": {"vector": 0.7, "bm25": 0.3},
                },
            }
        }
        strategy = BM25VectorStrategy(config)
        strategy.build_index(sample_documents)

        # Vector build should be called
        mock_vector_build.assert_called_once_with(sample_documents)

        # BM25 should be built (we can check the underlying store)
        assert strategy.bm25_strategy.is_index_ready()

    def test_get_index_stats(self, temp_index_path):
        """Test getting combined index statistics."""
        config = {
            "retrieval": {
                "final_k": 4,
                "vector": {"search_type": "similarity", "k": 10},
                "bm25": {"k": 10, "persist_path": temp_index_path["bm25"]},
                "fusion": {
                    "algorithm": "rrf",
                    "rrf_k": 60,
                    "weights": {"vector": 0.7, "bm25": 0.3},
                },
            }
        }
        strategy = BM25VectorStrategy(config)

        stats = strategy.get_index_stats()
        assert stats["strategy"] == "bm25_vector"
        assert "vector" in stats
        assert "bm25" in stats
        assert "fusion" in stats
        assert stats["fusion"]["algorithm"] == "rrf"
        assert stats["fusion"]["weights"]["vector"] == 0.7
