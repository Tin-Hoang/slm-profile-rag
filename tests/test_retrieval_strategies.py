"""Tests for retrieval strategies."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.retrieval.strategies.bm25 import BM25Strategy
from src.retrieval.strategies.bm25_vector import BM25VectorStrategy
from src.retrieval.strategies.vector import VectorStrategy


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


@pytest.fixture
def mock_vectorstore_manager():
    """Create a mock VectorStoreManager."""
    # VectorStoreManager is imported inside VectorStrategy using lazy import
    # from src.vectorstore, so we patch it there
    with patch("src.vectorstore.VectorStoreManager") as mock:
        mock_manager = MagicMock()
        mock_manager.persist_directory = tempfile.mkdtemp()
        mock.return_value = mock_manager
        yield mock_manager


# --- BM25Strategy Tests ---


def test_bm25_init(temp_index_path):
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


def test_bm25_build_and_retrieve(sample_documents, temp_index_path):
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


def test_bm25_load_index(sample_documents, temp_index_path):
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


def test_bm25_as_retriever(sample_documents, temp_index_path):
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


def test_bm25_get_index_stats(sample_documents, temp_index_path):
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


# --- BM25VectorStrategy Tests ---


def test_bm25_vector_init(temp_index_path):
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
def test_bm25_vector_build_index_calls_both(
    mock_vector_load, mock_vector_build, sample_documents, temp_index_path
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


def test_bm25_vector_get_index_stats(temp_index_path):
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


@patch("src.retrieval.strategies.vector.VectorStrategy.load_index")
@patch("src.retrieval.strategies.vector.VectorStrategy.is_index_ready")
@patch("src.retrieval.strategies.vector.VectorStrategy.retrieve")
def test_bm25_vector_retrieve_with_fusion(
    mock_vector_retrieve,
    mock_vector_ready,
    mock_vector_load,
    sample_documents,
    temp_index_path,
):
    """Test retrieve combines vector and BM25 results."""
    mock_vector_load.return_value = True
    mock_vector_ready.return_value = True
    mock_vector_retrieve.return_value = sample_documents[:2]

    config = {
        "retrieval": {
            "final_k": 3,
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
    strategy.bm25_strategy.build_index(sample_documents)
    strategy._is_initialized = True

    results = strategy.retrieve("machine learning AI", k=3)

    assert len(results) <= 3
    assert all(isinstance(doc, Document) for doc in results)


@patch("src.retrieval.strategies.vector.VectorStrategy.load_index")
@patch("src.retrieval.strategies.vector.VectorStrategy.is_index_ready")
@patch("src.retrieval.strategies.vector.VectorStrategy.retrieve")
def test_bm25_vector_retrieve_vector_only_fallback(
    mock_vector_retrieve,
    mock_vector_ready,
    mock_vector_load,
    sample_documents,
    temp_index_path,
):
    """Test retrieve falls back to vector only when BM25 fails."""
    mock_vector_load.return_value = True
    mock_vector_ready.return_value = True
    mock_vector_retrieve.return_value = sample_documents[:2]

    config = {
        "retrieval": {
            "final_k": 3,
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
    strategy._is_initialized = True
    # Don't build BM25 index - it should return empty

    results = strategy.retrieve("query", k=2)

    # Should return vector results as fallback
    assert len(results) == 2


@patch("src.retrieval.strategies.vector.VectorStrategy.load_index")
@patch("src.retrieval.strategies.vector.VectorStrategy.is_index_ready")
def test_bm25_vector_retrieve_bm25_only_fallback(
    mock_vector_ready, mock_vector_load, sample_documents, temp_index_path
):
    """Test retrieve falls back to BM25 only when vector fails."""
    mock_vector_load.return_value = True
    mock_vector_ready.return_value = False  # Vector not ready

    config = {
        "retrieval": {
            "final_k": 3,
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
    strategy.bm25_strategy.build_index(sample_documents)
    strategy._is_initialized = True

    results = strategy.retrieve("machine learning", k=2)

    # Should return BM25 results as fallback
    assert len(results) <= 2


@patch("src.retrieval.strategies.vector.VectorStrategy.load_index")
@patch("src.retrieval.strategies.vector.VectorStrategy.is_index_ready")
def test_bm25_vector_retrieve_both_empty(mock_vector_ready, mock_vector_load, temp_index_path):
    """Test retrieve returns empty when both fail."""
    mock_vector_load.return_value = True
    mock_vector_ready.return_value = False

    config = {
        "retrieval": {
            "final_k": 3,
            "vector": {"search_type": "similarity", "k": 10},
            "bm25": {"k": 10, "persist_path": temp_index_path["bm25"]},
            "fusion": {"algorithm": "rrf", "rrf_k": 60},
        }
    }
    strategy = BM25VectorStrategy(config)
    strategy._is_initialized = True
    # Neither index is ready

    results = strategy.retrieve("query", k=2)

    assert results == []


def test_bm25_vector_retrieve_not_initialized_raises(temp_index_path):
    """Test retrieve raises when not initialized and can't load."""
    config = {
        "retrieval": {
            "final_k": 3,
            "vector": {"search_type": "similarity", "k": 10},
            "bm25": {"k": 10, "persist_path": temp_index_path["bm25"]},
            "fusion": {"algorithm": "rrf", "rrf_k": 60},
        }
    }
    strategy = BM25VectorStrategy(config)

    # Mock load_index to return False
    with patch.object(strategy, "load_index", return_value=False):
        with pytest.raises(ValueError, match="Indexes not available"):
            strategy.retrieve("query")


@patch("src.retrieval.strategies.vector.VectorStrategy.load_index")
def test_bm25_vector_load_index_both_success(mock_vector_load, sample_documents, temp_index_path):
    """Test load_index succeeds when both indexes load."""
    mock_vector_load.return_value = True

    config = {
        "retrieval": {
            "final_k": 4,
            "vector": {"search_type": "similarity", "k": 10},
            "bm25": {"k": 10, "persist_path": temp_index_path["bm25"]},
            "fusion": {"algorithm": "rrf", "rrf_k": 60},
        }
    }
    strategy = BM25VectorStrategy(config)

    # Build BM25 first so it can be loaded
    strategy.bm25_strategy.build_index(sample_documents)

    # Create new strategy instance
    strategy2 = BM25VectorStrategy(config)
    result = strategy2.load_index()

    assert result is True
    assert strategy2._is_initialized


@patch("src.retrieval.strategies.vector.VectorStrategy.load_index")
def test_bm25_vector_load_index_vector_only(mock_vector_load, temp_index_path):
    """Test load_index with only vector succeeding."""
    mock_vector_load.return_value = True

    config = {
        "retrieval": {
            "final_k": 4,
            "vector": {"search_type": "similarity", "k": 10},
            "bm25": {"k": 10, "persist_path": temp_index_path["bm25"]},
            "fusion": {"algorithm": "rrf", "rrf_k": 60},
        }
    }
    strategy = BM25VectorStrategy(config)

    result = strategy.load_index()

    # Should succeed with partial functionality
    assert result is True
    assert strategy._is_initialized


@patch("src.retrieval.strategies.vector.VectorStrategy.load_index")
def test_bm25_vector_load_index_bm25_only(mock_vector_load, sample_documents, temp_index_path):
    """Test load_index with only BM25 succeeding."""
    mock_vector_load.return_value = False

    config = {
        "retrieval": {
            "final_k": 4,
            "vector": {"search_type": "similarity", "k": 10},
            "bm25": {"k": 10, "persist_path": temp_index_path["bm25"]},
            "fusion": {"algorithm": "rrf", "rrf_k": 60},
        }
    }
    strategy = BM25VectorStrategy(config)
    strategy.bm25_strategy.build_index(sample_documents)

    strategy2 = BM25VectorStrategy(config)
    result = strategy2.load_index()

    assert result is True
    assert strategy2._is_initialized


@patch("src.retrieval.strategies.vector.VectorStrategy.load_index")
def test_bm25_vector_load_index_both_fail(mock_vector_load, temp_index_path):
    """Test load_index fails when both fail."""
    mock_vector_load.return_value = False

    config = {
        "retrieval": {
            "final_k": 4,
            "vector": {"search_type": "similarity", "k": 10},
            "bm25": {"k": 10, "persist_path": temp_index_path["bm25"]},
            "fusion": {"algorithm": "rrf", "rrf_k": 60},
        }
    }
    strategy = BM25VectorStrategy(config)

    result = strategy.load_index()

    assert result is False


@patch("src.retrieval.strategies.vector.VectorStrategy.load_index")
@patch("src.retrieval.strategies.vector.VectorStrategy.is_index_ready")
@patch("src.retrieval.strategies.vector.VectorStrategy.as_retriever")
def test_bm25_vector_as_retriever(
    mock_vector_retriever,
    mock_vector_ready,
    mock_vector_load,
    sample_documents,
    temp_index_path,
):
    """Test as_retriever creates FusionRetriever."""
    from langchain_core.retrievers import BaseRetriever

    mock_vector_load.return_value = True
    mock_vector_ready.return_value = True
    # Use spec to satisfy Pydantic validation
    mock_retriever = MagicMock(spec=BaseRetriever)
    mock_vector_retriever.return_value = mock_retriever

    config = {
        "retrieval": {
            "final_k": 4,
            "vector": {"search_type": "similarity", "k": 10},
            "bm25": {"k": 10, "persist_path": temp_index_path["bm25"]},
            "fusion": {"algorithm": "rrf", "rrf_k": 60},
        }
    }
    strategy = BM25VectorStrategy(config)
    strategy.bm25_strategy.build_index(sample_documents)
    strategy._is_initialized = True

    retriever = strategy.as_retriever()

    assert retriever is not None


@patch("src.retrieval.strategies.vector.VectorStrategy.load_index")
@patch("src.retrieval.strategies.vector.VectorStrategy.is_index_ready")
def test_bm25_vector_as_retriever_no_retrievers_raises(
    mock_vector_ready, mock_vector_load, temp_index_path
):
    """Test as_retriever raises when no retrievers available."""
    mock_vector_load.return_value = True
    mock_vector_ready.return_value = False

    config = {
        "retrieval": {
            "final_k": 4,
            "vector": {"search_type": "similarity", "k": 10},
            "bm25": {"k": 10, "persist_path": temp_index_path["bm25"]},
            "fusion": {"algorithm": "rrf", "rrf_k": 60},
        }
    }
    strategy = BM25VectorStrategy(config)
    strategy._is_initialized = True

    with pytest.raises(ValueError, match="No retrievers available"):
        strategy.as_retriever()


@patch("src.retrieval.strategies.vector.VectorStrategy.load_index")
def test_bm25_vector_delete_index(mock_vector_load, sample_documents, temp_index_path):
    """Test delete_index clears indexes."""
    mock_vector_load.return_value = True

    config = {
        "retrieval": {
            "final_k": 4,
            "vector": {"search_type": "similarity", "k": 10},
            "bm25": {"k": 10, "persist_path": temp_index_path["bm25"]},
            "fusion": {"algorithm": "rrf", "rrf_k": 60},
        }
    }
    strategy = BM25VectorStrategy(config)
    strategy.bm25_strategy.build_index(sample_documents)
    strategy._is_initialized = True

    strategy.delete_index()

    assert not strategy._is_initialized


# --- VectorStrategy Tests ---


def test_vector_init(mock_vectorstore_manager):
    """Test VectorStrategy initialization."""
    config = {
        "retrieval": {
            "vector": {
                "search_type": "similarity",
                "k": 4,
                "search_kwargs": {},
            }
        }
    }
    strategy = VectorStrategy(config)

    assert strategy.name == "vector"
    assert strategy.k == 4
    assert strategy.search_type == "similarity"


def test_vector_build_index(mock_vectorstore_manager, sample_documents):
    """Test building vector index."""
    config = {"retrieval": {"vector": {"search_type": "similarity", "k": 4}}}
    strategy = VectorStrategy(config)

    strategy.build_index(sample_documents)

    mock_vectorstore_manager.create_vectorstore.assert_called_once_with(sample_documents)
    assert strategy._is_initialized


def test_vector_load_index_success(mock_vectorstore_manager):
    """Test loading vector index successfully."""
    config = {"retrieval": {"vector": {"search_type": "similarity", "k": 4}}}

    with patch("src.retrieval.strategies.vector.Path") as mock_path:
        mock_path.return_value.exists.return_value = True

        strategy = VectorStrategy(config)
        result = strategy.load_index()

        assert result is True
        assert strategy._is_initialized
        mock_vectorstore_manager.load_vectorstore.assert_called_once()


def test_vector_load_index_not_found(mock_vectorstore_manager):
    """Test loading when index not found."""
    config = {"retrieval": {"vector": {"search_type": "similarity", "k": 4}}}

    with patch("src.retrieval.strategies.vector.Path") as mock_path:
        mock_path.return_value.exists.return_value = False

        strategy = VectorStrategy(config)
        result = strategy.load_index()

        assert result is False


def test_vector_load_index_error(mock_vectorstore_manager):
    """Test loading with error."""
    config = {"retrieval": {"vector": {"search_type": "similarity", "k": 4}}}
    mock_vectorstore_manager.load_vectorstore.side_effect = Exception("Load error")

    with patch("src.retrieval.strategies.vector.Path") as mock_path:
        mock_path.return_value.exists.return_value = True

        strategy = VectorStrategy(config)
        result = strategy.load_index()

        assert result is False


def test_vector_retrieve(mock_vectorstore_manager, sample_documents):
    """Test retrieving documents."""
    config = {"retrieval": {"vector": {"search_type": "similarity", "k": 4}}}
    mock_vectorstore_manager.similarity_search.return_value = sample_documents[:2]

    strategy = VectorStrategy(config)
    strategy._is_initialized = True

    results = strategy.retrieve("machine learning", k=2)

    assert len(results) == 2
    mock_vectorstore_manager.similarity_search.assert_called_once_with("machine learning", k=2)


def test_vector_retrieve_loads_if_not_initialized(mock_vectorstore_manager, sample_documents):
    """Test retrieve loads index if not initialized."""
    config = {"retrieval": {"vector": {"search_type": "similarity", "k": 4}}}
    mock_vectorstore_manager.similarity_search.return_value = sample_documents[:2]

    with patch("src.retrieval.strategies.vector.Path") as mock_path:
        mock_path.return_value.exists.return_value = True

        strategy = VectorStrategy(config)
        strategy._is_initialized = False

        strategy.retrieve("query", k=2)

        mock_vectorstore_manager.load_vectorstore.assert_called_once()


def test_vector_as_retriever(mock_vectorstore_manager):
    """Test getting LangChain retriever."""
    config = {"retrieval": {"vector": {"search_type": "similarity", "k": 4}}}
    mock_retriever = MagicMock()
    mock_vectorstore_manager.get_retriever.return_value = mock_retriever

    strategy = VectorStrategy(config)
    strategy._is_initialized = True

    retriever = strategy.as_retriever()

    assert retriever is mock_retriever


def test_vector_as_retriever_with_kwargs(mock_vectorstore_manager):
    """Test getting retriever with custom kwargs."""
    config = {"retrieval": {"vector": {"search_type": "similarity", "k": 4}}}
    mock_retriever = MagicMock()
    mock_vectorstore_manager.get_retriever.return_value = mock_retriever

    strategy = VectorStrategy(config)
    strategy._is_initialized = True

    strategy.as_retriever(search_type="mmr", search_kwargs={"k": 10})

    mock_vectorstore_manager.get_retriever.assert_called_once()
    call_kwargs = mock_vectorstore_manager.get_retriever.call_args.kwargs
    assert call_kwargs.get("search_type") == "mmr"


def test_vector_get_index_stats_initialized(mock_vectorstore_manager):
    """Test getting index stats when initialized."""
    config = {"retrieval": {"vector": {"search_type": "similarity", "k": 4}}}
    mock_store = MagicMock()
    mock_store._collection.count.return_value = 100
    mock_vectorstore_manager.vectorstore = mock_store
    mock_vectorstore_manager.persist_directory = "/test/path"
    mock_vectorstore_manager.collection_name = "test_collection"

    strategy = VectorStrategy(config)
    strategy._is_initialized = True

    stats = strategy.get_index_stats()

    assert stats["strategy"] == "vector"
    assert stats["num_documents"] == 100
    assert stats["collection_name"] == "test_collection"


def test_vector_get_index_stats_error(mock_vectorstore_manager):
    """Test getting index stats handles errors."""
    config = {"retrieval": {"vector": {"search_type": "similarity", "k": 4}}}
    mock_store = MagicMock()
    mock_store._collection.count.side_effect = Exception("Stats error")
    mock_vectorstore_manager.vectorstore = mock_store

    strategy = VectorStrategy(config)
    strategy._is_initialized = True

    stats = strategy.get_index_stats()

    # Should still return base stats
    assert stats["strategy"] == "vector"
