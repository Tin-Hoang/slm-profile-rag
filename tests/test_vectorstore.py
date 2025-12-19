"""Tests for vectorstore module."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.vectorstore import VectorStoreManager, get_vectorstore_manager


@pytest.fixture
def mock_config():
    """Create a mock config."""
    with patch("src.vectorstore.get_config") as mock:
        mock.return_value.get.side_effect = lambda key, default=None: {
            "embeddings.model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embeddings.device": "cpu",
            "vectorstore.collection_name": "test_collection",
            "retrieval.vector.search_type": "similarity",
            "retrieval.vector.search_kwargs": {},
            "retrieval.vector.k": 4,
        }.get(key, default)
        mock.return_value.get_env.side_effect = lambda key, default=None: {
            "CHROMA_PERSIST_DIR": tempfile.mkdtemp(),
        }.get(key, default)
        yield mock


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings."""
    with patch("src.vectorstore.HuggingFaceEmbeddings") as mock:
        mock_embedding = MagicMock()
        mock_embedding.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_embedding.embed_query.return_value = [0.1, 0.2, 0.3]
        mock.return_value = mock_embedding
        yield mock


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="This is a test document about AI.",
            metadata={"source": "test1.txt"},
        ),
        Document(
            page_content="Machine learning is a subset of AI.",
            metadata={"source": "test2.txt"},
        ),
        Document(
            page_content="Deep learning uses neural networks.",
            metadata={"source": "test3.txt"},
        ),
    ]


# --- VectorStoreManager Tests ---


def test_vectorstore_manager_init(mock_config, mock_embeddings):
    """Test VectorStoreManager initialization."""
    manager = VectorStoreManager()

    assert manager is not None
    assert manager.embeddings is not None
    assert manager.vectorstore is None


def test_create_vectorstore_empty_documents_raises(mock_config, mock_embeddings):
    """Test create_vectorstore raises for empty documents."""
    manager = VectorStoreManager()

    with pytest.raises(ValueError, match="No documents provided"):
        manager.create_vectorstore([])


def test_create_vectorstore_success(mock_config, mock_embeddings, sample_documents):
    """Test create_vectorstore creates store successfully."""
    manager = VectorStoreManager()

    with patch("src.vectorstore.Chroma") as mock_chroma:
        mock_chroma.from_documents.return_value = MagicMock()

        result = manager.create_vectorstore(sample_documents)

        assert result is not None
        mock_chroma.from_documents.assert_called_once()


def test_create_vectorstore_error(mock_config, mock_embeddings, sample_documents):
    """Test create_vectorstore handles errors."""
    manager = VectorStoreManager()

    with patch("src.vectorstore.Chroma") as mock_chroma:
        mock_chroma.from_documents.side_effect = Exception("Chroma error")

        with pytest.raises(Exception, match="Chroma error"):
            manager.create_vectorstore(sample_documents)


def test_load_vectorstore_not_found(mock_config, mock_embeddings):
    """Test load_vectorstore raises for missing store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use temp directory but point to non-existent subdirectory for load
        nonexistent_subdir = f"{tmpdir}/nonexistent_store"

        with patch("src.vectorstore.get_config") as mock:
            mock.return_value.get.side_effect = lambda key, default=None: {
                "embeddings.model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "embeddings.device": "cpu",
                "vectorstore.collection_name": "test_collection",
            }.get(key, default)
            # Use valid temp path for init, but set persist_directory after
            mock.return_value.get_env.return_value = tmpdir

            manager = VectorStoreManager()
            # Override with non-existent path for the test
            manager.persist_directory = nonexistent_subdir

            with pytest.raises(FileNotFoundError, match="Vector store not found"):
                manager.load_vectorstore()


def test_load_vectorstore_success(mock_config, mock_embeddings):
    """Test load_vectorstore loads successfully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.vectorstore.get_config") as mock:
            mock.return_value.get.side_effect = lambda key, default=None: {
                "embeddings.model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "embeddings.device": "cpu",
                "vectorstore.collection_name": "test_collection",
            }.get(key, default)
            mock.return_value.get_env.return_value = tmpdir

            manager = VectorStoreManager()

            with patch("src.vectorstore.Chroma") as mock_chroma:
                mock_store = MagicMock()
                mock_store._collection.count.return_value = 10
                mock_chroma.return_value = mock_store

                result = manager.load_vectorstore()

                assert result is not None
                assert manager.vectorstore is not None


def test_load_vectorstore_error(mock_config, mock_embeddings):
    """Test load_vectorstore handles errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.vectorstore.get_config") as mock:
            mock.return_value.get.side_effect = lambda key, default=None: {
                "embeddings.model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "embeddings.device": "cpu",
                "vectorstore.collection_name": "test_collection",
            }.get(key, default)
            mock.return_value.get_env.return_value = tmpdir

            manager = VectorStoreManager()

            with patch("src.vectorstore.Chroma") as mock_chroma:
                mock_chroma.side_effect = Exception("Load error")

                with pytest.raises(Exception, match="Load error"):
                    manager.load_vectorstore()


def test_get_retriever_loads_if_not_loaded(mock_config, mock_embeddings):
    """Test get_retriever loads vectorstore if not loaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.vectorstore.get_config") as mock:
            mock.return_value.get.side_effect = lambda key, default=None: {
                "embeddings.model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "embeddings.device": "cpu",
                "vectorstore.collection_name": "test_collection",
                "retrieval.vector.search_type": "similarity",
                "retrieval.vector.search_kwargs": {},
                "retrieval.vector.k": 4,
            }.get(key, default)
            mock.return_value.get_env.return_value = tmpdir

            manager = VectorStoreManager()

            with patch("src.vectorstore.Chroma") as mock_chroma:
                mock_store = MagicMock()
                mock_store._collection.count.return_value = 10
                mock_chroma.return_value = mock_store

                retriever = manager.get_retriever()

                assert retriever is not None


def test_get_retriever_similarity_search(mock_config, mock_embeddings):
    """Test get_retriever with similarity search."""
    manager = VectorStoreManager()
    mock_store = MagicMock()
    manager.vectorstore = mock_store

    manager.get_retriever()

    mock_store.as_retriever.assert_called_once()


def test_get_retriever_mmr_search(mock_embeddings):
    """Test get_retriever with MMR search."""
    with patch("src.vectorstore.get_config") as mock:
        mock.return_value.get.side_effect = lambda key, default=None: {
            "embeddings.model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embeddings.device": "cpu",
            "vectorstore.collection_name": "test_collection",
            "retrieval.vector.search_type": "mmr",
            "retrieval.vector.search_kwargs": {"fetch_k": 20, "lambda_mult": 0.5},
            "retrieval.vector.k": 4,
        }.get(key, default)
        mock.return_value.get_env.return_value = tempfile.mkdtemp()

        manager = VectorStoreManager()
        mock_store = MagicMock()
        manager.vectorstore = mock_store

        manager.get_retriever()

        call_args = mock_store.as_retriever.call_args
        assert call_args is not None
        assert call_args.kwargs.get("search_type") == "mmr"


def test_add_documents_loads_if_not_loaded(mock_config, mock_embeddings, sample_documents):
    """Test add_documents loads vectorstore if not loaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.vectorstore.get_config") as mock:
            mock.return_value.get.side_effect = lambda key, default=None: {
                "embeddings.model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "embeddings.device": "cpu",
                "vectorstore.collection_name": "test_collection",
            }.get(key, default)
            mock.return_value.get_env.return_value = tmpdir

            manager = VectorStoreManager()

            with patch("src.vectorstore.Chroma") as mock_chroma:
                mock_store = MagicMock()
                mock_store._collection.count.return_value = 10
                mock_chroma.return_value = mock_store

                manager.add_documents(sample_documents)

                mock_store.add_documents.assert_called_once_with(sample_documents)


def test_add_documents_error(mock_config, mock_embeddings, sample_documents):
    """Test add_documents handles errors."""
    manager = VectorStoreManager()
    mock_store = MagicMock()
    mock_store.add_documents.side_effect = Exception("Add error")
    manager.vectorstore = mock_store

    with pytest.raises(Exception, match="Add error"):
        manager.add_documents(sample_documents)


def test_delete_collection(mock_config, mock_embeddings):
    """Test delete_collection deletes the collection."""
    manager = VectorStoreManager()
    mock_store = MagicMock()
    manager.vectorstore = mock_store

    with patch("src.vectorstore.chromadb") as mock_chromadb:
        mock_client = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        manager.delete_collection()

        mock_client.delete_collection.assert_called_once()
        assert manager.vectorstore is None


def test_delete_collection_error(mock_config, mock_embeddings):
    """Test delete_collection handles errors."""
    manager = VectorStoreManager()

    with patch("src.vectorstore.chromadb") as mock_chromadb:
        mock_chromadb.PersistentClient.side_effect = Exception("Delete error")

        with pytest.raises(Exception, match="Delete error"):
            manager.delete_collection()


def test_similarity_search_loads_if_not_loaded(mock_config, mock_embeddings):
    """Test similarity_search loads vectorstore if not loaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.vectorstore.get_config") as mock:
            mock.return_value.get.side_effect = lambda key, default=None: {
                "embeddings.model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "embeddings.device": "cpu",
                "vectorstore.collection_name": "test_collection",
            }.get(key, default)
            mock.return_value.get_env.return_value = tmpdir

            manager = VectorStoreManager()

            with patch("src.vectorstore.Chroma") as mock_chroma:
                mock_store = MagicMock()
                mock_store._collection.count.return_value = 10
                mock_store.similarity_search.return_value = []
                mock_chroma.return_value = mock_store

                results = manager.similarity_search("test query")

                assert results == []


def test_similarity_search(mock_config, mock_embeddings, sample_documents):
    """Test similarity_search performs search."""
    manager = VectorStoreManager()
    mock_store = MagicMock()
    mock_store.similarity_search.return_value = sample_documents[:2]
    manager.vectorstore = mock_store

    results = manager.similarity_search("AI and machine learning", k=2)

    assert len(results) == 2
    mock_store.similarity_search.assert_called_once_with("AI and machine learning", k=2)


# --- Factory Function Tests ---


def test_get_vectorstore_manager():
    """Test get_vectorstore_manager returns instance."""
    with patch("src.vectorstore.get_config") as mock:
        mock.return_value.get.side_effect = lambda key, default=None: {
            "embeddings.model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embeddings.device": "cpu",
            "vectorstore.collection_name": "test_collection",
        }.get(key, default)
        mock.return_value.get_env.return_value = tempfile.mkdtemp()

        with patch("src.vectorstore.HuggingFaceEmbeddings"):
            manager = get_vectorstore_manager()

            assert isinstance(manager, VectorStoreManager)
