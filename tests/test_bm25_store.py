"""Tests for BM25 store."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.retrieval.stores.bm25_store import BM25Store


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Python is a programming language known for its simplicity.",
            metadata={"source": "doc1.txt"},
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "doc2.txt"},
        ),
        Document(
            page_content="Deep learning uses neural networks with many layers.",
            metadata={"source": "doc3.txt"},
        ),
        Document(
            page_content="Natural language processing helps computers understand text.",
            metadata={"source": "doc4.txt"},
        ),
        Document(
            page_content="Python is widely used for machine learning and data science.",
            metadata={"source": "doc5.txt"},
        ),
    ]


@pytest.fixture
def temp_bm25_path(tmp_path):
    """Create temporary path for BM25 index."""
    return str(tmp_path / "bm25_test")


class TestBM25Store:
    """Tests for BM25Store class."""

    def test_init(self, temp_bm25_path):
        """Test BM25Store initialization."""
        store = BM25Store(persist_path=temp_bm25_path, tokenizer="simple")
        assert store.persist_path == Path(temp_bm25_path)
        assert store.tokenizer_type == "simple"
        assert not store.is_built()

    def test_simple_tokenize(self, temp_bm25_path):
        """Test simple tokenization."""
        store = BM25Store(persist_path=temp_bm25_path)
        tokens = store._simple_tokenize("Hello, World! This is a test.")
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_build_index(self, sample_documents, temp_bm25_path):
        """Test building BM25 index."""
        store = BM25Store(persist_path=temp_bm25_path)
        store.build_index(sample_documents)

        assert store.is_built()
        assert len(store.documents) == 5
        assert len(store.tokenized_corpus) == 5

    def test_build_index_empty_raises(self, temp_bm25_path):
        """Test that building with empty documents raises error."""
        store = BM25Store(persist_path=temp_bm25_path)
        with pytest.raises(ValueError, match="No documents provided"):
            store.build_index([])

    def test_search(self, sample_documents, temp_bm25_path):
        """Test BM25 search."""
        store = BM25Store(persist_path=temp_bm25_path)
        store.build_index(sample_documents)

        # Search for Python
        results = store.search("Python programming", k=3)
        assert len(results) <= 3
        # Should find the Python-related documents
        assert any("Python" in doc.page_content for doc in results)

    def test_search_with_scores(self, sample_documents, temp_bm25_path):
        """Test BM25 search with scores."""
        store = BM25Store(persist_path=temp_bm25_path)
        store.build_index(sample_documents)

        results = store.search_with_scores("machine learning", k=3)
        assert len(results) <= 3
        # Results should be tuples of (doc, score)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert score >= 0

    def test_save_and_load(self, sample_documents, temp_bm25_path):
        """Test saving and loading BM25 index."""
        # Build and save
        store1 = BM25Store(persist_path=temp_bm25_path)
        store1.build_index(sample_documents)
        store1.save()

        # Load in new instance
        store2 = BM25Store(persist_path=temp_bm25_path)
        assert store2.load()
        assert store2.is_built()
        assert len(store2.documents) == 5

        # Search should work on loaded store
        results = store2.search("Python", k=2)
        assert len(results) > 0

    def test_load_nonexistent_returns_false(self, temp_bm25_path):
        """Test that loading nonexistent index returns False."""
        store = BM25Store(persist_path=temp_bm25_path)
        assert not store.load()

    def test_delete(self, sample_documents, temp_bm25_path):
        """Test deleting BM25 index."""
        store = BM25Store(persist_path=temp_bm25_path)
        store.build_index(sample_documents)
        store.save()

        # Verify saved
        assert Path(temp_bm25_path).exists()

        # Delete
        store.delete()
        assert not Path(temp_bm25_path).exists()
        assert not store.is_built()

    def test_get_stats(self, sample_documents, temp_bm25_path):
        """Test getting index statistics."""
        store = BM25Store(persist_path=temp_bm25_path)
        store.build_index(sample_documents)

        stats = store.get_stats()
        assert stats["num_documents"] == 5
        assert stats["tokenizer"] == "simple"
        assert stats["is_built"]
        assert "doc_hash" in stats
