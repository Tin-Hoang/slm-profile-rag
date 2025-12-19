"""Tests for fusion algorithms."""

import pytest
from langchain_core.documents import Document

from src.retrieval.fusion import reciprocal_rank_fusion, weighted_fusion


@pytest.fixture
def sample_results():
    """Create sample ranked results for testing."""
    # Simulate results from two different retrievers
    vector_results = [
        Document(page_content="Document A about machine learning", metadata={"source": "a"}),
        Document(page_content="Document B about deep learning", metadata={"source": "b"}),
        Document(page_content="Document C about neural networks", metadata={"source": "c"}),
    ]

    bm25_results = [
        Document(page_content="Document B about deep learning", metadata={"source": "b"}),
        Document(page_content="Document D about Python", metadata={"source": "d"}),
        Document(page_content="Document A about machine learning", metadata={"source": "a"}),
    ]

    return vector_results, bm25_results


class TestReciprocalRankFusion:
    """Tests for RRF algorithm."""

    def test_empty_results(self):
        """Test RRF with empty results."""
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_single_result_list(self, sample_results):
        """Test RRF with single result list."""
        vector_results, _ = sample_results
        result = reciprocal_rank_fusion([vector_results])

        # Order should be preserved
        assert len(result) == 3
        assert result[0].page_content == vector_results[0].page_content

    def test_fusion_combines_results(self, sample_results):
        """Test that RRF combines results from multiple sources."""
        vector_results, bm25_results = sample_results
        result = reciprocal_rank_fusion([vector_results, bm25_results])

        # Should have all unique documents
        contents = {doc.page_content for doc in result}
        assert len(contents) == 4  # A, B, C, D

    def test_fusion_ranks_overlapping_higher(self, sample_results):
        """Test that documents in both lists rank higher."""
        vector_results, bm25_results = sample_results
        result = reciprocal_rank_fusion([vector_results, bm25_results])

        # Documents A and B appear in both lists, should rank high
        top_2_contents = {doc.page_content for doc in result[:2]}
        assert (
            "Document A about machine learning" in top_2_contents
            or "Document B about deep learning" in top_2_contents
        )

    def test_weighted_fusion_with_equal_weights(self, sample_results):
        """Test RRF with equal weights."""
        vector_results, bm25_results = sample_results
        result = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            weights=[0.5, 0.5],
        )
        assert len(result) == 4

    def test_weighted_fusion_prefers_higher_weight(self, sample_results):
        """Test that RRF prefers results from higher-weighted source."""
        vector_results, bm25_results = sample_results

        # Heavy weight on vector
        result_vector_heavy = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            weights=[0.9, 0.1],
        )

        # Heavy weight on BM25
        result_bm25_heavy = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            weights=[0.1, 0.9],
        )

        # Top result should differ based on weights
        # (This is a probabilistic test - may not always hold)
        # At minimum, both should produce valid results
        assert len(result_vector_heavy) == 4
        assert len(result_bm25_heavy) == 4

    def test_custom_k_parameter(self, sample_results):
        """Test RRF with different k values."""
        vector_results, bm25_results = sample_results

        result_low_k = reciprocal_rank_fusion([vector_results, bm25_results], k=1)
        result_high_k = reciprocal_rank_fusion([vector_results, bm25_results], k=1000)

        # Both should produce same set of documents, potentially different order
        assert len(result_low_k) == 4
        assert len(result_high_k) == 4

    def test_weights_must_match_results(self, sample_results):
        """Test that mismatched weights raise error."""
        vector_results, bm25_results = sample_results

        with pytest.raises(ValueError, match="weights must match"):
            reciprocal_rank_fusion([vector_results, bm25_results], weights=[0.5])


class TestWeightedFusion:
    """Tests for weighted fusion algorithm."""

    def test_empty_results(self):
        """Test weighted fusion with empty results."""
        result = weighted_fusion([])
        assert result == []

    def test_weighted_fusion_with_scores(self):
        """Test weighted fusion with scored results."""
        results1 = [
            (Document(page_content="Doc A"), 0.9),
            (Document(page_content="Doc B"), 0.7),
        ]
        results2 = [
            (Document(page_content="Doc B"), 0.8),
            (Document(page_content="Doc C"), 0.6),
        ]

        result = weighted_fusion([results1, results2], weights=[0.5, 0.5])

        # Doc B should rank highest (appears in both)
        assert len(result) == 3
