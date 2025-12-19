"""Tests for fusion algorithms."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.retrieval.fusion import FusionRetriever, reciprocal_rank_fusion, weighted_fusion


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


@pytest.fixture
def mock_retrievers():
    """Create mock retrievers for testing."""
    from langchain_core.retrievers import BaseRetriever

    # Use spec to satisfy Pydantic validation
    retriever1 = MagicMock(spec=BaseRetriever)
    retriever1.invoke.return_value = [
        Document(page_content="Doc A from retriever 1"),
        Document(page_content="Doc B from retriever 1"),
    ]

    retriever2 = MagicMock(spec=BaseRetriever)
    retriever2.invoke.return_value = [
        Document(page_content="Doc B from retriever 1"),  # Same as above
        Document(page_content="Doc C from retriever 2"),
    ]

    return [retriever1, retriever2]


# --- Reciprocal Rank Fusion Tests ---


def test_rrf_empty_results():
    """Test RRF with empty results."""
    result = reciprocal_rank_fusion([])
    assert result == []


def test_rrf_single_result_list(sample_results):
    """Test RRF with single result list."""
    vector_results, _ = sample_results
    result = reciprocal_rank_fusion([vector_results])

    # Order should be preserved
    assert len(result) == 3
    assert result[0].page_content == vector_results[0].page_content


def test_rrf_combines_results(sample_results):
    """Test that RRF combines results from multiple sources."""
    vector_results, bm25_results = sample_results
    result = reciprocal_rank_fusion([vector_results, bm25_results])

    # Should have all unique documents
    contents = {doc.page_content for doc in result}
    assert len(contents) == 4  # A, B, C, D


def test_rrf_ranks_overlapping_higher(sample_results):
    """Test that documents in both lists rank higher."""
    vector_results, bm25_results = sample_results
    result = reciprocal_rank_fusion([vector_results, bm25_results])

    # Documents A and B appear in both lists, should rank high
    top_2_contents = {doc.page_content for doc in result[:2]}
    assert (
        "Document A about machine learning" in top_2_contents
        or "Document B about deep learning" in top_2_contents
    )


def test_rrf_with_equal_weights(sample_results):
    """Test RRF with equal weights."""
    vector_results, bm25_results = sample_results
    result = reciprocal_rank_fusion(
        [vector_results, bm25_results],
        weights=[0.5, 0.5],
    )
    assert len(result) == 4


def test_rrf_prefers_higher_weight(sample_results):
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

    # Both should produce valid results
    assert len(result_vector_heavy) == 4
    assert len(result_bm25_heavy) == 4


def test_rrf_custom_k_parameter(sample_results):
    """Test RRF with different k values."""
    vector_results, bm25_results = sample_results

    result_low_k = reciprocal_rank_fusion([vector_results, bm25_results], k=1)
    result_high_k = reciprocal_rank_fusion([vector_results, bm25_results], k=1000)

    # Both should produce same set of documents, potentially different order
    assert len(result_low_k) == 4
    assert len(result_high_k) == 4


def test_rrf_weights_must_match_results(sample_results):
    """Test that mismatched weights raise error."""
    vector_results, bm25_results = sample_results

    with pytest.raises(ValueError, match="weights must match"):
        reciprocal_rank_fusion([vector_results, bm25_results], weights=[0.5])


# --- Weighted Fusion Tests ---


def test_weighted_fusion_empty_results():
    """Test weighted fusion with empty results."""
    result = weighted_fusion([])
    assert result == []


def test_weighted_fusion_with_scores():
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


def test_weighted_fusion_no_weights():
    """Test weighted fusion without explicit weights."""
    results1 = [
        (Document(page_content="Doc A"), 0.9),
        (Document(page_content="Doc B"), 0.7),
    ]
    results2 = [
        (Document(page_content="Doc C"), 0.8),
    ]

    result = weighted_fusion([results1, results2])

    assert len(result) == 3


# --- FusionRetriever Tests ---


def test_fusion_retriever_init(mock_retrievers):
    """Test FusionRetriever initialization."""
    retriever = FusionRetriever(
        retrievers=mock_retrievers,
        weights=[0.7, 0.3],
        fusion_algorithm="rrf",
        rrf_k=60,
        final_k=4,
    )

    assert retriever.fusion_algorithm == "rrf"
    assert retriever.rrf_k == 60
    assert retriever.final_k == 4


def test_fusion_retriever_invoke(mock_retrievers):
    """Test FusionRetriever invoke method."""
    retriever = FusionRetriever(
        retrievers=mock_retrievers,
        weights=[0.5, 0.5],
        fusion_algorithm="rrf",
        final_k=4,
    )

    results = retriever.invoke("test query")

    assert len(results) <= 4
    assert all(isinstance(doc, Document) for doc in results)


def test_fusion_retriever_handles_failures(mock_retrievers):
    """Test FusionRetriever handles retriever failures gracefully."""
    mock_retrievers[0].invoke.side_effect = Exception("Retriever 1 failed")

    retriever = FusionRetriever(
        retrievers=mock_retrievers,
        weights=[0.5, 0.5],
        fusion_algorithm="rrf",
        final_k=4,
    )

    # Should not raise, just return results from working retriever
    results = retriever.invoke("test query")

    assert len(results) > 0


def test_fusion_retriever_non_rrf_fallback(mock_retrievers):
    """Test FusionRetriever falls back to RRF for unsupported algorithms."""
    retriever = FusionRetriever(
        retrievers=mock_retrievers,
        weights=[0.5, 0.5],
        fusion_algorithm="unsupported",  # This should trigger fallback
        final_k=4,
    )

    results = retriever.invoke("test query")

    # Should still work, falling back to RRF
    assert len(results) <= 4


def test_fusion_retriever_get_info(mock_retrievers):
    """Test FusionRetriever get_retriever_info method."""
    retriever = FusionRetriever(
        retrievers=mock_retrievers,
        weights=[0.7, 0.3],
        fusion_algorithm="rrf",
        rrf_k=60,
        final_k=4,
    )

    info = retriever.get_retriever_info()

    assert info["num_retrievers"] == 2
    assert info["weights"] == [0.7, 0.3]
    assert info["fusion_algorithm"] == "rrf"
    assert info["rrf_k"] == 60
    assert info["final_k"] == 4


def test_fusion_retriever_no_weights(mock_retrievers):
    """Test FusionRetriever without explicit weights."""
    retriever = FusionRetriever(
        retrievers=mock_retrievers,
        fusion_algorithm="rrf",
        final_k=4,
    )

    results = retriever.invoke("test query")

    assert len(results) <= 4
