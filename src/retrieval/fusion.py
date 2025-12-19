"""Fusion algorithms for combining multiple retrieval results."""

import logging
from collections import defaultdict
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    results_list: list[list[Document]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[Document]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF is a simple but effective method for combining rankings from
    multiple retrieval systems. It's based on the formula:
        RRF(d) = sum(1 / (k + rank(d)))

    where k is a constant (typically 60) and rank(d) is the position
    of document d in each ranking (1-indexed).

    Args:
        results_list: List of ranked document lists from different retrievers
        k: RRF constant (higher = less aggressive re-ranking, default: 60)
        weights: Optional weights for each result list (must sum to 1.0)

    Returns:
        Combined and re-ranked list of documents

    Reference:
        Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
        Reciprocal rank fusion outperforms condorcet and individual
        rank learning methods. SIGIR.
    """
    if not results_list:
        return []

    # Normalize weights if provided
    if weights is None:
        weights = [1.0] * len(results_list)
    else:
        if len(weights) != len(results_list):
            raise ValueError("Number of weights must match number of result lists")
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

    # Calculate RRF scores
    # Use page_content as document identifier (could also use metadata)
    doc_scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for result_idx, results in enumerate(results_list):
        weight = weights[result_idx]
        for rank, doc in enumerate(results, start=1):
            # Create a unique key for the document
            doc_key = doc.page_content

            # RRF formula with weight
            score = weight * (1.0 / (k + rank))
            doc_scores[doc_key] += score

            # Keep the document object (prefer first occurrence)
            if doc_key not in doc_map:
                doc_map[doc_key] = doc

    # Sort by RRF score (descending)
    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

    # Return sorted documents
    return [doc_map[key] for key in sorted_keys]


def weighted_fusion(
    results_list: list[list[tuple[Document, float]]],
    weights: list[float] | None = None,
) -> list[Document]:
    """Combine multiple scored result lists using weighted scoring.

    This method is useful when retrievers provide relevance scores.
    Documents are combined by weighted sum of their scores.

    Args:
        results_list: List of (document, score) tuples from different retrievers
        weights: Weights for each result list (normalized internally)

    Returns:
        Combined and re-ranked list of documents
    """
    if not results_list:
        return []

    # Normalize weights
    if weights is None:
        weights = [1.0] * len(results_list)
    else:
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

    # Combine scores
    doc_scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for result_idx, results in enumerate(results_list):
        weight = weights[result_idx]
        for doc, score in results:
            doc_key = doc.page_content
            doc_scores[doc_key] += weight * score
            if doc_key not in doc_map:
                doc_map[doc_key] = doc

    # Sort by combined score
    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

    return [doc_map[key] for key in sorted_keys]


class FusionRetriever(BaseRetriever):
    """A retriever that combines results from multiple retrievers using fusion.

    This retriever implements the Ensemble pattern, allowing multiple
    retrieval strategies to be combined for better recall and precision.

    Attributes:
        retrievers: List of retrievers to combine
        weights: Weights for each retriever (optional)
        fusion_algorithm: Algorithm to use ('rrf' or 'weighted')
        rrf_k: Constant for RRF algorithm
        final_k: Number of documents to return after fusion
    """

    retrievers: list[BaseRetriever]
    weights: list[float] | None = None
    fusion_algorithm: str = "rrf"
    rrf_k: int = 60
    final_k: int = 4

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,  # noqa: ARG002
    ) -> list[Document]:
        """Get documents from all retrievers and fuse results.

        Args:
            query: Query string to search for
            run_manager: Callback manager

        Returns:
            Fused and re-ranked list of documents
        """
        # Collect results from all retrievers
        all_results: list[list[Document]] = []

        for retriever in self.retrievers:
            try:
                results = retriever.invoke(query)
                all_results.append(results)
                logger.debug(f"Retriever returned {len(results)} documents")
            except Exception as e:
                logger.warning(f"Retriever failed: {e}")
                all_results.append([])

        # Apply fusion algorithm
        if self.fusion_algorithm == "rrf":
            fused = reciprocal_rank_fusion(
                all_results,
                k=self.rrf_k,
                weights=self.weights,
            )
        else:
            # For non-RRF, we don't have scores, so use RRF as fallback
            logger.warning(
                f"Fusion algorithm '{self.fusion_algorithm}' not supported "
                "for rank-only results, falling back to RRF"
            )
            fused = reciprocal_rank_fusion(all_results, k=self.rrf_k, weights=self.weights)

        # Return top k results
        result = fused[: self.final_k]
        logger.debug(f"Fusion returned {len(result)} documents from {len(fused)} total")

        return result

    def get_retriever_info(self) -> dict[str, Any]:
        """Get information about the fusion configuration.

        Returns:
            Dictionary with fusion configuration details
        """
        return {
            "num_retrievers": len(self.retrievers),
            "weights": self.weights,
            "fusion_algorithm": self.fusion_algorithm,
            "rrf_k": self.rrf_k,
            "final_k": self.final_k,
        }
