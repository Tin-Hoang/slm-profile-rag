"""Script to build indexes for retrieval strategies."""

import logging
import sys
from pathlib import Path

from .config_loader import get_config
from .document_processor import process_documents

# Import retrieval components (this registers strategies with the factory)
from .retrieval import RetrieverFactory
from .retrieval.strategies import BM25Strategy, BM25VectorStrategy, VectorStrategy  # noqa: F401

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_index(
    documents_dir: str | None = None,
    strategy: str | None = None,
    force_rebuild: bool = False,
):
    """Build index for the specified retrieval strategy.

    Args:
        documents_dir: Path to documents directory (uses config default if None)
        strategy: Retrieval strategy to use (uses config default if None)
        force_rebuild: If True, delete existing indexes before building
    """
    logger.info("=" * 60)
    logger.info("Building Index for Profile Chatbot")
    logger.info("=" * 60)

    # Load configuration
    config = get_config()

    # Get strategy from config if not provided
    if strategy is None:
        strategy = config.get("retrieval.strategy", "vector")

    logger.info(f"Retrieval Strategy: {strategy}")

    # Get documents directory
    if documents_dir is None:
        documents_dir = config.get_env("DOCUMENTS_DIR", "./data/documents")

    # Log main document info
    main_doc_enabled = config.get("main_document.enabled", False)
    main_doc_path = config.get("main_document.path", "")
    if main_doc_enabled and main_doc_path:
        logger.info("Main Document Feature: ENABLED")
        logger.info(f"  Main document ({Path(main_doc_path).name}) will be excluded from index")
        logger.info("  (It's loaded directly into prompts, not retrieved)")
        logger.info("")

    documents_path = Path(documents_dir)

    if not documents_path.exists():
        logger.error(f"Documents directory not found: {documents_dir}")
        logger.error("Please create the directory and add your profile documents.")
        sys.exit(1)

    # Check if directory has files
    files = list(documents_path.rglob("*"))
    if not files or all(f.is_dir() for f in files):
        logger.warning(f"No files found in {documents_dir}")
        logger.warning("Please add your PDF, Word, HTML, or text documents to this directory.")
        sys.exit(1)

    # Process documents
    logger.info(f"Processing documents from: {documents_dir}")
    documents = process_documents(documents_dir)

    if not documents:
        logger.error("No documents were processed. Check your files and try again.")
        sys.exit(1)

    logger.info(f"Successfully processed {len(documents)} document chunks")

    # Handle force rebuild for BM25 index
    if force_rebuild and strategy in ["bm25", "bm25_vector"]:
        bm25_path = Path(config.get("retrieval.bm25.persist_path", "./bm25_index"))
        if bm25_path.exists():
            import shutil

            logger.warning(f"Force rebuild - deleting BM25 index at {bm25_path}")
            shutil.rmtree(bm25_path)

    # Handle force rebuild for vector store
    if force_rebuild and strategy in ["vector", "bm25_vector"]:
        try:
            from .vectorstore import get_vectorstore_manager

            vectorstore_manager = get_vectorstore_manager()
            logger.warning("Force rebuild - deleting vector store...")
            vectorstore_manager.delete_collection()
        except Exception as e:
            logger.info(f"No existing vector collection to delete: {e}")

    # Create retrieval strategy
    logger.info(f"Creating retrieval strategy: {strategy}")

    try:
        # Get the full config as dict for the strategy
        config_dict = {
            "retrieval": {
                "strategy": strategy,
                "final_k": config.get("retrieval.final_k", 4),
                "vector": {
                    "search_type": config.get("retrieval.vector.search_type", "similarity"),
                    "k": config.get("retrieval.vector.k", 10),
                    "search_kwargs": config.get("retrieval.vector.search_kwargs", {}),
                },
                "bm25": {
                    "k": config.get("retrieval.bm25.k", 10),
                    "persist_path": config.get("retrieval.bm25.persist_path", "./bm25_index"),
                    "tokenizer": config.get("retrieval.bm25.tokenizer", "simple"),
                },
                "fusion": {
                    "algorithm": config.get("retrieval.fusion.algorithm", "rrf"),
                    "rrf_k": config.get("retrieval.fusion.rrf_k", 60),
                    "weights": config.get("retrieval.fusion.weights", {"vector": 0.7, "bm25": 0.3}),
                },
            }
        }

        retrieval_strategy = RetrieverFactory.create(strategy, config_dict)

        # Build the index
        logger.info("Building index (this may take a few minutes)...")
        retrieval_strategy.build_index(documents)

        # Get stats
        stats = retrieval_strategy.get_index_stats()

        logger.info("=" * 60)
        logger.info("✅ Index built successfully!")
        logger.info(f"   Strategy: {strategy}")
        logger.info(f"   Documents: {len(documents)} chunks")

        if strategy == "vector":
            logger.info(f"   Vector Store: {stats.get('persist_directory', 'N/A')}")
        elif strategy == "bm25":
            logger.info(f"   BM25 Index: {stats.get('persist_path', 'N/A')}")
        elif strategy == "bm25_vector":
            vector_stats = stats.get("vector", {})
            bm25_stats = stats.get("bm25", {})
            logger.info(f"   Vector Store: {vector_stats.get('persist_directory', 'N/A')}")
            logger.info(f"   BM25 Index: {bm25_stats.get('persist_path', 'N/A')}")
            fusion_stats = stats.get("fusion", {})
            logger.info(
                f"   Fusion: {fusion_stats.get('algorithm', 'rrf')} "
                f"(vector={fusion_stats.get('weights', {}).get('vector', 0.7)}, "
                f"bm25={fusion_stats.get('weights', {}).get('bm25', 0.3)})"
            )

        # Log main document status
        if main_doc_enabled and main_doc_path:
            logger.info(f"   Main Doc: {Path(main_doc_path).name} (loaded directly)")

        logger.info("=" * 60)
        logger.info("\n🚀 You can now run the chatbot with: streamlit run app.py\n")

    except Exception as e:
        logger.error(f"Error building index: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build retrieval index from documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build with default strategy from config
  python -m src.build_vectorstore

  # Build vector-only index
  python -m src.build_vectorstore --strategy vector

  # Build hybrid BM25 + Vector index
  python -m src.build_vectorstore --strategy bm25_vector

  # Force rebuild from scratch
  python -m src.build_vectorstore --strategy bm25_vector --force-rebuild

Available strategies:
  vector      - Semantic similarity search using embeddings
  bm25        - Lexical/keyword search using BM25 algorithm
  bm25_vector - Hybrid combining BM25 and vector search (default)
        """,
    )
    parser.add_argument(
        "--documents-dir",
        type=str,
        default=None,
        help="Path to documents directory (default: from config)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["vector", "bm25", "bm25_vector"],
        help="Retrieval strategy to use (default: from config)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete existing indexes before building",
    )

    args = parser.parse_args()

    build_index(
        documents_dir=args.documents_dir,
        strategy=args.strategy,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
