"""Script to build vector store from documents."""

import logging
import sys
from pathlib import Path

from .config_loader import get_config
from .document_processor import process_documents
from .vectorstore import get_vectorstore_manager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_vectorstore(documents_dir: str | None = None, force_rebuild: bool = False):
    """Build vector store from documents directory.

    Args:
        documents_dir: Path to documents directory (uses config default if None)
        force_rebuild: If True, delete existing vector store before building
    """
    logger.info("=" * 60)
    logger.info("Building Vector Store for Profile Chatbot")
    logger.info("=" * 60)

    # Load configuration
    config = get_config()

    # Get documents directory
    if documents_dir is None:
        documents_dir = config.get_env("DOCUMENTS_DIR", "./data/documents")

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

    # Initialize vector store manager
    vectorstore_manager = get_vectorstore_manager()

    # Handle force rebuild
    if force_rebuild:
        try:
            logger.warning("Force rebuild enabled - deleting existing vector store...")
            vectorstore_manager.delete_collection()
        except Exception as e:
            logger.info(f"No existing collection to delete: {e}")

    # Create vector store
    logger.info("Creating vector store (this may take a few minutes)...")
    try:
        vectorstore_manager.create_vectorstore(documents)
        logger.info("=" * 60)
        logger.info("✅ Vector store built successfully!")
        logger.info(f"   Location: {vectorstore_manager.persist_directory}")
        logger.info(f"   Collection: {vectorstore_manager.collection_name}")
        logger.info(f"   Documents: {len(documents)} chunks")
        logger.info("=" * 60)
        logger.info("\n🚀 You can now run the chatbot with: streamlit run app.py\n")

    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        sys.exit(1)


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Build vector store from documents")
    parser.add_argument(
        "--documents-dir",
        type=str,
        default=None,
        help="Path to documents directory (default: from config)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete existing vector store before building",
    )

    args = parser.parse_args()

    build_vectorstore(documents_dir=args.documents_dir, force_rebuild=args.force_rebuild)


if __name__ == "__main__":
    main()
