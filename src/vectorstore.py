"""Vector store module for ChromaDB operations."""

import logging
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config_loader import get_config

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage ChromaDB vector store for document embeddings."""

    def __init__(self):
        """Initialize vector store manager with configuration."""
        self.config = get_config()

        # Get embedding configuration
        embedding_model = self.config.get(
            "embeddings.model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        device = self.config.get("embeddings.device", "cpu")

        # Initialize embeddings
        logger.info(f"Initializing embeddings model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Get vector store configuration
        self.persist_directory = self.config.get_env("CHROMA_PERSIST_DIR", "./chroma_db")
        self.collection_name = self.config.get("vectorstore.collection_name", "profile_documents")

        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.vectorstore: Chroma | None = None

    def create_vectorstore(self, documents: list[Document]) -> Chroma:
        """Create a new vector store from documents.

        Args:
            documents: List of Document objects to embed

        Returns:
            Chroma vector store instance
        """
        if not documents:
            msg = "No documents provided to create vector store"
            raise ValueError(msg)

        logger.info(f"Creating vector store with {len(documents)} documents...")

        try:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
            )

            logger.info(f"Vector store created successfully at {self.persist_directory}")
            return self.vectorstore

        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    def load_vectorstore(self) -> Chroma:
        """Load existing vector store from disk.

        Returns:
            Chroma vector store instance
        """
        persist_dir = Path(self.persist_directory)

        if not persist_dir.exists():
            msg = f"Vector store not found at {self.persist_directory}"
            raise FileNotFoundError(msg)

        logger.info(f"Loading vector store from {self.persist_directory}...")

        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )

            # Verify the vector store has documents
            collection_count = self.vectorstore._collection.count()
            logger.info(f"Vector store loaded with {collection_count} embeddings")

            return self.vectorstore

        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise

    def get_retriever(self, **kwargs: Any) -> Any:
        """Get retriever for the vector store.

        Args:
            **kwargs: Additional arguments for the retriever

        Returns:
            Retriever instance
        """
        if self.vectorstore is None:
            self.load_vectorstore()

        # Get search configuration from retrieval config
        search_type = self.config.get("retrieval.vector.search_type", "similarity")
        search_kwargs_config = self.config.get("retrieval.vector.search_kwargs", {})
        k = self.config.get("retrieval.vector.k", 4)

        # Build search_kwargs based on search_type
        if search_type == "similarity":
            filtered_kwargs = {"k": k}
        elif search_type == "mmr":
            filtered_kwargs = {
                "k": k,
                "fetch_k": search_kwargs_config.get("fetch_k", 20),
                "lambda_mult": search_kwargs_config.get("lambda_mult", 0.5),
            }
        else:
            filtered_kwargs = {"k": k}

        # Override with provided kwargs
        filtered_kwargs.update(kwargs)

        logger.debug(
            f"Creating retriever with search_type={search_type}, search_kwargs={filtered_kwargs}"
        )

        return self.vectorstore.as_retriever(search_type=search_type, search_kwargs=filtered_kwargs)

    def add_documents(self, documents: list[Document]) -> None:
        """Add new documents to existing vector store.

        Args:
            documents: List of Document objects to add
        """
        if self.vectorstore is None:
            self.load_vectorstore()

        logger.info(f"Adding {len(documents)} documents to vector store...")

        try:
            self.vectorstore.add_documents(documents)
            logger.info("Documents added successfully")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        logger.warning(f"Deleting collection: {self.collection_name}")

        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            client.delete_collection(name=self.collection_name)
            self.vectorstore = None
            logger.info("Collection deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Perform similarity search.

        Args:
            query: Query string
            k: Number of documents to return
            **kwargs: Additional search parameters

        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            self.load_vectorstore()

        return self.vectorstore.similarity_search(query, k=k, **kwargs)


def get_vectorstore_manager() -> VectorStoreManager:
    """Get singleton vector store manager instance."""
    return VectorStoreManager()
