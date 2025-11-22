"""RAG pipeline implementation using Ollama and ChromaDB."""

import logging
from typing import Any

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

from slm_profile_rag.config import Config
from slm_profile_rag.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline for profile Q&A."""

    def __init__(self, config: Config):
        """Initialize RAG pipeline.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None

    def initialize(self):
        """Initialize all components of the RAG pipeline."""
        # Initialize Ollama LLM
        self.llm = Ollama(
            model=self.config.ollama_model,
            base_url=self.config.ollama_base_url,
            temperature=self.config.get("model.temperature", 0.7),
        )

        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model=self.config.ollama_model, base_url=self.config.ollama_base_url
        )

        # Initialize or load vector store
        persist_directory = self.config.get(
            "vector_store.persist_directory", self.config.vector_store_path
        )
        collection_name = self.config.get("vector_store.collection_name", "profile_documents")

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

    def load_documents(self, force_reload: bool = False):
        """Load documents into the vector store.

        Args:
            force_reload: Force reload documents even if vector store exists.
        """
        # Check if vector store already has documents
        try:
            doc_count = self.vector_store._collection.count()
        except AttributeError:
            # Fallback if _collection is not available
            doc_count = 0

        if not force_reload and doc_count > 0:
            logger.info(f"Vector store already contains {doc_count} documents")
            return

        # Process documents
        processor = DocumentProcessor(
            chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap
        )

        try:
            documents = processor.process_documents(self.config.profile_docs_path)
            logger.info(f"Processed {len(documents)} document chunks")

            # Add documents to vector store
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} chunks to vector store")
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Warning: {e}")
            logger.info("Vector store will be empty until documents are added.")

    def create_qa_chain(self):
        """Create the question-answering chain."""
        # Get prompt templates from config
        system_prompt = self.config.get("prompts.system_prompt", "")
        qa_prompt_template = self.config.get("prompts.qa_prompt", "{context}\n\n{question}")

        # Create custom prompt
        prompt = PromptTemplate(
            template=f"{system_prompt}\n\n{qa_prompt_template}",
            input_variables=["context", "question"],
        )

        # Create retrieval QA chain
        top_k = self.config.get("retrieval.top_k", 3)
        search_type = self.config.get("retrieval.search_type", "similarity")

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type=search_type, search_kwargs={"k": top_k}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

    def query(self, question: str) -> dict[str, Any]:
        """Query the RAG system.

        Args:
            question: User question.

        Returns:
            Dictionary with answer and source documents.
        """
        if not self.qa_chain:
            raise RuntimeError("QA chain not initialized. Call create_qa_chain() first.")

        result = self.qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "source_documents": result.get("source_documents", []),
        }

    def setup(self, force_reload: bool = False):
        """Setup the complete RAG pipeline.

        Args:
            force_reload: Force reload documents even if vector store exists.
        """
        logger.info("Initializing RAG pipeline...")
        self.initialize()
        logger.info("Loading documents...")
        self.load_documents(force_reload=force_reload)
        logger.info("Creating QA chain...")
        self.create_qa_chain()
        logger.info("RAG pipeline ready!")
