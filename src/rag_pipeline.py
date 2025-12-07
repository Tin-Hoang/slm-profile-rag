"""RAG pipeline for retrieval-augmented generation."""

import logging
from collections.abc import Iterator
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from .config_loader import get_config
from .llm_handler import get_llm_handler
from .main_document_loader import get_main_document_loader
from .response_enhancer import get_response_enhancer
from .vectorstore import get_vectorstore_manager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline for question answering."""

    def __init__(self, llm_handler=None):
        """Initialize RAG pipeline.

        Args:
            llm_handler: Optional LLMHandler instance for custom provider/model
        """
        self.config = get_config()
        self.llm_handler = llm_handler or get_llm_handler()
        self.vectorstore_manager = get_vectorstore_manager()
        self.response_enhancer = get_response_enhancer()
        self.main_doc_loader = get_main_document_loader()

        # Load vector store
        self.vectorstore_manager.load_vectorstore()

        # Get LLM
        self.llm = self.llm_handler.get_llm()

        # Get retriever
        self.retriever = self.vectorstore_manager.get_retriever()

        # Load main document if enabled
        self.main_doc_content = ""
        if self.config.get("main_document.enabled", False):
            self.main_doc_content = self.main_doc_loader.load_main_document()
            if self.main_doc_content:
                token_count = self.main_doc_loader.count_tokens(self.main_doc_content)
                logger.info(f"Main document loaded: {token_count} tokens available in context")

        # Calculate and log token budget if main doc is loaded
        if self.main_doc_content:
            budget = self._calculate_context_budget()
            logger.info(
                f"Token budget - Main doc: {budget['main_doc_tokens']}, "
                f"Available for retrieval: {budget['available_for_retrieval']}"
            )

        # Setup prompt template
        self.prompt_template = self._create_prompt_template()

        # Create QA chain using LCEL
        self.qa_chain = self._create_qa_chain()

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create prompt template for RAG with main document support.

        Returns:
            ChatPromptTemplate instance
        """
        system_prompt = self.llm_handler.get_system_prompt()

        # Structure: System Prompt → Main Doc (BEFORE) → VectorDB Context → Question
        template = f"""{system_prompt}

{{main_document_section}}
{{context}}

Question: {{question}}

Answer: """

        return ChatPromptTemplate.from_template(template)

    def _format_docs(self, docs: list[Document]) -> str:
        """Format documents into a single string.

        Args:
            docs: List of documents

        Returns:
            Formatted string
        """
        if not docs:
            return ""
        return "\n\n".join(doc.page_content for doc in docs)

    def _format_main_doc(self) -> str:
        """Format main document section for prompt.

        Returns:
            Formatted main document section (empty if not available)
        """
        if not self.main_doc_content:
            return ""

        # Clear section header for main document
        return f"""
=== ESSENTIAL PROFILE INFORMATION ===
(This information is always available and takes priority)

{self.main_doc_content}

=== ADDITIONAL CONTEXT FROM DOCUMENTS ===
"""

    def _create_qa_chain(self):
        """Create retrieval QA chain using LCEL with main document.

        Returns:
            LCEL chain
        """
        # Create the RAG chain with main document positioned BEFORE retrieval context
        chain = (
            {
                "main_document_section": lambda _: self._format_main_doc(),
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        return chain

    def query(self, question: str) -> dict[str, Any]:
        """Query the RAG pipeline.

        Args:
            question: User question

        Returns:
            Dictionary with 'result' and optionally 'source_documents'
        """
        logger.info(f"Processing query: {question}")

        try:
            # Get the answer from the chain
            answer = self.qa_chain.invoke(question)

            # Enhance the response for better tone and professionalism
            if self.config.get("rag.enhance_responses", True):
                original_answer = answer
                answer = self.response_enhancer.enhance_with_context(answer, question)
                if answer != original_answer:
                    logger.debug("Response enhanced for better tone")

            # Retrieve source documents separately if needed
            source_documents = []
            if self.config.get("rag.include_sources", True):
                source_documents = self.retriever.invoke(question)
                logger.debug(f"Retrieved {len(source_documents)} source documents")

            return {
                "result": answer,
                "source_documents": source_documents,
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "result": "I encountered a technical issue. Please try rephrasing your question or reach out directly to discuss further.",
                "source_documents": [],
            }

    def get_answer(self, question: str) -> str:
        """Get answer to a question (simplified interface).

        Args:
            question: User question

        Returns:
            Answer string
        """
        response = self.query(question)
        return response.get("result", "I couldn't generate an answer.")

    def get_answer_with_sources(self, question: str) -> tuple[str, list[Document]]:
        """Get answer with source documents.

        Args:
            question: User question

        Returns:
            Tuple of (answer, source_documents)
        """
        response = self.query(question)
        answer = response.get("result", "I couldn't generate an answer.")
        sources = response.get("source_documents", [])
        return answer, sources

    def stream_query(self, question: str) -> Iterator[str]:
        """Stream the response token by token.

        Args:
            question: User question

        Yields:
            Response chunks as they are generated
        """
        logger.info(f"Streaming query: {question}")

        try:
            # Stream the response using LCEL's stream method
            yield from self.qa_chain.stream(question)

        except Exception as e:
            logger.error(f"Error streaming query: {e}")
            yield "I encountered a technical issue. Please try rephrasing your question."

    def stream_answer(self, question: str) -> Iterator[str]:
        """Stream answer to a question (simplified interface).

        Args:
            question: User question

        Yields:
            Answer chunks as they are generated
        """
        yield from self.stream_query(question)

    def get_source_documents(self, question: str) -> list[Document]:
        """Get source documents for a question (can run in parallel with streaming).

        Args:
            question: User question

        Returns:
            List of source documents
        """
        if self.config.get("rag.include_sources", True):
            return self.retriever.invoke(question)
        return []

    def format_sources(self, sources: list[Document]) -> str:
        """Format source documents for display.

        Args:
            sources: List of source documents

        Returns:
            Formatted source string
        """
        if not sources:
            return ""

        source_max_length = self.config.get("rag.source_max_length", 150)
        formatted_sources = []

        for i, doc in enumerate(sources, 1):
            source_name = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f" (Page {page + 1})" if page != "" else ""

            content = doc.page_content[:source_max_length]
            if len(doc.page_content) > source_max_length:
                content += "..."

            formatted_sources.append(f"{i}. **{source_name}{page_info}**\n   {content}")

        return "\n\n".join(formatted_sources)

    def get_main_document_info(self) -> dict[str, Any]:
        """Get information about loaded main document.

        Returns:
            Dictionary with main document metadata
        """
        if not self.main_doc_content:
            return {
                "enabled": self.config.get("main_document.enabled", False),
                "loaded": False,
            }

        return {
            "enabled": True,
            "loaded": True,
            "tokens": self.main_doc_loader.count_tokens(self.main_doc_content),
            "path": str(self.main_doc_loader.path),
            "size_bytes": len(self.main_doc_content.encode("utf-8")),
        }

    def reload_main_document(self) -> bool:
        """Reload main document (useful for runtime updates).

        Returns:
            True if reload successful, False otherwise
        """
        try:
            self.main_doc_loader.invalidate_cache()
            self.main_doc_content = self.main_doc_loader.load_main_document()
            logger.info("Main document reloaded successfully")

            # Recalculate budget if main doc is loaded
            if self.main_doc_content:
                budget = self._calculate_context_budget()
                logger.info(
                    f"Token budget after reload - Main doc: {budget['main_doc_tokens']}, "
                    f"Available for retrieval: {budget['available_for_retrieval']}"
                )

            return True
        except Exception as e:
            logger.error(f"Error reloading main document: {e}")
            return False

    def _calculate_context_budget(self) -> dict[str, int]:
        """Calculate and log token budget distribution.

        Returns:
            Dictionary with token budget breakdown
        """
        # Model-specific context windows
        model_name = self.config.get("llm.model", "llama3.2:3b")
        context_windows = {
            "llama3.2": 8192,
            "llama3.1": 128000,
            "phi3": 4096,
            "gemma2": 8192,
        }

        # Extract base model name (before colon)
        base_model = model_name.split(":")[0] if ":" in model_name else model_name

        model_context_window = context_windows.get(base_model, 8192)

        max_output_tokens = self.config.get("llm.max_tokens", 512)
        main_doc_tokens = self.main_doc_loader.count_tokens(self.main_doc_content)

        # Reserve space: Main Doc + Output + Safety Buffer
        buffer_tokens = 500
        available_for_retrieval = (
            model_context_window - main_doc_tokens - max_output_tokens - buffer_tokens
        )

        budget = {
            "model_context_window": model_context_window,
            "main_doc_tokens": main_doc_tokens,
            "max_output_tokens": max_output_tokens,
            "buffer_tokens": buffer_tokens,
            "available_for_retrieval": max(0, available_for_retrieval),
            "total_input_budget": model_context_window - max_output_tokens,
        }

        logger.debug(f"Token budget: {budget}")

        # Warning if main doc is too large
        usage_percent = (
            (main_doc_tokens / model_context_window) * 100 if model_context_window > 0 else 0
        )
        if usage_percent > 50:
            logger.warning(
                f"Main document uses {usage_percent:.1f}% of context window. "
                f"Consider summarizing or reducing size."
            )

        return budget


def get_rag_pipeline() -> RAGPipeline:
    """Get RAG pipeline instance."""
    return RAGPipeline()
