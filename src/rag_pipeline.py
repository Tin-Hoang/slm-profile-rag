"""RAG pipeline for retrieval-augmented generation."""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from .config_loader import get_config
from .llm_handler import get_llm_handler
from .response_enhancer import get_response_enhancer
from .vectorstore import get_vectorstore_manager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline for question answering."""

    def __init__(self):
        """Initialize RAG pipeline."""
        self.config = get_config()
        self.llm_handler = get_llm_handler()
        self.vectorstore_manager = get_vectorstore_manager()
        self.response_enhancer = get_response_enhancer()

        # Load vector store
        self.vectorstore_manager.load_vectorstore()

        # Get LLM
        self.llm = self.llm_handler.get_llm()

        # Get retriever
        self.retriever = self.vectorstore_manager.get_retriever()

        # Setup prompt template
        self.prompt_template = self._create_prompt_template()

        # Create QA chain using LCEL
        self.qa_chain = self._create_qa_chain()

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create prompt template for RAG.

        Returns:
            ChatPromptTemplate instance
        """
        system_prompt = self.llm_handler.get_system_prompt()

        template = f"""{system_prompt}

Context from documents:
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
        return "\n\n".join(doc.page_content for doc in docs)

    def _create_qa_chain(self):
        """Create retrieval QA chain using LCEL.

        Returns:
            LCEL chain
        """
        # Create the RAG chain using LCEL
        chain = (
            {
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


def get_rag_pipeline() -> RAGPipeline:
    """Get RAG pipeline instance."""
    return RAGPipeline()
