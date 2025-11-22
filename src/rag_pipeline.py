"""RAG pipeline for retrieval-augmented generation."""

import logging
from typing import Any

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from .config_loader import get_config
from .llm_handler import get_llm_handler
from .vectorstore import get_vectorstore_manager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline for question answering."""

    def __init__(self):
        """Initialize RAG pipeline."""
        self.config = get_config()
        self.llm_handler = get_llm_handler()
        self.vectorstore_manager = get_vectorstore_manager()

        # Load vector store
        self.vectorstore_manager.load_vectorstore()

        # Get LLM
        self.llm = self.llm_handler.get_llm()

        # Setup prompt template
        self.prompt_template = self._create_prompt_template()

        # Create QA chain
        self.qa_chain = self._create_qa_chain()

    def _create_prompt_template(self) -> PromptTemplate:
        """Create prompt template for RAG.

        Returns:
            PromptTemplate instance
        """
        system_prompt = self.llm_handler.get_system_prompt()

        template = f"""{system_prompt}

Context from documents:
{{context}}

Question: {{question}}

Answer: """

        return PromptTemplate(template=template, input_variables=["context", "question"])

    def _create_qa_chain(self) -> RetrievalQA:
        """Create retrieval QA chain.

        Returns:
            RetrievalQA chain
        """
        retriever = self.vectorstore_manager.get_retriever()

        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=self.config.get("rag.include_sources", True),
            chain_type_kwargs={"prompt": self.prompt_template},
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
            response = self.qa_chain.invoke({"query": question})

            # Log retrieved sources
            if "source_documents" in response:
                logger.debug(f"Retrieved {len(response['source_documents'])} source documents")

            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "result": f"I apologize, but I encountered an error: {e}",
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
