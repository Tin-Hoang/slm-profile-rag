"""RAG pipeline for retrieval-augmented generation."""

import logging
from collections.abc import Iterator
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from .config_loader import get_config
from .llm_handler import get_llm_handler
from .main_document_loader import get_main_document_loader
from .response_enhancer import get_response_enhancer

# Import retrieval components
from .retrieval import RetrieverFactory
from .retrieval.strategies import BM25Strategy, BM25VectorStrategy, VectorStrategy  # noqa: F401

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline for question answering."""

    def __init__(self, llm_handler=None, retrieval_strategy: str | None = None):
        """Initialize RAG pipeline.

        Args:
            llm_handler: Optional LLMHandler instance for custom provider/model
            retrieval_strategy: Override retrieval strategy (uses config if None)
        """
        self.config = get_config()
        self.llm_handler = llm_handler or get_llm_handler()
        self.response_enhancer = get_response_enhancer()
        self.main_doc_loader = get_main_document_loader()

        # Get LLM
        self.llm = self.llm_handler.get_llm()

        # Initialize retrieval strategy
        self.retrieval_strategy_name = retrieval_strategy or self.config.get(
            "retrieval.strategy", "vector"
        )
        self.retrieval_strategy = self._create_retrieval_strategy()

        # Get retriever from strategy
        self.retriever = self._get_retriever()

        logger.info(f"Initialized RAG pipeline with '{self.retrieval_strategy_name}' strategy")

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
        """Create prompt template for RAG with main document and chat history support.

        Returns:
            ChatPromptTemplate instance
        """
        system_prompt = self.llm_handler.get_system_prompt()

        # Structure: System Prompt → Chat History → Main Doc → VectorDB Context → Question
        template = f"""{system_prompt}

{{chat_history}}

{{main_document_section}}
{{context}}

Question: {{question}}

Answer: """

        return ChatPromptTemplate.from_template(template)

    def _format_chat_history(self, chat_history: list[dict] | None) -> str:
        """Format chat history for inclusion in prompt.

        Args:
            chat_history: List of message dicts with 'role' and 'content' keys

        Returns:
            Formatted chat history string
        """
        if not chat_history:
            return ""

        formatted_lines = []
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                formatted_lines.append(f"User: {content}")
            elif role == "assistant":
                formatted_lines.append(f"Assistant: {content}")

        if formatted_lines:
            return "=== PREVIOUS CONVERSATION ===\n" + "\n".join(formatted_lines) + "\n\n"

        return ""

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
        """Create retrieval QA chain using LCEL with main document and chat history.

        Returns:
            LCEL chain
        """
        # Create the RAG chain with main document positioned BEFORE retrieval context
        # Chat history will be injected via the invoke/stream methods
        # The chain expects a dict with 'question' and optionally 'chat_history'
        chain = (
            {
                "main_document_section": lambda _: self._format_main_doc(),
                "context": (lambda x: x.get("question", x) if isinstance(x, dict) else x)
                | self.retriever
                | self._format_docs,
                "question": lambda x: x.get("question", x) if isinstance(x, dict) else x,
                "chat_history": lambda x: self._format_chat_history(x.get("chat_history", []))
                if isinstance(x, dict)
                else "",
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        return chain

    def query(self, question: str, chat_history: list[dict] | None = None) -> dict[str, Any]:
        """Query the RAG pipeline.

        Args:
            question: User question
            chat_history: Optional list of previous messages with 'role' and 'content' keys

        Returns:
            Dictionary with 'result' and optionally 'source_documents'
        """
        logger.info(f"Processing query: {question}")
        if chat_history:
            logger.debug(f"Including {len(chat_history)} previous messages in context")

        try:
            # Prepare input with chat history
            chain_input = {
                "question": question,
                "chat_history": chat_history or [],
            }

            # Get the answer from the chain
            answer = self.qa_chain.invoke(chain_input)

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

    def get_answer(self, question: str, chat_history: list[dict] | None = None) -> str:
        """Get answer to a question (simplified interface).

        Args:
            question: User question
            chat_history: Optional list of previous messages with 'role' and 'content' keys

        Returns:
            Answer string
        """
        response = self.query(question, chat_history=chat_history)
        return response.get("result", "I couldn't generate an answer.")

    def get_answer_with_sources(
        self, question: str, chat_history: list[dict] | None = None
    ) -> tuple[str, list[Document]]:
        """Get answer with source documents.

        Args:
            question: User question
            chat_history: Optional list of previous messages with 'role' and 'content' keys

        Returns:
            Tuple of (answer, source_documents)
        """
        response = self.query(question, chat_history=chat_history)
        answer = response.get("result", "I couldn't generate an answer.")
        sources = response.get("source_documents", [])
        return answer, sources

    def stream_query(self, question: str, chat_history: list[dict] | None = None) -> Iterator[str]:
        """Stream the response token by token.

        Args:
            question: User question
            chat_history: Optional list of previous messages with 'role' and 'content' keys

        Yields:
            Response chunks as they are generated
        """
        logger.info(f"Streaming query: {question}")
        if chat_history:
            logger.debug(f"Including {len(chat_history)} previous messages in context")

        try:
            # Prepare input with chat history
            chain_input = {
                "question": question,
                "chat_history": chat_history or [],
            }

            # Stream the response using LCEL's stream method
            yield from self.qa_chain.stream(chain_input)

        except Exception as e:
            logger.error(f"Error streaming query: {e}")
            yield "I encountered a technical issue. Please try rephrasing your question."

    def stream_answer(self, question: str, chat_history: list[dict] | None = None) -> Iterator[str]:
        """Stream answer to a question (simplified interface).

        Args:
            question: User question
            chat_history: Optional list of previous messages with 'role' and 'content' keys

        Yields:
            Answer chunks as they are generated
        """
        yield from self.stream_query(question, chat_history=chat_history)

    def get_source_documents(
        self, question: str, _chat_history: list[dict] | None = None
    ) -> list[Document]:
        """Get source documents for a question (can run in parallel with streaming).

        Args:
            question: User question
            _chat_history: Optional list of previous messages (not used for retrieval, kept for API consistency)

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

        Note: Chat history tokens are handled dynamically per query and are not
        included in this static budget calculation. The app.py truncation logic
        ensures chat history fits within the configured limits.

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

        # Reserve space: Main Doc + Output + Safety Buffer + Chat History (estimated)
        buffer_tokens = 500

        # Estimate chat history tokens (max configured limit)
        max_history_tokens = self.config.get("chat.max_history_tokens", 2000)
        estimated_history_tokens = (
            max_history_tokens if self.config.get("chat.enable_history", True) else 0
        )

        available_for_retrieval = (
            model_context_window
            - main_doc_tokens
            - max_output_tokens
            - buffer_tokens
            - estimated_history_tokens
        )

        budget = {
            "model_context_window": model_context_window,
            "main_doc_tokens": main_doc_tokens,
            "max_output_tokens": max_output_tokens,
            "buffer_tokens": buffer_tokens,
            "estimated_chat_history_tokens": estimated_history_tokens,
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

        # Warning if estimated total usage is high
        total_estimated = main_doc_tokens + estimated_history_tokens + buffer_tokens
        total_percent = (
            (total_estimated / model_context_window) * 100 if model_context_window > 0 else 0
        )
        if total_percent > 70:
            logger.warning(
                f"Estimated context usage ({total_percent:.1f}%) is high. "
                f"Consider reducing main document size or chat history limits."
            )

        return budget

    def _create_retrieval_strategy(self):
        """Create the retrieval strategy based on configuration.

        Returns:
            BaseRetrieverStrategy instance
        """
        # Build config dict for the strategy
        config_dict = {
            "retrieval": {
                "strategy": self.retrieval_strategy_name,
                "final_k": self.config.get("retrieval.final_k", 4),
                "vector": {
                    "search_type": self.config.get("retrieval.vector.search_type", "similarity"),
                    "k": self.config.get("retrieval.vector.k", 10),
                    "search_kwargs": self.config.get("retrieval.vector.search_kwargs", {}),
                },
                "bm25": {
                    "k": self.config.get("retrieval.bm25.k", 10),
                    "persist_path": self.config.get("retrieval.bm25.persist_path", "./bm25_index"),
                    "tokenizer": self.config.get("retrieval.bm25.tokenizer", "simple"),
                },
                "fusion": {
                    "algorithm": self.config.get("retrieval.fusion.algorithm", "rrf"),
                    "rrf_k": self.config.get("retrieval.fusion.rrf_k", 60),
                    "weights": self.config.get(
                        "retrieval.fusion.weights", {"vector": 0.7, "bm25": 0.3}
                    ),
                },
            }
        }

        strategy = RetrieverFactory.create(self.retrieval_strategy_name, config_dict)

        # Load existing index
        if not strategy.load_index():
            logger.warning(
                f"Could not load index for '{self.retrieval_strategy_name}' strategy. "
                f"Run 'python -m src.build_vectorstore --strategy {self.retrieval_strategy_name}' first."
            )

        return strategy

    def _get_retriever(self) -> BaseRetriever:
        """Get the LangChain retriever from the strategy.

        Returns:
            BaseRetriever instance
        """
        final_k = self.config.get("retrieval.final_k", 4)
        return self.retrieval_strategy.as_retriever(k=final_k)

    def get_retrieval_info(self) -> dict[str, Any]:
        """Get information about the current retrieval strategy.

        Returns:
            Dictionary with retrieval strategy information
        """
        return {
            "strategy": self.retrieval_strategy_name,
            "stats": self.retrieval_strategy.get_index_stats(),
        }


def get_rag_pipeline(retrieval_strategy: str | None = None) -> RAGPipeline:
    """Get RAG pipeline instance.

    Args:
        retrieval_strategy: Override retrieval strategy (uses config if None)

    Returns:
        RAGPipeline instance
    """
    return RAGPipeline(retrieval_strategy=retrieval_strategy)
