"""Main document loader for guaranteed context availability."""

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from .config_loader import get_config
from .document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class MainDocumentLoader:
    """Load and manage the main profile document."""

    def __init__(self):
        """Initialize main document loader with configuration."""
        self.config = get_config()
        self.doc_processor = DocumentProcessor()

        # Cache management
        self._cached_content: Optional[str] = None
        self._cached_file_hash: Optional[str] = None
        self._last_check_time: float = 0

        # Config
        self.enabled = self.config.get("main_document.enabled", False)
        path_str = self.config.get_env(
            "MAIN_DOCUMENT_PATH", self.config.get("main_document.path", "")
        )
        self.path = Path(path_str) if path_str else None
        self.max_tokens = self.config.get("main_document.max_tokens", 10000)
        self.summarize_if_exceeds = self.config.get("main_document.summarize_if_exceeds", True)
        self.cache_enabled = self.config.get("main_document.cache_enabled", True)
        self.cache_check_interval = self.config.get("main_document.cache_check_interval", 60)

    def load_main_document(self) -> str:
        """Load main document with caching and error handling.

        Returns:
            Main document content as string (empty if disabled/error)
        """
        if not self.enabled:
            logger.debug("Main document feature is disabled")
            return ""

        if not self.path or not self.path.exists():
            if self.config.get("main_document.fail_silently", True):
                logger.warning(f"Main document not found: {self.path}")
                return ""
            else:
                msg = f"Main document not found: {self.path}"
                raise FileNotFoundError(msg)

        # Check cache validity
        if self._should_use_cache():
            logger.debug("Using cached main document")
            return self._cached_content or ""

        # Load fresh content
        try:
            content = self._load_document_by_format()

            # Token management
            token_count = self.count_tokens(content)
            logger.info(f"Main document loaded: {token_count} tokens from {self.path.name}")

            if token_count > self.max_tokens:
                logger.warning(
                    f"Main document exceeds limit ({token_count} > {self.max_tokens} tokens)"
                )
                if self.summarize_if_exceeds:
                    content = self._summarize_content(content)
                    logger.info(f"Summarized to {self.count_tokens(content)} tokens")
                else:
                    content = self.truncate_to_tokens(content, self.max_tokens)
                    logger.info(f"Truncated to {self.max_tokens} tokens")

            # Update cache
            if self.cache_enabled:
                self._update_cache(content)

            return content

        except Exception as e:
            logger.error(f"Error loading main document: {e}")
            if self.config.get("main_document.fail_silently", True):
                return ""
            raise

    def _load_document_by_format(self) -> str:
        """Auto-detect format and load document.

        Returns:
            Document content as string
        """
        if not self.path:
            return ""

        extension = self.path.suffix.lower()

        logger.debug(f"Auto-detected format: {extension}")

        # Use existing DocumentProcessor for format handling
        try:
            # Load as LangChain documents
            if extension == ".pdf":
                docs = self.doc_processor.load_pdf(self.path)
            elif extension in [".docx", ".doc"]:
                docs = self.doc_processor.load_docx(self.path)
            elif extension in [".html", ".htm"]:
                docs = self.doc_processor.load_html(self.path)
            elif extension in [".md", ".txt"]:
                docs = self.doc_processor.load_text(self.path)
            else:
                # Fallback: try as text
                logger.warning(f"Unsupported format {extension}, treating as text")
                with open(self.path, encoding="utf-8") as f:
                    return f.read()

            # Combine all document chunks (main doc shouldn't be chunked)
            content = "\n\n".join(doc.page_content for doc in docs)
            return content

        except Exception as e:
            logger.error(f"Error loading {extension} file: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        try:
            import tiktoken

            # Use cl100k_base encoding (GPT-4, llama compatible)
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

        except ImportError:
            # Fallback: rough estimation (1 token ≈ 4 characters)
            logger.warning("tiktoken not available, using rough estimation (1 token ≈ 4 chars)")
            return len(text) // 4

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to maximum tokens.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated text
        """
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)

            if len(tokens) <= max_tokens:
                return text

            truncated_tokens = tokens[:max_tokens]
            truncated_text = encoding.decode(truncated_tokens)

            return truncated_text + "\n\n[... content truncated ...]"

        except ImportError:
            # Fallback: character-based truncation
            char_limit = max_tokens * 4
            if len(text) <= char_limit:
                return text
            return text[:char_limit] + "\n\n[... content truncated ...]"

    def _summarize_content(self, content: str) -> str:
        """Summarize content using LLM if exceeds token limit.

        Args:
            content: Original content

        Returns:
            Summarized content
        """
        from .llm_handler import get_llm_handler

        logger.info("Summarizing main document using LLM...")

        try:
            llm_handler = get_llm_handler()
            llm = llm_handler.get_llm()

            summarization_prompt = self.config.get(
                "main_document.summarization_prompt",
                "Summarize the following professional profile, preserving all critical information:",
            )

            target_tokens = self.config.get(
                "main_document.summarization_target_tokens", int(self.max_tokens * 0.8)
            )

            # Truncate input if way too large (to fit in LLM context)
            max_input_tokens = 30000  # Conservative limit
            if self.count_tokens(content) > max_input_tokens:
                content = self.truncate_to_tokens(content, max_input_tokens)
                logger.warning(
                    f"Input content too large, truncated to {max_input_tokens} tokens for summarization"
                )

            full_prompt = f"""{summarization_prompt}

---
{content}
---

Provide a comprehensive summary (target: ~{target_tokens} tokens):"""

            summary = llm.invoke(full_prompt)

            # Verify summary is actually shorter
            summary_tokens = self.count_tokens(summary)
            if summary_tokens > self.max_tokens:
                logger.warning(
                    f"Summary still exceeds limit ({summary_tokens} tokens), truncating"
                )
                summary = self.truncate_to_tokens(summary, self.max_tokens)

            return summary

        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            logger.info("Falling back to truncation")
            return self.truncate_to_tokens(content, self.max_tokens)

    def _should_use_cache(self) -> bool:
        """Check if cached content is still valid.

        Returns:
            True if cache should be used, False otherwise
        """
        if not self.cache_enabled or self._cached_content is None:
            return False

        # Check if enough time has passed
        current_time = time.time()
        if current_time - self._last_check_time < self.cache_check_interval:
            return True

        # Check if file has changed
        current_hash = self._calculate_file_hash()
        if current_hash == self._cached_file_hash:
            self._last_check_time = current_time
            return True

        return False

    def _calculate_file_hash(self) -> str:
        """Calculate MD5 hash of file for change detection.

        Returns:
            MD5 hash string
        """
        if not self.path or not self.path.exists():
            return ""

        hash_md5 = hashlib.md5()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _update_cache(self, content: str) -> None:
        """Update cache with new content.

        Args:
            content: Content to cache
        """
        self._cached_content = content
        self._cached_file_hash = self._calculate_file_hash()
        self._last_check_time = time.time()
        logger.debug("Main document cache updated")

    def invalidate_cache(self) -> None:
        """Manually invalidate cache (useful for testing/reloading)."""
        self._cached_content = None
        self._cached_file_hash = None
        self._last_check_time = 0
        logger.info("Main document cache invalidated")


# Singleton instance
_loader_instance: Optional[MainDocumentLoader] = None


def get_main_document_loader() -> MainDocumentLoader:
    """Get singleton main document loader instance.

    Returns:
        MainDocumentLoader instance
    """
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = MainDocumentLoader()
    return _loader_instance


def reload_main_document_loader() -> MainDocumentLoader:
    """Reload main document loader (useful for testing).

    Returns:
        New MainDocumentLoader instance
    """
    global _loader_instance
    _loader_instance = MainDocumentLoader()
    return _loader_instance

