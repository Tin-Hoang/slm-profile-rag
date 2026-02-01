"""Tests for chat history feature in RAG pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from src.rag_pipeline import RAGPipeline


class TestFormatChatHistory:
    """Test _format_chat_history method."""

    def test_format_chat_history_empty(self):
        """Test formatting empty chat history."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_handler.get_llm.return_value = MagicMock()
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            result = pipeline._format_chat_history(None)
            assert result == ""

            result = pipeline._format_chat_history([])
            assert result == ""

    def test_format_chat_history_single_user_message(self):
        """Test formatting single user message."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_handler.get_llm.return_value = MagicMock()
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            history = [{"role": "user", "content": "Hello"}]
            result = pipeline._format_chat_history(history)

            assert "PREVIOUS CONVERSATION" in result
            assert "User: Hello" in result

    def test_format_chat_history_single_assistant_message(self):
        """Test formatting single assistant message."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_handler.get_llm.return_value = MagicMock()
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            history = [{"role": "assistant", "content": "Hi there!"}]
            result = pipeline._format_chat_history(history)

            assert "PREVIOUS CONVERSATION" in result
            assert "Assistant: Hi there!" in result

    def test_format_chat_history_conversation(self):
        """Test formatting full conversation."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_handler.get_llm.return_value = MagicMock()
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            history = [
                {"role": "user", "content": "What is your name?"},
                {"role": "assistant", "content": "I'm an AI assistant."},
                {"role": "user", "content": "Tell me more."},
            ]
            result = pipeline._format_chat_history(history)

            assert "PREVIOUS CONVERSATION" in result
            assert "User: What is your name?" in result
            assert "Assistant: I'm an AI assistant." in result
            assert "User: Tell me more." in result

    def test_format_chat_history_missing_role(self):
        """Test formatting with missing role field."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_handler.get_llm.return_value = MagicMock()
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            history = [{"content": "Some message"}]
            result = pipeline._format_chat_history(history)

            # Should return empty string if role is missing
            assert result == ""

    def test_format_chat_history_missing_content(self):
        """Test formatting with missing content field."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_handler.get_llm.return_value = MagicMock()
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            history = [{"role": "user", "content": ""}]
            result = pipeline._format_chat_history(history)

            # Function formats empty content (doesn't filter it out)
            assert "PREVIOUS CONVERSATION" in result
            assert "User: " in result

    def test_format_chat_history_invalid_role(self):
        """Test formatting with invalid role."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_handler.get_llm.return_value = MagicMock()
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            history = [{"role": "system", "content": "System message"}]
            result = pipeline._format_chat_history(history)

            # Should return empty string for non-user/assistant roles
            assert result == ""


class TestQueryWithChatHistory:
    """Test query method with chat history."""

    def test_query_without_chat_history(self):
        """Test query without chat history."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = "Test response"
            mock_handler.get_llm.return_value = mock_llm
            mock_handler.get_system_prompt.return_value = "System prompt"
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            response = pipeline.query("What is AI?")

            assert "result" in response
            assert isinstance(response["result"], str)
            assert len(response["result"]) > 0

    def test_query_with_chat_history(self):
        """Test query with chat history."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = "Test response"
            mock_handler.get_llm.return_value = mock_llm
            mock_handler.get_system_prompt.return_value = "System prompt"
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            history = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
            response = pipeline.query("What is AI?", chat_history=history)

            assert "result" in response
            assert isinstance(response["result"], str)
            assert len(response["result"]) > 0

    def test_query_with_empty_chat_history(self):
        """Test query with empty chat history list."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = "Test response"
            mock_handler.get_llm.return_value = mock_llm
            mock_handler.get_system_prompt.return_value = "System prompt"
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            response = pipeline.query("What is AI?", chat_history=[])

            assert "result" in response
            assert isinstance(response["result"], str)


class TestGetAnswerWithChatHistory:
    """Test get_answer method with chat history."""

    def test_get_answer_without_chat_history(self):
        """Test get_answer without chat history."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = "Test answer"
            mock_handler.get_llm.return_value = mock_llm
            mock_handler.get_system_prompt.return_value = "System prompt"
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            answer = pipeline.get_answer("What is AI?")

            assert isinstance(answer, str)
            assert len(answer) > 0

    def test_get_answer_with_chat_history(self):
        """Test get_answer with chat history."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = "Test answer"
            mock_handler.get_llm.return_value = mock_llm
            mock_handler.get_system_prompt.return_value = "System prompt"
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            history = [{"role": "user", "content": "Previous question"}]
            answer = pipeline.get_answer("What is AI?", chat_history=history)

            assert isinstance(answer, str)
            assert len(answer) > 0


class TestGetAnswerWithSourcesChatHistory:
    """Test get_answer_with_sources method with chat history."""

    def test_get_answer_with_sources_without_chat_history(self):
        """Test get_answer_with_sources without chat history."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = "Test answer"
            mock_handler.get_llm.return_value = mock_llm
            mock_handler.get_system_prompt.return_value = "System prompt"
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            answer, sources = pipeline.get_answer_with_sources("What is AI?")

            assert isinstance(answer, str)
            assert isinstance(sources, list)

    def test_get_answer_with_sources_with_chat_history(self):
        """Test get_answer_with_sources with chat history."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = "Test answer"
            mock_handler.get_llm.return_value = mock_llm
            mock_handler.get_system_prompt.return_value = "System prompt"
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            history = [{"role": "user", "content": "Previous question"}]
            answer, sources = pipeline.get_answer_with_sources("What is AI?", chat_history=history)

            assert isinstance(answer, str)
            assert isinstance(sources, list)


class TestStreamQueryWithChatHistory:
    """Test stream_query method with chat history."""

    def test_stream_query_without_chat_history(self):
        """Test stream_query without chat history."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_llm = MagicMock()
            mock_llm.stream.return_value = iter(["Test", " response", " chunks"])
            mock_handler.get_llm.return_value = mock_llm
            mock_handler.get_system_prompt.return_value = "System prompt"
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            chunks = list(pipeline.stream_query("What is AI?"))

            assert len(chunks) > 0
            assert all(isinstance(chunk, str) for chunk in chunks)

    def test_stream_query_with_chat_history(self):
        """Test stream_query with chat history."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_llm = MagicMock()
            mock_llm.stream.return_value = iter(["Test", " response", " chunks"])
            mock_handler.get_llm.return_value = mock_llm
            mock_handler.get_system_prompt.return_value = "System prompt"
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            history = [{"role": "user", "content": "Previous question"}]
            chunks = list(pipeline.stream_query("What is AI?", chat_history=history))

            assert len(chunks) > 0
            assert all(isinstance(chunk, str) for chunk in chunks)

    def test_stream_query_error_handling(self):
        """Test stream_query error handling."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_llm = MagicMock()
            mock_llm.stream.side_effect = Exception("Stream error")
            mock_handler.get_llm.return_value = mock_llm
            mock_handler.get_system_prompt.return_value = "System prompt"
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            chunks = list(pipeline.stream_query("What is AI?"))

            # Should yield error message
            assert len(chunks) > 0
            assert "technical issue" in chunks[0].lower()


class TestStreamAnswerWithChatHistory:
    """Test stream_answer method with chat history."""

    def test_stream_answer_without_chat_history(self):
        """Test stream_answer without chat history."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_llm = MagicMock()
            mock_llm.stream.return_value = iter(["Answer", " chunks"])
            mock_handler.get_llm.return_value = mock_llm
            mock_handler.get_system_prompt.return_value = "System prompt"
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            chunks = list(pipeline.stream_answer("What is AI?"))

            assert len(chunks) > 0
            assert all(isinstance(chunk, str) for chunk in chunks)

    def test_stream_answer_with_chat_history(self):
        """Test stream_answer with chat history."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_llm = MagicMock()
            mock_llm.stream.return_value = iter(["Answer", " chunks"])
            mock_handler.get_llm.return_value = mock_llm
            mock_handler.get_system_prompt.return_value = "System prompt"
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            history = [{"role": "user", "content": "Previous question"}]
            chunks = list(pipeline.stream_answer("What is AI?", chat_history=history))

            assert len(chunks) > 0
            assert all(isinstance(chunk, str) for chunk in chunks)


class TestChatHistoryInPrompt:
    """Test that chat history is properly included in prompts."""

    def test_prompt_template_includes_chat_history_placeholder(self):
        """Test that prompt template includes chat_history placeholder."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_handler.get_llm.return_value = MagicMock()
            mock_handler.get_system_prompt.return_value = "System prompt"
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            template_str = pipeline.prompt_template.messages[0].prompt.template

            # Should contain chat_history placeholder
            assert "{chat_history}" in template_str

    def test_chat_history_position_in_template(self):
        """Test that chat history appears in correct position in template."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_handler.get_llm.return_value = MagicMock()
            mock_handler.get_system_prompt.return_value = "System prompt"
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            template_str = pipeline.prompt_template.messages[0].prompt.template

            # Chat history should come after system prompt but before main doc and context
            chat_history_pos = template_str.find("{chat_history}")
            main_doc_pos = template_str.find("{main_document_section}")
            context_pos = template_str.find("{context}")
            question_pos = template_str.find("{question}")

            # Verify order: system -> chat_history -> main_doc -> context -> question
            assert chat_history_pos < main_doc_pos
            assert chat_history_pos < context_pos
            assert chat_history_pos < question_pos


class TestChatHistoryIntegration:
    """Integration tests for chat history feature."""

    def test_multiple_queries_with_history(self):
        """Test multiple queries building up chat history."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = "Test response"
            mock_handler.get_llm.return_value = mock_llm
            mock_handler.get_system_prompt.return_value = "System prompt"
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()

            # First query - no history
            response1 = pipeline.query("What is AI?")
            assert "result" in response1

            # Second query - with history from first query
            history = [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": response1["result"]},
            ]
            response2 = pipeline.query("Tell me more", chat_history=history)
            assert "result" in response2

            # Third query - with extended history
            extended_history = history + [
                {"role": "user", "content": "Tell me more"},
                {"role": "assistant", "content": response2["result"]},
            ]
            response3 = pipeline.query("What are the applications?", chat_history=extended_history)
            assert "result" in response3

    def test_chat_history_with_main_document(self):
        """Test chat history works together with main document."""
        with patch("src.rag_pipeline.get_llm_handler") as mock_llm_handler:
            mock_handler = MagicMock()
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = "Test response"
            mock_handler.get_llm.return_value = mock_llm
            mock_handler.get_system_prompt.return_value = "System prompt"
            mock_llm_handler.return_value = mock_handler

            pipeline = RAGPipeline()
            history = [
                {"role": "user", "content": "What is your background?"},
                {"role": "assistant", "content": "I have experience in AI."},
            ]

            response = pipeline.query("Tell me more about your projects", chat_history=history)
            assert "result" in response
            assert isinstance(response["result"], str)


class TestAppChatHistoryHelpers:
    """Test chat history helper functions from app.py."""

    def test_extract_chat_history_empty(self):
        """Test extract_chat_history with empty messages."""
        try:
            from app import extract_chat_history
        except ImportError:
            pytest.skip("Cannot import app module (streamlit may not be available)")

        result = extract_chat_history([])
        assert result == []

        result = extract_chat_history(None)
        assert result == []

    def test_extract_chat_history_single_message(self):
        """Test extract_chat_history with single message."""
        try:
            from app import extract_chat_history
        except ImportError:
            pytest.skip("Cannot import app module (streamlit may not be available)")

        messages = [{"role": "user", "content": "Hello"}]
        result = extract_chat_history(messages, exclude_last=True)
        assert result == []

        result = extract_chat_history(messages, exclude_last=False)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_extract_chat_history_multiple_messages(self):
        """Test extract_chat_history with multiple messages."""
        try:
            from app import extract_chat_history
        except ImportError:
            pytest.skip("Cannot import app module (streamlit may not be available)")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ]

        # Exclude last message
        result = extract_chat_history(messages, exclude_last=True)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

        # Include last message
        result = extract_chat_history(messages, exclude_last=False)
        assert len(result) == 3

    def test_extract_chat_history_filters_invalid_roles(self):
        """Test extract_chat_history filters out invalid roles."""
        try:
            from app import extract_chat_history
        except ImportError:
            pytest.skip("Cannot import app module (streamlit may not be available)")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "System message"},
            {"role": "assistant", "content": "Hi!"},
        ]

        result = extract_chat_history(messages, exclude_last=False)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_extract_chat_history_filters_empty_content(self):
        """Test extract_chat_history filters out empty content."""
        try:
            from app import extract_chat_history
        except ImportError:
            pytest.skip("Cannot import app module (streamlit may not be available)")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "How are you?"},
        ]

        result = extract_chat_history(messages, exclude_last=False)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "user"

    def test_truncate_chat_history_empty(self):
        """Test truncate_chat_history with empty history."""
        try:
            from app import truncate_chat_history
        except ImportError:
            pytest.skip("Cannot import app module (streamlit may not be available)")

        result = truncate_chat_history([])
        assert result == []

        result = truncate_chat_history([], max_turns=5)
        assert result == []

    def test_truncate_chat_history_no_limits(self):
        """Test truncate_chat_history with no limits."""
        try:
            from app import truncate_chat_history
        except ImportError:
            pytest.skip("Cannot import app module (streamlit may not be available)")

        history = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer 2"},
        ]

        result = truncate_chat_history(history)
        assert len(result) == 4
        assert result == history

    def test_truncate_chat_history_by_turns(self):
        """Test truncate_chat_history by turn count."""
        try:
            from app import truncate_chat_history
        except ImportError:
            pytest.skip("Cannot import app module (streamlit may not be available)")

        history = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer 2"},
            {"role": "user", "content": "Question 3"},
            {"role": "assistant", "content": "Answer 3"},
        ]

        # Keep only 2 most recent turns (4 messages)
        result = truncate_chat_history(history, max_turns=2)
        assert len(result) == 4
        assert result[0]["content"] == "Question 2"
        assert result[-1]["content"] == "Answer 3"

    def test_truncate_chat_history_by_tokens(self):
        """Test truncate_chat_history by token limit."""
        try:
            from app import truncate_chat_history
        except ImportError:
            pytest.skip("Cannot import app module (streamlit may not be available)")

        # Create history with varying message lengths
        history = [
            {"role": "user", "content": "Short question"},
            {"role": "assistant", "content": "Short answer"},
            {"role": "user", "content": "This is a much longer question that will use more tokens"},
            {
                "role": "assistant",
                "content": "This is a much longer answer that will also use more tokens",
            },
        ]

        # Truncate to a small token limit (should keep only recent messages that fit)
        result = truncate_chat_history(history, max_tokens=50)
        assert len(result) <= len(history)
        # Should keep most recent messages that fit within token limit

    def test_truncate_chat_history_by_turns_and_tokens(self):
        """Test truncate_chat_history with both turn and token limits."""
        try:
            from app import truncate_chat_history
        except ImportError:
            pytest.skip("Cannot import app module (streamlit may not be available)")

        history = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer 2"},
        ]

        # First apply turn limit, then token limit
        result = truncate_chat_history(history, max_turns=1, max_tokens=1000)
        assert len(result) <= 2  # 1 turn = 2 messages

    def test_truncate_chat_history_zero_limits(self):
        """Test truncate_chat_history with zero limits."""
        try:
            from app import truncate_chat_history
        except ImportError:
            pytest.skip("Cannot import app module (streamlit may not be available)")

        history = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
        ]

        # Zero limits don't truncate (function only truncates if > 0)
        # So it returns the original history unchanged
        result = truncate_chat_history(history, max_turns=0)
        assert result == history  # Returns original when limit is 0

        result = truncate_chat_history(history, max_tokens=0)
        assert result == history  # Returns original when limit is 0
