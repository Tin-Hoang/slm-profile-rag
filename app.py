"""Streamlit app for LLM Profile Chatbot."""

import logging
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import get_config
from src.llm_handler import (
    detect_default_provider,
    get_available_groq_models,
    get_available_ollama_models,
    get_default_groq_model,
    get_default_ollama_model,
    get_llm_handler,
)
from src.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None

    if "config" not in st.session_state:
        st.session_state.config = None

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    # LLM Provider settings
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = detect_default_provider()

    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = ""

    if "groq_model" not in st.session_state:
        st.session_state.groq_model = get_default_groq_model()

    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = get_default_ollama_model()


def load_config():
    """Load configuration."""
    if st.session_state.config is None:
        try:
            st.session_state.config = get_config()
            logger.info("Configuration loaded successfully")
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
            st.stop()


def load_rag_pipeline(force_reload: bool = False):
    """Load RAG pipeline with current provider settings.

    Args:
        force_reload: Force reload even if pipeline exists
    """
    if st.session_state.rag_pipeline is None or force_reload:
        with st.spinner("Loading RAG pipeline... This may take a moment."):
            try:
                # Get provider settings from session state
                provider = st.session_state.llm_provider
                api_key = st.session_state.groq_api_key or None
                model = None

                if provider == "groq":
                    model = st.session_state.groq_model
                elif provider == "ollama":
                    model = st.session_state.ollama_model

                # Create LLM handler with overrides
                llm_handler = get_llm_handler(
                    provider_override=provider,
                    api_key_override=api_key,
                    model_override=model,
                )

                st.session_state.rag_pipeline = RAGPipeline(llm_handler=llm_handler)
                logger.info(f"RAG pipeline loaded with provider: {provider}")

            except FileNotFoundError:
                st.error(
                    "❌ Vector store not found! Please build it first:\n\n"
                    "```bash\n"
                    "python -m src.build_vectorstore\n"
                    "```"
                )
                st.info(
                    "Make sure you have:\n"
                    "1. Added your documents to `data/documents/`\n"
                    "2. Run the build command above\n"
                    "3. Started Ollama and pulled the model"
                )
                st.stop()

            except Exception as e:
                error_str = str(e).lower()

                # Check if it's a Groq API error - try fallback to Ollama
                if st.session_state.llm_provider == "groq" and (
                    "401" in error_str
                    or "invalid" in error_str
                    or "api_key" in error_str
                    or "429" in error_str
                    or "rate" in error_str
                ):
                    logger.warning(f"Groq API error: {e}. Falling back to Ollama.")

                    # Show warning to user
                    st.warning(
                        "⚠️ **Groq API Error** - Falling back to Ollama\n\n"
                        "Possible causes:\n"
                        "- Invalid API key\n"
                        "- Rate limit exceeded (free tier cap)\n"
                        "- API key expired\n\n"
                        "**Get a new key at:** https://console.groq.com\n\n"
                        "Using local Ollama instead..."
                    )

                    # Fallback to Ollama
                    try:
                        st.session_state.llm_provider = "ollama"
                        ollama_model = st.session_state.ollama_model or get_default_ollama_model()

                        llm_handler = get_llm_handler(
                            provider_override="ollama",
                            model_override=ollama_model,
                        )

                        st.session_state.rag_pipeline = RAGPipeline(llm_handler=llm_handler)
                        logger.info("Fallback to Ollama successful")
                        st.success(f"✅ Now using Ollama with model: {ollama_model}")
                        return

                    except Exception as fallback_error:
                        st.error(f"Fallback to Ollama also failed: {fallback_error}")
                        st.info(
                            "Make sure Ollama is running:\n"
                            "- Start Ollama: `ollama serve`\n"
                            "- Pull a model: `ollama pull llama3.2:3b`"
                        )
                        logger.error(f"Ollama fallback error: {fallback_error}")
                        st.stop()

                # Not a Groq error or other error
                st.error(f"Error loading RAG pipeline: {e}")
                if st.session_state.llm_provider == "groq":
                    st.info(
                        "Groq issues:\n"
                        "- Check your API key is valid\n"
                        "- Get a free key at https://console.groq.com"
                    )
                else:
                    st.info(
                        "Ollama issues:\n"
                        "- Ollama not running: `ollama serve`\n"
                        "- Model not pulled: `ollama pull llama3.2:3b`"
                    )
                logger.error(f"RAG pipeline error: {e}", exc_info=True)
                st.stop()


def configure_page():
    """Configure Streamlit page settings."""
    config = st.session_state.config

    page_title = config.get("ui.page_title", "💬 Profile Q&A Chatbot")
    page_icon = config.get("ui.page_icon", "🤖")
    layout = config.get("ui.layout", "centered")

    st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)


def display_header():
    """Display app header."""
    config = st.session_state.config

    profile_name = config.get("profile.name", "Candidate")
    profile_title = config.get("profile.title", "Professional")
    greeting = config.format_template(config.get("profile.greeting", ""))

    st.title(f"💬 Chat with {profile_name}'s Profile")
    st.markdown(f"**{profile_title}**")

    if greeting:
        st.info(greeting)

    st.divider()


def display_sidebar():
    """Display sidebar with information and controls."""
    config = st.session_state.config

    with st.sidebar:
        st.header("ℹ️ About")

        profile_name = config.get("profile.name", "the candidate")
        st.markdown(
            f"This chatbot can answer questions about **{profile_name}'s** "
            "professional background, skills, experience, and projects."
        )

        st.divider()

        # Example questions
        st.subheader("💡 Example Questions")
        example_questions = config.get("ui.example_questions", [])

        for question in example_questions:
            formatted_q = config.format_template(question)
            if st.button(formatted_q, key=f"example_{hash(formatted_q)}"):
                st.session_state.pending_question = formatted_q
                st.rerun()

        st.divider()

        # LLM Provider Selection
        st.subheader("🤖 LLM Provider")

        # Provider selector
        providers = ["ollama", "groq"]
        current_provider_idx = (
            providers.index(st.session_state.llm_provider)
            if st.session_state.llm_provider in providers
            else 0
        )

        new_provider = st.radio(
            "Select Provider",
            providers,
            index=current_provider_idx,
            format_func=lambda x: "🦙 Ollama (Local)" if x == "ollama" else "⚡ Groq (Cloud)",
            key="provider_radio",
            help="Ollama runs locally. Groq requires an API key but is faster.",
        )

        # Groq-specific settings
        if new_provider == "groq":
            st.markdown("---")
            groq_key = st.text_input(
                "Groq API Key",
                value=st.session_state.groq_api_key,
                type="password",
                placeholder="Enter your Groq API key",
                help="Get a free key at https://console.groq.com",
                key="groq_key_input",
            )

            groq_models = get_available_groq_models()
            current_model_idx = (
                groq_models.index(st.session_state.groq_model)
                if st.session_state.groq_model in groq_models
                else 0
            )

            new_model = st.selectbox(
                "Groq Model",
                groq_models,
                index=current_model_idx,
                key="groq_model_select",
            )

            # Check if settings changed
            settings_changed = (
                new_provider != st.session_state.llm_provider
                or groq_key != st.session_state.groq_api_key
                or new_model != st.session_state.groq_model
            )

            if settings_changed and st.button("🔄 Apply Changes", type="primary"):
                st.session_state.llm_provider = new_provider
                st.session_state.groq_api_key = groq_key
                st.session_state.groq_model = new_model
                st.session_state.rag_pipeline = None  # Force reload
                st.rerun()
        else:
            # Ollama selected - show model selector
            st.markdown("---")
            ollama_models = get_available_ollama_models()
            current_ollama_idx = (
                ollama_models.index(st.session_state.ollama_model)
                if st.session_state.ollama_model in ollama_models
                else 0
            )

            new_ollama_model = st.selectbox(
                "Ollama Model",
                ollama_models,
                index=current_ollama_idx,
                key="ollama_model_select",
                help="Make sure the model is pulled: ollama pull <model>",
            )

            settings_changed = (
                new_provider != st.session_state.llm_provider
                or new_ollama_model != st.session_state.ollama_model
            )

            if settings_changed and st.button("🔄 Apply Changes", type="primary"):
                st.session_state.llm_provider = new_provider
                st.session_state.ollama_model = new_ollama_model
                st.session_state.rag_pipeline = None
                st.rerun()

        # Show current active settings
        st.markdown("---")
        st.caption(f"**Active:** {st.session_state.llm_provider.upper()}")
        if st.session_state.llm_provider == "groq":
            st.caption(f"**Model:** {st.session_state.groq_model}")
        else:
            st.caption(f"**Model:** {st.session_state.ollama_model}")

        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        # Settings
        with st.expander("⚙️ Advanced Settings"):
            st.markdown("**Vector Store**")
            st.markdown(f"- Collection: {config.get('vectorstore.collection_name', 'N/A')}")
            st.markdown(f"- Top K: {config.get('vectorstore.search_kwargs.k', 'N/A')}")

            st.markdown("**Document Processing**")
            st.markdown(f"- Chunk Size: {config.get('document_processing.chunk_size', 'N/A')}")
            st.markdown(f"- Overlap: {config.get('document_processing.chunk_overlap', 'N/A')}")


def display_chat_messages():
    """Display chat messages."""
    config = st.session_state.config
    show_timestamps = config.get("ui.show_timestamps", True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display timestamp if enabled
            if show_timestamps and "timestamp" in message:
                st.caption(f"_{message['timestamp']}_")

            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    st.markdown(message["sources"])


def process_user_input(prompt: str):
    """Process user input and generate response.

    Args:
        prompt: User's question
    """
    config = st.session_state.config
    rag_pipeline = st.session_state.rag_pipeline

    # Add user message
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        if config.get("ui.show_timestamps", True):
            st.caption(f"_{timestamp}_")

    # Generate streaming response
    with st.chat_message("assistant"):
        try:
            # Create placeholder for streaming content
            response_placeholder = st.empty()

            # Stream the response token by token with spinner
            streamed_content = ""
            with st.spinner("Thinking (this may take a few minutes on a free-tier HF space)..."):
                for chunk in rag_pipeline.stream_answer(prompt):
                    streamed_content += chunk
                    response_placeholder.markdown(streamed_content + "▌")

            # Remove cursor and show final streamed content
            response_placeholder.markdown(streamed_content)

            # Enhance the response for better tone (post-stream enhancement)
            answer = streamed_content
            if config.get("rag.enhance_responses", True):
                from src.response_enhancer import get_response_enhancer

                enhancer = get_response_enhancer()
                enhanced_answer = enhancer.enhance_with_context(streamed_content, prompt)
                if enhanced_answer != streamed_content:
                    answer = enhanced_answer
                    # Update display with enhanced version
                    response_placeholder.markdown(answer)
                    logger.debug("Response enhanced for better tone")

            # Get source documents (after streaming completes)
            sources = rag_pipeline.get_source_documents(prompt)
            sources_text = ""
            if config.get("rag.include_sources", True) and sources:
                sources_text = rag_pipeline.format_sources(sources)
                with st.expander("📚 Sources"):
                    st.markdown(sources_text)

            # Add to message history
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "timestamp": timestamp,
                    "sources": sources_text,
                }
            )

        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {e}"
            st.error(error_msg)
            logger.error(f"Error generating response: {e}", exc_info=True)

            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg, "timestamp": timestamp}
            )


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()

    # Load configuration
    load_config()

    # Configure page
    configure_page()

    # Load RAG pipeline
    load_rag_pipeline()

    # Display UI
    display_header()
    display_sidebar()

    # Display chat messages
    display_chat_messages()

    # Process pending question from example button click
    if st.session_state.pending_question:
        pending = st.session_state.pending_question
        st.session_state.pending_question = None
        process_user_input(pending)
        st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask me anything about the profile..."):
        process_user_input(prompt)

    # Limit chat history
    max_history = st.session_state.config.get("ui.max_chat_history", 50)
    if len(st.session_state.messages) > max_history:
        st.session_state.messages = st.session_state.messages[-max_history:]


if __name__ == "__main__":
    main()
