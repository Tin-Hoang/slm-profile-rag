"""Gradio UI for the RAG chatbot."""

import gradio as gr

from slm_profile_rag.config import Config
from slm_profile_rag.rag_pipeline import RAGPipeline


def create_ui(config: Config, rag_pipeline: RAGPipeline):
    """Create Gradio UI for the chatbot.

    Args:
        config: Configuration object.
        rag_pipeline: Initialized RAG pipeline.

    Returns:
        Gradio Blocks interface.
    """

    def chat(message: str, history: list) -> str:
        """Process chat message.

        Args:
            message: User message.
            history: Chat history.

        Returns:
            Bot response.
        """
        try:
            result = rag_pipeline.query(message)
            answer = result["answer"]

            # Optionally include source information
            sources = result.get("source_documents", [])
            if sources and config.debug:
                answer += "\n\n---\nSources:\n"
                for i, doc in enumerate(sources[:2], 1):
                    source = doc.metadata.get("source", "Unknown")
                    answer += f"{i}. {source}\n"

            return answer
        except Exception as e:
            return f"Error: {str(e)}"

    # Get UI configuration
    title = config.get("ui.title", "Profile Q&A Assistant")
    description = config.get(
        "ui.description", "Ask questions about my background, experience, and skills"
    )
    examples = config.get("ui.examples", [])
    theme = config.get("ui.theme", "default")

    # Create Gradio interface
    with gr.Blocks(title=title, theme=theme) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)

        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(
            label="Ask a question",
            placeholder="Type your question here...",
            lines=2,
        )

        with gr.Row():
            submit = gr.Button("Submit", variant="primary")
            clear = gr.Button("Clear")

        if examples:
            gr.Examples(examples=examples, inputs=msg)

        # Event handlers
        def user_message(user_msg, history):
            return "", history + [[user_msg, None]]

        def bot_response(history):
            user_msg = history[-1][0]
            bot_msg = chat(user_msg, history)
            history[-1][1] = bot_msg
            return history

        msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, chatbot, chatbot
        )
        submit.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    return demo


def launch_ui(
    config_path: str = "config.yaml",
    share: bool = False,
    force_reload: bool = False,
    server_name: str = "127.0.0.1",
):
    """Launch the Gradio UI.

    Args:
        config_path: Path to configuration file.
        share: Whether to create a public share link.
        force_reload: Force reload documents.
        server_name: Server address to bind to. Use "0.0.0.0" for all interfaces.
    """
    config = Config(config_path)
    rag_pipeline = RAGPipeline(config)
    rag_pipeline.setup(force_reload=force_reload)

    demo = create_ui(config, rag_pipeline)
    share_setting = config.get("ui.share", share)

    # Use 0.0.0.0 for Docker/Hugging Face Spaces, 127.0.0.1 for local dev
    import os

    if os.getenv("GRADIO_SERVER_NAME"):
        server_name = os.getenv("GRADIO_SERVER_NAME")

    demo.launch(share=share_setting, server_name=server_name)


if __name__ == "__main__":
    launch_ui()
