# SLM Profile RAG

A RAG (Retrieval-Augmented Generation) chatbot for recruiters to Q&A on your profile using Small Language Models (SLM) served via Ollama.

## Features

- 🤖 **RAG-powered Q&A**: Answer questions about your profile using retrieval-augmented generation
- 📄 **Multi-format Support**: Process PDF, HTML, TXT, and Markdown documents
- 🔧 **Easy Configuration**: YAML-based settings with environment variable support
- 🎨 **Simple UI**: Clean Gradio interface for easy interaction
- 🚀 **Ollama Integration**: Uses local SLM models for privacy and control
- 📦 **Modern Python Stack**: Built with UV, pyproject.toml, hatchling, and ruff

## Prerequisites

- Python 3.9 or higher
- [UV](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.ai/) installed and running locally

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tin-Hoang/slm-profile-rag.git
   cd slm-profile-rag
   ```

2. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Create a virtual environment and install dependencies**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"
   ```

4. **Install and setup Ollama**:
   - Download from [ollama.ai](https://ollama.ai/)
   - Pull a model (e.g., llama2):
     ```bash
     ollama pull llama2
     ```

## Configuration

1. **Create environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env`** with your settings:
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama2
   PROFILE_DOCS_PATH=./profile_docs
   ```

3. **Customize `config.yaml`** for fine-tuning behavior (optional)

## Adding Your Profile Documents

Place your profile documents in the `profile_docs/` directory:

```bash
profile_docs/
├── resume.pdf
├── linkedin_profile.html
├── project_reports/
│   ├── project1.pdf
│   └── project2.md
└── cover_letter.txt
```

Supported formats:
- PDF (`.pdf`)
- HTML (`.html`)
- Text (`.txt`)
- Markdown (`.md`)

## Usage

### Running Locally

Start the chatbot application:

```bash
python -m slm_profile_rag.app
```

Or using the module directly:

```bash
python src/slm_profile_rag/app.py
```

The UI will be available at `http://localhost:7860`

### Force Reload Documents

To reload all documents into the vector store:

```bash
python -c "from slm_profile_rag.app import launch_ui; launch_ui(force_reload=True)"
```

## Development

### Code Formatting and Linting

Format code with ruff:
```bash
ruff format .
```

Lint code:
```bash
ruff check .
```

Auto-fix linting issues:
```bash
ruff check --fix .
```

### Running Tests

```bash
pytest
```

## Deployment to Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)

2. Add the following files to your Space:
   - All project files
   - Create `app.py` in the root:
     ```python
     from slm_profile_rag.app import launch_ui
     
     if __name__ == "__main__":
         launch_ui()
     ```

3. Configure Space secrets for environment variables

4. The Space will automatically build and deploy

## Project Structure

```
slm-profile-rag/
├── src/
│   └── slm_profile_rag/
│       ├── __init__.py
│       ├── app.py                 # Gradio UI
│       ├── config.py              # Configuration loader
│       ├── document_processor.py  # Document loading & processing
│       └── rag_pipeline.py        # RAG implementation
├── profile_docs/                  # Your profile documents
├── config.yaml                    # Application configuration
├── pyproject.toml                 # Project metadata & dependencies
├── .env.example                   # Environment template
└── README.md
```

## How It Works

1. **Document Loading**: Your profile documents are loaded from `profile_docs/`
2. **Text Splitting**: Documents are split into chunks for better retrieval
3. **Embeddings**: Text chunks are converted to vector embeddings using Ollama
4. **Vector Store**: Embeddings are stored in ChromaDB for efficient retrieval
5. **Query Processing**: User questions are embedded and similar chunks are retrieved
6. **Answer Generation**: Retrieved context + question are sent to the SLM for answer generation

## Customization

### Changing the Model

Edit `.env` or `config.yaml`:
```yaml
model:
  name: "mistral"  # or "llama2", "codellama", etc.
  temperature: 0.7
```

### Adjusting Retrieval

Edit `config.yaml`:
```yaml
retrieval:
  top_k: 5  # Number of chunks to retrieve
  search_type: "similarity"
```

### Custom Prompts

Edit the prompt templates in `config.yaml`:
```yaml
prompts:
  system_prompt: |
    Your custom system prompt here...
```

## Troubleshooting

**Ollama connection errors**:
- Ensure Ollama is running: `ollama serve`
- Check the base URL in `.env` matches your Ollama server

**No documents found**:
- Verify documents are in `profile_docs/` directory
- Check file formats are supported
- Run with `force_reload=True` to reprocess documents

**Memory issues**:
- Reduce `chunk_size` in config.yaml
- Use a smaller model (e.g., `llama2:7b` instead of `llama2:13b`)

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
