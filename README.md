# 🤖 SLM Profile RAG Chatbot

A RAG (Retrieval Augmented Generation) chatbot that answers questions about your professional profile using your resume, project reports, and other documents.

## ✨ Features

- 📄 **Multi-format Support**: Process PDF, Word, HTML, and text documents
- 🧠 **RAG Pipeline**: Semantic search with vector database (ChromaDB)
- 🦙 **Ollama Integration**: Run small language models locally
- 🎨 **Clean UI**: Streamlit-based interface
- ⚙️ **Highly Configurable**: YAML-based settings for easy customization
- 🚀 **HuggingFace Spaces Ready**: Deploy with one click
- ✨ **Smart Response Enhancement**: Automatically removes negative language and adds professional, recruiter-friendly tone

## 🏗️ Architecture

<p align="center">
  <img src="docs/arch_system_design.png" alt="System Architecture Diagram" width="800" />
</p>
<p align="center"><em>Overall system design showing the RAG pipeline and component interactions</em></p>

Detailed architecture can be found in [ARCHITECTURE.md](ARCHITECTURE.md).

## 🛠️ Tech Stack

- **Python 3.10+** with UV package manager
- **LangChain** for RAG pipeline
- **ChromaDB** for vector storage
- **Ollama** for local LLM serving
- **Streamlit** for web interface
- **sentence-transformers** for embeddings
- **Ruff** for linting/formatting

## 📦 Installation

### Prerequisites

1. **Python 3.10+**
2. **UV Package Manager**: Install via:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **Ollama**: Install from [ollama.ai](https://ollama.ai)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd slm-profile-rag
   ```

2. **Install dependencies with UV**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.template .env
   # Edit .env with your settings
   ```

4. **Configure the chatbot**:
   - Edit `config.yaml` to set your name, title, and preferences
   - Adjust model settings, chunking parameters, and system prompt

5. **Add your documents**:
   ```bash
   # Place your PDFs, Word docs, HTML files in:
   mkdir -p data/documents
   # Copy your resume, project reports, LinkedIn profile, etc.
   ```

6. **Pull Ollama model**:
   ```bash
   ollama pull llama3.2:3b
   # Or other small models: phi3:mini, gemma2:2b
   ```

## 🚀 Usage

### Local Development

1. **Process documents and build vector database**:
   ```bash
   python -m src.build_vectorstore
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Open browser**: Navigate to `http://localhost:8501`

### Linting & Formatting

```bash
# Check code
uv run ruff check .

# Format code
uv run ruff format .

# Check and fix
uv run ruff check --fix .
```

## 🌐 Deployment to Hugging Face Spaces

### Option 1: Direct Upload

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select "Streamlit" as SDK
3. Upload all files from this repository
4. Add your documents to `data/documents/`
5. The Space will automatically build and deploy

### Option 2: Git Integration

1. Create a new Space and connect to Git
2. Push this repository:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
   git push hf main
   ```

### Important Notes for HF Spaces

- **Ollama in HF Spaces**: You'll need a persistent Ollama server or use the Space's GPU
- **Vector DB**: Pre-build your ChromaDB locally and include it (or rebuild on startup)
- **Memory**: Small models (3B-7B params) work best on free tier
- **Secrets**: Add API keys in Space settings if using alternative LLM providers

## ⚙️ Configuration

### `config.yaml` - Main Settings

```yaml
profile:
  name: "Your Name"  # ← Change this!
  title: "Your Title"

llm:
  model: "llama3.2:3b"  # Choose your model
  temperature: 0.7

document_processing:
  chunk_size: 1000
  chunk_overlap: 200
```

### `.env` - Environment Variables

```bash
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PERSIST_DIR=./chroma_db
```

## 📚 Project Structure

```
slm-profile-rag/
├── app.py                          # Streamlit app entry point
├── pyproject.toml                  # UV/pip dependencies & config
├── config.yaml                     # RAG & LLM settings
├── .env.example                    # Environment variables template
├── README.md
├── data/
│   └── documents/                  # Your profile documents
├── src/
│   ├── __init__.py
│   ├── document_processor.py       # Load & chunk documents
│   ├── vectorstore.py              # ChromaDB operations
│   ├── llm_handler.py              # Ollama/LLM interface
│   ├── rag_pipeline.py             # RAG chain logic
│   ├── response_enhancer.py        # Response post-processing (NEW!)
│   ├── config_loader.py            # Load config.yaml & .env
│   └── build_vectorstore.py        # CLI to build vector DB
├── chroma_db/                      # Vector database (auto-generated)
└── tests/                          # Unit tests
```

## 🎯 Recommended Models for HF Spaces

| Model | Size | Speed | Quality | HF Spaces Tier |
|-------|------|-------|---------|----------------|
| `llama3.2:3b` | 3B | Fast | Good | Free ✅ |
| `phi3:mini` | 3.8B | Fast | Good | Free ✅ |
| `gemma2:2b` | 2B | Very Fast | Decent | Free ✅ |
| `llama3.1:8b` | 8B | Medium | Great | Upgraded GPU |

## 🔧 Troubleshooting

### Ollama Connection Issues
```bash
# Check Ollama is running
ollama list

# Start Ollama service
ollama serve
```

### ChromaDB Errors
```bash
# Rebuild vector database
rm -rf chroma_db/
python -m src.build_vectorstore
```

### HuggingFace Spaces Issues
- Check logs in the Space's "Logs" tab
- Ensure `requirements.txt` is generated: `uv pip compile pyproject.toml -o requirements.txt`
- Verify GPU/CPU settings match your model size

## 📄 License

MIT License - see LICENSE file

---

**Note**: Remember to update `config.yaml` with your personal information before deploying!

