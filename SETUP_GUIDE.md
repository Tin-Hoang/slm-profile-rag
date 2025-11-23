# 🚀 Setup Guide - LLM Profile Chatbot

This guide will walk you through setting up your personal profile chatbot from scratch.

## 📋 Prerequisites

### 1. Install Python 3.10+

```bash
# Check your Python version
python --version  # Should be 3.10 or higher
```

### 2. Install UV Package Manager

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

### 3. Install Ollama

Download and install from [ollama.ai](https://ollama.ai)

```bash
# Verify installation
ollama --version
```

## 🛠️ Local Setup

### Step 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/Tin-Hoang/slm-profile-rag.git
cd slm-profile-rag

# Create virtual environment with UV
uv venv

# Activate virtual environment
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### Step 2: Configuration

#### 2.1 Environment Variables

```bash
# Copy environment template
cp env.template .env

# Edit .env file (optional, defaults work for local development)
# Update OLLAMA_BASE_URL if needed
```

#### 2.2 Configure Your Profile

Edit `config.yaml`:

```yaml
profile:
  name: "Your Full Name"  # ⚠️ CHANGE THIS
  title: "Your Professional Title"  # ⚠️ CHANGE THIS
  greeting: "Hi! I'm a chatbot trained on {name}'s professional background..."

llm:
  model: "llama3.2:3b"  # Choose based on your hardware
  temperature: 0.7

# Adjust other settings as needed
```

### Step 3: Pull LLM Model

```bash
# Pull the model specified in config.yaml
ollama pull llama3.2:3b

# Or try other models:
# ollama pull phi3:mini       # Microsoft Phi-3 (3.8B)
# ollama pull gemma2:2b       # Google Gemma 2 (2B)
# ollama pull llama3.1:8b     # Larger, better quality (if you have GPU)
```

### Step 4: Add Your Documents

```bash
# Navigate to documents directory
cd data/documents

# Copy your profile documents here
# Supported formats: PDF, DOCX, HTML, TXT, MD
```

**Recommended Documents:**
- ✅ Resume/CV (PDF or DOCX)
- ✅ LinkedIn profile (export as PDF)
- ✅ Project reports and case studies
- ✅ Portfolio descriptions
- ✅ Publications, certifications
- ✅ Cover letters, personal statements

**Tips:**
- Use descriptive filenames
- Ensure documents are well-structured with headings
- Include detailed information (the more context, the better!)
- Remove the sample `SAMPLE_README.md` file

### Step 5: Build Vector Store

```bash
# Make sure you're in the project root
cd ../..  # if you're still in data/documents

# Build the vector database
python -m src.build_vectorstore

# Output should show:
# - Processing documents from: ./data/documents
# - Successfully processed X document chunks
# - ✅ Vector store built successfully!
```

**Troubleshooting:**
- If no documents found: Check `data/documents/` directory
- If import errors: Ensure virtual environment is activated
- If memory issues: Reduce chunk_size in `config.yaml`

### Step 6: Start Ollama (if not running)

```bash
# Start Ollama server
ollama serve
```

Keep this terminal running, or run it as a background service.

### Step 7: Run the Application

```bash
# In a new terminal (with virtual environment activated)
streamlit run app.py

# The app will open at: http://localhost:8501
```

## 🎨 Customization

### Change UI Theme

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"  # Change to your color
backgroundColor = "#FFFFFF"
```

### Adjust RAG Settings

Edit `config.yaml`:

```yaml
document_processing:
  chunk_size: 1000      # Increase for more context per chunk
  chunk_overlap: 200    # Increase for better continuity

vectorstore:
  search_kwargs:
    k: 4                # Number of chunks to retrieve (increase for more context)

llm:
  temperature: 0.7      # Lower (0.3) for factual, higher (0.9) for creative
  max_tokens: 512       # Maximum response length
```

### Add Example Questions

Edit `config.yaml`:

```yaml
ui:
  example_questions:
    - "What is {name}'s background?"
    - "What programming languages does {name} know?"
    - "Tell me about {name}'s biggest project"
    # Add your own questions
```

### Customize System Prompt

Edit `config.yaml`:

```yaml
llm:
  system_prompt: |
    You are an AI assistant representing {name}.
    [Customize your instructions here]
```

## 🌐 Deployment to Hugging Face Spaces

### Option 1: Direct Upload

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose "Streamlit" SDK
4. Upload all project files
5. Add your documents to `data/documents/`
6. Space will automatically build and deploy

### Option 2: Git Integration

```bash
# Initialize git (if not already)
git init
git add .
git commit -m "Initial commit"

# Add Hugging Face as remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME

# Push to deploy
git push hf main
```

### Important for HF Spaces

1. **Pre-build Vector Store**: Build locally and include `chroma_db/` in git

```bash
# Build locally
python -m src.build_vectorstore

# Remove chroma_db from .gitignore temporarily
# Commit and push
git add chroma_db/
git commit -m "Add pre-built vector store"
git push hf main
```

2. **Model Size**: Use small models (3B-7B params) for free tier
   - Recommended: `llama3.2:3b`, `phi3:mini`, `gemma2:2b`

3. **Persistent Storage**: Enable in Space settings for ChromaDB

4. **Secrets**: Add API keys in Space Settings → Repository secrets

## 🧪 Testing

```bash
# Install dev dependencies
uv pip install pytest pytest-cov ruff

# Run tests
pytest

# Run linter
ruff check .

# Format code
ruff format .
```

## 🔧 Common Issues

### Issue: "Ollama connection failed"

**Solution:**
```bash
# Make sure Ollama is running
ollama serve

# Check if model is pulled
ollama list

# Pull model if missing
ollama pull llama3.2:3b
```

### Issue: "Vector store not found"

**Solution:**
```bash
# Rebuild vector store
python -m src.build_vectorstore --force-rebuild
```

### Issue: "No documents found"

**Solution:**
- Check `data/documents/` has files
- Verify file extensions are supported (.pdf, .docx, .html, .txt, .md)
- Check file permissions

### Issue: "Out of memory"

**Solution:**
```yaml
# Reduce chunk size in config.yaml
document_processing:
  chunk_size: 500  # Reduced from 1000

# Or use a smaller model
llm:
  model: "gemma2:2b"
```

### Issue: "Slow response times"

**Solution:**
- Use smaller model: `gemma2:2b` or `phi3:mini`
- Reduce retrieval chunks: `k: 2` instead of `k: 4`
- Lower temperature: `temperature: 0.5`
- Enable GPU for Ollama (if available)

## 📊 Performance Optimization

### For Local Development

```yaml
# config.yaml
embeddings:
  device: "cuda"  # If you have GPU

llm:
  model: "llama3.1:8b"  # Better quality with GPU
```

### For Production/HF Spaces

```yaml
embeddings:
  device: "cpu"
  model_name: "sentence-transformers/all-MiniLM-L6-v2"  # Fast, lightweight

llm:
  model: "llama3.2:3b"  # Best balance for CPU
  temperature: 0.6
  max_tokens: 384  # Faster responses
```

## 🎯 Next Steps

1. ✅ Customize your profile information
2. ✅ Add comprehensive documents
3. ✅ Test with various questions
4. ✅ Deploy to Hugging Face Spaces
5. ✅ Share the link with recruiters!

## 📞 Support

- Check [README.md](README.md) for general information
- Review [config.yaml](config.yaml) for all options
- See [tests/](tests/) for examples
- Open an issue for bugs

---

**Happy chatting! 🚀**
