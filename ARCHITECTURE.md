# 🏗️ Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                          │
│                    (Streamlit Web App)                       │
│                        app.py                                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                              │
│                  (rag_pipeline.py)                           │
│                                                              │
│  ┌────────────────┐      ┌──────────────┐                  │
│  │  System Prompt │      │  QA Chain    │                  │
│  │   Template     │─────▶│  (LangChain) │                  │
│  └────────────────┘      └──────┬───────┘                  │
└─────────────────────────────────┼──────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
        ┌───────────────────────┐   ┌──────────────────────┐
        │   Vector Store        │   │    LLM Handler       │
        │  (vectorstore.py)     │   │  (llm_handler.py)    │
        │                       │   │                      │
        │  ┌─────────────────┐  │   │  ┌───────────────┐  │
        │  │   ChromaDB      │  │   │  │    Ollama     │  │
        │  │  (Embeddings)   │  │   │  │  (llama3.2)   │  │
        │  └─────────────────┘  │   │  └───────────────┘  │
        │                       │   │                      │
        │  ┌─────────────────┐  │   └──────────────────────┘
        │  │ HuggingFace     │  │
        │  │ Embeddings      │  │
        │  │ (all-MiniLM)    │  │
        │  └─────────────────┘  │
        └───────────────────────┘
                    ▲
                    │
        ┌───────────┴────────────┐
        │ Document Processor     │
        │ (document_processor.py)│
        │                        │
        │  ┌──────────────────┐  │
        │  │  PDF Loader      │  │
        │  │  DOCX Loader     │  │
        │  │  HTML Loader     │  │
        │  │  Text Loader     │  │
        │  └──────────────────┘  │
        │                        │
        │  ┌──────────────────┐  │
        │  │ Text Splitter    │  │
        │  │ (Chunking)       │  │
        │  └──────────────────┘  │
        └────────────────────────┘
                    ▲
                    │
        ┌───────────┴────────────┐
        │   Source Documents     │
        │   data/documents/      │
        │                        │
        │  • Resume.pdf          │
        │  • LinkedIn.html       │
        │  • Projects.docx       │
        │  • ...                 │
        └────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                 Configuration Layer                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  config.yaml          env.template (.env)                   │
│  ├─ profile          ├─ OLLAMA_BASE_URL                     │
│  ├─ llm              ├─ CHROMA_PERSIST_DIR                  │
│  ├─ embeddings       ├─ DOCUMENTS_DIR                       │
│  ├─ vectorstore      ├─ LOG_LEVEL                           │
│  ├─ document_proc    └─ API_KEYS (optional)                 │
│  ├─ rag                                                      │
│  ├─ ui                                                       │
│  └─ logging                                                  │
│                                                              │
│         (config_loader.py)                                   │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1️⃣ Indexing Phase (One-time Setup)

```
Documents → Load → Chunk → Embed → Store in ChromaDB
   ↓          ↓       ↓       ↓           ↓
resume.pdf  PyPDF  Split   HF-     Vector Database
linkedin.html BS4   into    Embed   (Persistent)
projects.docx docx  chunks  Model
```

**Command:** `python -m src.build_vectorstore`

### 2️⃣ Query Phase (Runtime)

```
User Question
     ↓
Embed Question (same model as documents)
     ↓
Similarity Search in ChromaDB (get top-k chunks)
     ↓
Retrieve relevant document chunks
     ↓
Construct Prompt:
  System Prompt + Retrieved Context + User Question
     ↓
Send to Ollama LLM
     ↓
Generate Answer
     ↓
Display in Streamlit UI (with source citations)
```

## Component Details

### 📄 Document Processing Pipeline

```python
DocumentProcessor
├── Supported Formats
│   ├── PDF          → pypdf
│   ├── Word         → python-docx
│   ├── HTML         → BeautifulSoup4
│   └── Text/MD      → LangChain TextLoader
│
├── Chunking Strategy
│   ├── Size: 1000 chars (configurable)
│   ├── Overlap: 200 chars (configurable)
│   └── Separators: ["\n\n", "\n", ". ", " ", ""]
│
└── Output: List[Document]
    └── Each with content + metadata
```

### 🧠 Vector Store Architecture

```python
VectorStoreManager
├── Embedding Model
│   └── sentence-transformers/all-MiniLM-L6-v2
│       ├── Dimension: 384
│       ├── Speed: ~2000 sentences/sec (CPU)
│       └── Quality: Good for semantic search
│
├── ChromaDB
│   ├── Type: Persistent (SQLite)
│   ├── Location: ./chroma_db/
│   ├── Collection: profile_documents
│   └── Indexing: HNSW (approximate NN)
│
└── Retrieval
    ├── Search Type: Similarity (or MMR)
    ├── Top K: 4 (configurable)
    └── Distance: Cosine similarity
```

### 🤖 LLM Integration

```python
LLMHandler
├── Provider: Ollama
│   ├── Base URL: http://localhost:11434
│   └── Protocol: HTTP/REST API
│
├── Model Options
│   ├── llama3.2:3b  (Recommended)
│   ├── phi3:mini
│   ├── gemma2:2b
│   └── llama3.1:8b  (with GPU)
│
└── Parameters
    ├── Temperature: 0.7
    ├── Max Tokens: 512
    ├── Top P: 0.9
    └── Context Window: 8192 tokens
```

### 🔗 RAG Chain

```python
RetrievalQA Chain
├── Retriever
│   └── VectorStore.as_retriever(k=4)
│
├── Prompt Template
│   ├── System Prompt (from config)
│   ├── Retrieved Context (from vector store)
│   └── User Question
│
├── LLM
│   └── Ollama (configured model)
│
└── Output
    ├── Answer (generated text)
    └── Source Documents (citations)
```

## Configuration Hierarchy

```
1. Environment Variables (.env)
   ├── Override config.yaml values
   ├── Secrets (API keys)
   └── Runtime settings (ports, URLs)
      ↓
2. config.yaml
   ├── Application defaults
   ├── Model selection
   └── RAG parameters
      ↓
3. Code Defaults
   └── Fallback values if config missing
```

## Deployment Architecture

### Local Development

```
┌──────────────────┐
│   Developer      │
│   Machine        │
│                  │
│  ┌────────────┐  │
│  │  Ollama    │  │  ← Port 11434
│  │  Server    │  │
│  └────────────┘  │
│                  │
│  ┌────────────┐  │
│  │ Streamlit  │  │  ← Port 8501
│  │    App     │  │
│  └────────────┘  │
│                  │
│  ┌────────────┐  │
│  │ ChromaDB   │  │  ← ./chroma_db/
│  │  (local)   │  │
│  └────────────┘  │
└──────────────────┘
```

### Hugging Face Spaces

```
┌───────────────────────────────┐
│   HF Spaces Container         │
│                               │
│  ┌─────────────────────────┐  │
│  │  Dockerfile             │  │
│  │  ├─ Install Ollama      │  │
│  │  ├─ Pull Model          │  │
│  │  └─ Start Services      │  │
│  └─────────────────────────┘  │
│                               │
│  ┌──────────┐  ┌──────────┐  │
│  │  Ollama  │  │Streamlit │  │
│  │  Server  │  │   App    │  │
│  └──────────┘  └──────────┘  │
│                               │
│  ┌─────────────────────────┐  │
│  │  Persistent Storage     │  │
│  │  └─ chroma_db/          │  │
│  └─────────────────────────┘  │
└───────────────────────────────┘
         ↑
         │ HTTPS
         │
┌────────┴──────────┐
│   Public Users    │
│   (Recruiters)    │
└───────────────────┘
```

## Build & Packaging

### UV + Hatchling + Versioningit

```
pyproject.toml
├── [build-system]
│   ├── requires: ["hatchling", "versioningit"]
│   └── build-backend: "hatchling.build"
│
├── [project]
│   ├── name: "slm-profile-rag"
│   ├── version: <from git tags via versioningit>
│   └── dependencies: [...]
│
├── [tool.versioningit]
│   ├── Read git tags (v0.1.0, v0.2.0, etc.)
│   ├── Generate version string
│   └── Write to src/_version.py
│
└── [tool.ruff]
    ├── Linting rules
    └── Formatting config
```

### Version from Git Tags

```bash
# Tag release
git tag v0.1.0
git push origin v0.1.0

# Version automatically set
python -c "from src import __version__; print(__version__)"
# Output: 0.1.0

# Development version (after tag)
# Output: 0.1.0+5.g1a2b3c4  (5 commits after v0.1.0)
```

## Code Quality Pipeline

```
Developer Writes Code
        ↓
┌───────────────────┐
│   Pre-commit      │
│   (Optional)      │
└─────────┬─────────┘
          ↓
┌───────────────────┐
│  Ruff Check       │  ← Linting
│  Ruff Format      │  ← Formatting
└─────────┬─────────┘
          ↓
┌───────────────────┐
│  Git Commit       │
└─────────┬─────────┘
          ↓
┌───────────────────┐
│  Push to GitHub   │
└─────────┬─────────┘
          ↓
┌───────────────────┐
│  GitHub Actions   │
│  • Lint Check     │
│  • Format Check   │
│  • Tests (future) │
└───────────────────┘
```

## Module Dependencies

```
app.py
 ├─ src.config_loader
 ├─ src.rag_pipeline
 └─ streamlit

src.rag_pipeline
 ├─ src.config_loader
 ├─ src.llm_handler
 ├─ src.vectorstore
 └─ langchain

src.vectorstore
 ├─ src.config_loader
 ├─ chromadb
 └─ langchain_huggingface

src.llm_handler
 ├─ src.config_loader
 └─ langchain_community.llms

src.document_processor
 ├─ src.config_loader
 ├─ pypdf
 ├─ python-docx
 ├─ beautifulsoup4
 └─ langchain

src.config_loader
 ├─ pyyaml
 └─ python-dotenv

src.build_vectorstore
 ├─ src.document_processor
 └─ src.vectorstore
```

## Performance Characteristics

### Indexing (One-time)

| Documents | Chunks | Embedding Time | ChromaDB Insert | Total |
|-----------|--------|----------------|-----------------|-------|
| 5 PDFs    | ~100   | ~5 seconds     | ~1 second       | ~6s   |
| 20 PDFs   | ~400   | ~20 seconds    | ~2 seconds      | ~22s  |
| 50 PDFs   | ~1000  | ~50 seconds    | ~5 seconds      | ~55s  |

### Query (Runtime)

| Step | Time (CPU) | Time (GPU) |
|------|------------|------------|
| Embed query | 50ms | 10ms |
| Vector search | 10-50ms | 10-50ms |
| LLM inference | 2-5s | 0.5-1s |
| **Total** | **2-5s** | **0.5-1s** |

### Memory Usage

| Component | RAM | Disk |
|-----------|-----|------|
| Streamlit | ~200MB | - |
| Ollama (llama3.2:3b) | ~2GB | ~2GB |
| ChromaDB | ~100MB | ~50MB per 1k docs |
| Embeddings | ~500MB | ~500MB |
| **Total** | **~3GB** | **~3GB** |

## Security Architecture

```
User Input
    ↓
┌─────────────────────┐
│ Input Validation    │  ← Length limits
│ (Streamlit)         │  ← Character filtering
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│ RAG Pipeline        │  ← Context isolation
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│ LLM (Local)         │  ← No external API calls
│                     │  ← Data stays local
└─────────────────────┘

Secrets Management:
├─ .env (local)
├─ .gitignore (.env excluded)
└─ HF Spaces Secrets (cloud)
```

## Extensibility Points

### 🔌 Plugin Architecture

```python
# Easy to extend:

# 1. New document types
DocumentProcessor.load_custom_format()

# 2. New LLM providers
LLMHandler.get_openai_llm()
LLMHandler.get_anthropic_llm()

# 3. New retrieval strategies
VectorStoreManager.hybrid_search()

# 4. New UI features
app.py → add_authentication()
app.py → add_analytics()

# 5. New embedding models
VectorStoreManager(embedding_model="...")
```

---

**This architecture prioritizes:**
- ✅ Simplicity (easy to understand)
- ✅ Modularity (easy to extend)
- ✅ Performance (optimized for small-medium datasets)
- ✅ Privacy (local processing)
- ✅ Deployability (cloud-ready)

