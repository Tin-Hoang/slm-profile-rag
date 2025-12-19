# 🏗️ Architecture Overview

## System Architecture

<p align="center">
  <img src="docs/arch_system_design.png" alt="System Architecture Diagram" width="900" />
</p>
<p align="center"><em>High-level system architecture diagram illustrating the complete RAG pipeline flow</em></p>

### Detailed Component View

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
Documents → Load → Chunk → Build Indexes
   ↓          ↓       ↓           ↓
resume.pdf  PyPDF  Split    ┌─────────────────────────────┐
linkedin.html BS4   into    │  BM25 Index (keyword)       │
projects.docx docx  chunks  │  ./bm25_index/              │
                            ├─────────────────────────────┤
                            │  Vector Index (semantic)    │
                            │  ./chroma_db/ (ChromaDB)    │
                            └─────────────────────────────┘
```

**Command:** `python -m src.build_vectorstore`

**Strategy Options:**
- `--strategy bm25_vector` - Build both indexes (default, recommended)
- `--strategy vector` - Vector index only
- `--strategy bm25` - BM25 index only

### 2️⃣ Query Phase (Runtime) - WITH HYBRID SEARCH

```
User Question
     ↓
┌─────────────────────────────────────┐
│ Load Main Document (if enabled)      │
│  • Check cache validity              │
│  • Auto-detect format                │
│  • Count tokens                      │
│  • Summarize if needed               │
└────────────┬────────────────────────┘
             ↓
      [Main Doc Content]
             │
             ├─────────────────────────────────────┐
             ↓                                     ↓
┌─────────────────────────────────────┐    Main Doc (Priority)
│      Hybrid Retrieval Strategy       │          ↓
│                                      │   [Always Available]
│  ┌─────────────┐  ┌─────────────┐   │   [10k tokens max]
│  │   BM25      │  │   Vector    │   │
│  │  (keyword)  │  │ (semantic)  │   │
│  └──────┬──────┘  └──────┬──────┘   │
│         │                │          │
│         └───────┬────────┘          │
│                 ↓                   │
│    Reciprocal Rank Fusion (RRF)     │
│         (70% vector, 30% BM25)      │
└────────────────┬────────────────────┘
                 ↓
         [Retrieved Context]
                 │
                 └─────────┬───────────────────┘
                           ↓
                  Construct Prompt:
       System Prompt + Main Doc + Retrieved Context + Question
                           ↓
                  Send to Ollama LLM
                           ↓
                  Generate Answer
                           ↓
             Display in Streamlit UI (with source citations)
```

## Component Details

### 📑 Main Document Integration

The Main Document feature ensures critical profile information is always available in the LLM context, regardless of VectorDB retrieval quality.

```python
MainDocumentLoader
├── Format Auto-Detection
│   ├── Markdown (.md)    → LangChain TextLoader
│   ├── Plain Text (.txt) → LangChain TextLoader
│   ├── PDF (.pdf)        → Existing PDF loader
│   ├── Word (.docx)      → Existing DOCX loader
│   └── HTML (.html)      → Existing HTML loader
│
├── Token Management
│   ├── Counting: tiktoken (cl100k_base encoding)
│   ├── Max Limit: 10,000 tokens (configurable)
│   ├── Truncation: Smart token-based trimming
│   └── Summarization: LLM-based if exceeds limit
│
├── Caching Strategy
│   ├── File hash-based invalidation (MD5)
│   ├── Configurable check interval (60s default)
│   └── Automatic reload on file changes
│
└── Integration Point
    └── Positioned BEFORE VectorDB context (high priority)
```

**Architecture Flow:**
```
Main Document (Priority Context)
         ↓
    [Essential Info Always Available]
         ↓
VectorDB Retrieval (Additional Context)
         ↓
    [Supplementary Information]
         ↓
Combined Context → LLM → Response
```

**Benefits:**
- ✅ Critical information never missed by retrieval
- ✅ Auto-format detection (no manual config)
- ✅ Intelligent token management with LLM summarization
- ✅ Efficient caching for performance
- ✅ Graceful degradation if unavailable

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

### 🔍 Retrieval Strategy System

The retrieval system uses a pluggable strategy pattern for extensibility.

```python
RetrieverFactory
├── Registered Strategies
│   ├── "vector"      → VectorStrategy (semantic search)
│   ├── "bm25"        → BM25Strategy (keyword search)
│   ├── "bm25_vector" → BM25VectorStrategy (hybrid)
│   └── (future: "page_index", "graph_vector")
│
└── Strategy Interface (BaseRetrieverStrategy)
    ├── build_index(documents) → Build/update index
    ├── load_index()           → Load from disk
    ├── retrieve(query, k)     → Get relevant docs
    └── as_retriever()         → LangChain compatible
```

#### Hybrid Search (BM25 + Vector)

```
┌─────────────────────────────────────────────────────────────────┐
│                   BM25VectorStrategy                            │
│                                                                 │
│  ┌─────────────────────┐       ┌─────────────────────────┐     │
│  │   BM25Retriever     │       │   VectorRetriever       │     │
│  │   (keyword match)   │       │   (semantic match)      │     │
│  │                     │       │                         │     │
│  │  • Exact terms      │       │  • Meaning/context      │     │
│  │  • Abbreviations    │       │  • Synonyms             │     │
│  │  • Proper nouns     │       │  • Related concepts     │     │
│  └──────────┬──────────┘       └───────────┬─────────────┘     │
│             │ (k=10)                       │ (k=10)            │
│             └──────────┬───────────────────┘                   │
│                        ↓                                       │
│         Reciprocal Rank Fusion (RRF)                           │
│         weights: {vector: 0.7, bm25: 0.3}                      │
│                        ↓                                       │
│              Top K results (k=4)                               │
└─────────────────────────────────────────────────────────────────┘
```

**RRF Formula:** `score(d) = Σ (weight × 1/(k + rank(d)))`

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
    ├── Top K: 10 (before fusion)
    └── Distance: Cosine similarity
```

### 📝 BM25 Store Architecture

```python
BM25Store
├── Algorithm: BM25Okapi (rank-bm25)
│
├── Tokenization
│   ├── "simple": Whitespace + punctuation split
│   └── "nltk": NLTK word_tokenize (optional)
│
├── Persistence
│   ├── Location: ./bm25_index/
│   ├── Format: Pickle (index + documents)
│   └── Metadata: JSON (hash, stats)
│
└── Retrieval
    ├── Top K: 10 (before fusion)
    └── Scoring: BM25 term frequency
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
RAGPipeline
├── Retrieval Strategy
│   └── RetrieverFactory.create(strategy_name)
│       ├── "vector"      → Vector-only retriever
│       ├── "bm25"        → BM25-only retriever
│       └── "bm25_vector" → Fusion retriever (default)
│
├── Prompt Template
│   ├── System Prompt (from config)
│   ├── Main Document (priority context)
│   ├── Retrieved Context (from strategy)
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
   ├── Retrieval strategy (vector, bm25, bm25_vector)
   └── RAG parameters
      ↓
3. Code Defaults
   └── Fallback values if config missing
```

### Retrieval Configuration

```yaml
retrieval:
  strategy: "bm25_vector"      # Which strategy to use
  final_k: 4                   # Documents returned to LLM

  vector:
    search_type: "similarity"  # or "mmr"
    k: 10                      # Docs before fusion

  bm25:
    k: 10                      # Docs before fusion
    persist_path: "./bm25_index"
    tokenizer: "simple"

  fusion:
    algorithm: "rrf"           # Reciprocal Rank Fusion
    rrf_k: 60                  # RRF constant
    weights:
      vector: 0.7
      bm25: 0.3
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
 ├─ src.retrieval (RetrieverFactory, strategies)
 ├─ src.main_document_loader
 ├─ src.response_enhancer
 └─ langchain

src.retrieval
 ├─ src.retrieval.base (BaseRetrieverStrategy)
 ├─ src.retrieval.factory (RetrieverFactory)
 ├─ src.retrieval.fusion (RRF, FusionRetriever)
 ├─ src.retrieval.stores.bm25_store
 └─ src.retrieval.strategies.*

src.retrieval.strategies.vector
 ├─ src.vectorstore
 └─ src.retrieval.base

src.retrieval.strategies.bm25
 ├─ src.retrieval.stores.bm25_store
 └─ rank_bm25

src.retrieval.strategies.bm25_vector
 ├─ src.retrieval.strategies.vector
 ├─ src.retrieval.strategies.bm25
 └─ src.retrieval.fusion

src.main_document_loader
 ├─ src.config_loader
 ├─ src.document_processor
 ├─ src.llm_handler
 ├─ tiktoken
 └─ pathlib, hashlib, time

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
 ├─ src.retrieval (RetrieverFactory)
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

# 3. New retrieval strategies (extensible system!)
@RetrieverFactory.register("page_index")
class PageIndexStrategy(BaseRetrieverStrategy):
    """Vision-based document retrieval (ColPali)"""
    ...

@RetrieverFactory.register("graph_vector")
class GraphVectorStrategy(BaseRetrieverStrategy):
    """Knowledge graph + vector hybrid"""
    ...

# 4. New UI features
app.py → add_authentication()
app.py → add_analytics()

# 5. New embedding models
VectorStoreManager(embedding_model="...")

# 6. New fusion algorithms
# Add to src/retrieval/fusion.py
def custom_fusion(results_list, weights):
    ...
```

### Adding a New Retrieval Strategy

1. Create `src/retrieval/strategies/my_strategy.py`
2. Implement `BaseRetrieverStrategy` interface
3. Register with `@RetrieverFactory.register("my_strategy")`
4. Add config section in `config.yaml`
5. Import in `src/retrieval/strategies/__init__.py`

```python
from src.retrieval import RetrieverFactory
from src.retrieval.base import BaseRetrieverStrategy

@RetrieverFactory.register("my_strategy")
class MyStrategy(BaseRetrieverStrategy):
    @property
    def name(self) -> str:
        return "my_strategy"

    def build_index(self, documents): ...
    def load_index(self) -> bool: ...
    def retrieve(self, query, k=4): ...
    def as_retriever(self, **kwargs): ...
```

---

**This architecture prioritizes:**
- ✅ Simplicity (easy to understand)
- ✅ Modularity (easy to extend)
- ✅ Performance (optimized for small-medium datasets)
- ✅ Privacy (local processing)
- ✅ Deployability (cloud-ready)
