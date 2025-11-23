# Main Document Feature Guide

## Overview

The Main Document feature provides guaranteed context availability for your RAG chatbot by ensuring critical profile information is always included in the LLM prompt, positioned before VectorDB retrieval results.

## How It Works

### 1. Architecture

```
Query → [Main Document] → [VectorDB Retrieval] → [Combined Context] → LLM → Response
           ↑ PRIORITY         ↑ SUPPLEMENTARY
```

### 2. Loading Process

1. **File Detection**: Auto-detects format from extension
2. **Content Loading**: Uses appropriate loader (PDF, DOCX, HTML, MD, TXT)
3. **Token Counting**: Calculates tokens using tiktoken
4. **Size Management**:
   - If ≤ 10k tokens → Use as-is
   - If > 10k tokens → LLM summarization to ~8k tokens
5. **Caching**: Stores in memory, reloads only on file change

### 3. Prompt Structure

```
[System Prompt]

=== ESSENTIAL PROFILE INFORMATION ===
(This information is always available and takes priority)

[Main Document Content - 10k tokens max]

=== ADDITIONAL CONTEXT FROM DOCUMENTS ===

[VectorDB Retrieved Chunks]

Question: [User Question]

Answer:
```

## Configuration

### Basic Setup

```yaml
main_document:
  enabled: true
  path: "data/documents/main_profile.md"
  max_tokens: 10000
```

### Advanced Options

```yaml
main_document:
  enabled: true
  path: "data/documents/main_profile.md"
  max_tokens: 10000
  position: "before"  # before/after VectorDB context

  # Summarization
  summarize_if_exceeds: true
  summarization_target_tokens: 8000
  summarization_prompt: |
    Custom summarization instructions...

  # Caching
  cache_enabled: true
  cache_check_interval: 60  # seconds

  # Error handling
  fail_silently: true
  fallback_to_vectordb_only: true
```

## Best Practices

### Content Structure

**DO Include:**
- ✅ Full name and professional title
- ✅ Contact information (email, LinkedIn, etc.)
- ✅ Current role and responsibilities
- ✅ Core technical skills (with proficiency levels)
- ✅ Major projects with metrics/achievements
- ✅ Education and certifications
- ✅ Career summary/objective

**DON'T Include:**
- ❌ Sensitive personal information
- ❌ Redundant lengthy descriptions
- ❌ Information better suited for retrieval (blog posts, detailed docs)
- ❌ Frequently changing information

### Format Recommendations

| Format | Best For | Notes |
|--------|----------|-------|
| **Markdown** (.md) | Structured profiles | Readable, easy to edit, recommended |
| **Text** (.txt) | Simple profiles | Plain text, fastest to load |
| **PDF** (.pdf) | Existing resumes | Auto-parsed, may need cleanup |
| **Word** (.docx) | Existing documents | Good formatting preservation |
| **HTML** (.html) | LinkedIn exports | Auto-parsed, clean content |

### Token Management

**Token Budget Example** (llama3.2:3b with 8192 context window):

| Component | Tokens | Percentage |
|-----------|--------|------------|
| Main Document | 7,000 | 85% |
| VectorDB Context | 500 | 6% |
| Output Generation | 512 | 6% |
| Safety Buffer | 180 | 3% |
| **Total** | **8,192** | **100%** |

**Optimization Tips:**
1. Keep main doc under 7,000 tokens for optimal retrieval space
2. Use bullet points instead of verbose paragraphs
3. Remove redundant information
4. Let LLM summarize if necessary

## Troubleshooting

### Issue: Main document not loading

**Check:**
```bash
# Verify file exists
ls -lh data/documents/main_profile.md

# Check logs
tail -f app.log | grep "main_document"
```

**Solution:**
- Verify `path` in config.yaml is correct
- Check file permissions
- Ensure format is supported

### Issue: Content truncated/summarized unexpectedly

**Diagnosis:**
- Check token count in logs: "Main document loaded: X tokens"
- If X > 10,000, content will be summarized

**Solution:**
- Reduce content length
- Increase `max_tokens` in config
- Disable summarization: `summarize_if_exceeds: false`

### Issue: Responses don't seem to use main document

**Check:**
```python
# In Python console:
from src.rag_pipeline import get_rag_pipeline

pipeline = get_rag_pipeline()
info = pipeline.get_main_document_info()
print(info)
```

**Verify:**
- `enabled: true` in info
- `loaded: true` in info
- `tokens > 0` in info

## API Reference

### MainDocumentLoader Class

```python
from src.main_document_loader import get_main_document_loader

loader = get_main_document_loader()

# Load document
content = loader.load_main_document()

# Count tokens
tokens = loader.count_tokens("your text here")

# Truncate to token limit
truncated = loader.truncate_to_tokens(content, max_tokens=5000)

# Invalidate cache (force reload)
loader.invalidate_cache()
```

### RAGPipeline Methods

```python
from src.rag_pipeline import get_rag_pipeline

pipeline = get_rag_pipeline()

# Get main document info
info = pipeline.get_main_document_info()
# Returns: {'enabled': bool, 'loaded': bool, 'tokens': int, 'path': str, ...}

# Reload main document at runtime
success = pipeline.reload_main_document()

# Get token budget breakdown
budget = pipeline._calculate_context_budget()
```

## Examples

### Example 1: Markdown Main Profile

```markdown
# Jane Doe

**Title**: Senior Data Scientist | ML Engineer
**Email**: jane.doe@example.com
**LinkedIn**: linkedin.com/in/janedoe
**Location**: San Francisco, CA

## Professional Summary
Data Scientist with 8+ years building production ML systems. Specialized in NLP, recommendation systems, and MLOps.

## Core Skills
- **Languages**: Python, SQL, R
- **ML/DL**: PyTorch, TensorFlow, scikit-learn, Hugging Face
- **MLOps**: Docker, Kubernetes, MLflow, AWS SageMaker
- **Data**: Spark, Airflow, PostgreSQL, Redis

## Experience

### Tech Corp - Senior Data Scientist (2020-Present)
- Built recommendation engine serving 5M+ users, improving CTR by 23%
- Deployed 12 ML models to production using CI/CD pipelines
- Led team of 4 data scientists on personalization initiatives

### StartupXYZ - Data Scientist (2017-2020)
- Developed NLP pipeline for customer support automation (85% accuracy)
- Reduced model training costs by 40% through infrastructure optimization

## Education
- **M.S. Computer Science**, Stanford University (2017)
- **B.S. Mathematics**, UC Berkeley (2015)

## Certifications
- AWS Certified Machine Learning - Specialty
- Google Professional Data Engineer
```

### Example 2: Minimal Text Profile

```text
John Smith - AI Research Engineer
Email: john.smith@email.com
GitHub: github.com/johnsmith

5+ years in deep learning and computer vision.

Key Skills: PyTorch, TensorFlow, OpenCV, CUDA, Python, C++

Notable Projects:
- Real-time object detection system (45 FPS, 92% mAP)
- Federated learning framework for healthcare ML
- Open-source contributor: torchvision, detectron2

Education: Ph.D. Computer Science, MIT (2019)
```

## Migration Guide

### From VectorDB-Only to Main Document

**Before:**
- All content in `data/documents/` processed equally
- Retrieval quality determines context

**After:**
1. **Identify critical information** that must always be available
2. **Create main document** with this essential content
3. **Enable feature** in config.yaml
4. **Keep other documents** in `data/documents/` for supplementary info
5. **Test queries** to verify main document is used

**You don't need to remove existing documents** - main document supplements, not replaces, VectorDB retrieval!

## Advanced: Token Budget Calculation

The system automatically calculates and manages token budgets:

```python
budget = {
    "model_context_window": 8192,      # Total available
    "main_doc_tokens": 7000,           # Main document
    "max_output_tokens": 512,          # LLM generation
    "buffer_tokens": 500,              # Safety buffer
    "available_for_retrieval": 180,    # Remaining for VectorDB
    "total_input_budget": 7680,        # Total input space
}
```

**Warnings:**
- If main doc > 50% of context window, system logs warning
- Adjust `max_tokens` or enable summarization to optimize

## Performance Considerations

### Memory Usage
- **Cached content**: ~1-5 MB per main document
- **Token counting**: Negligible (< 10ms)
- **Summarization**: 2-10 seconds (only if needed)

### Loading Times
- **First load**: 50-200ms (depends on format)
- **Cached loads**: < 1ms
- **Cache validation**: < 5ms (file hash check)

### Recommendations
- Enable caching for production (default: true)
- Set `cache_check_interval` to 300+ for rarely changing docs
- Use markdown or text for fastest loading

---

**For more help, see [ARCHITECTURE.md](../ARCHITECTURE.md) or [README.md](../README.md).**
