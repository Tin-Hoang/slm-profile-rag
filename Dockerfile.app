# Dockerfile for Streamlit RAG Application (for Docker Compose)
# Use with docker-compose.yml - Ollama runs as a separate service
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install CPU-only PyTorch first (saves ~2GB by avoiding CUDA dependencies)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ ./src/
COPY app.py .
COPY config.yaml .
COPY .streamlit/ ./.streamlit/

# Create necessary directories
RUN mkdir -p data/documents chroma_db bm25_index

# Create startup script
COPY <<'EOF' /app/start.sh
#!/bin/bash
set -e

echo "Starting SLM Profile RAG Application..."

# Wait for Ollama to be available (if using Ollama)
if [ -n "$OLLAMA_HOST" ]; then
    echo "Waiting for Ollama at $OLLAMA_HOST..."
    for i in {1..30}; do
        if curl -sf "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
            echo "Ollama is ready!"
            break
        fi
        echo "Waiting for Ollama... ($i/30)"
        sleep 2
    done
fi

# Build retrieval indexes if documents exist and indexes don't
if [ -d "data/documents" ] && [ "$(ls -A data/documents 2>/dev/null)" ]; then
    if [ ! -f "chroma_db/chroma.sqlite3" ]; then
        echo "Building retrieval indexes (BM25 + Vector)..."
        python -m src.build_vectorstore || echo "Index build failed, will build on first use"
    else
        echo "Indexes already exist, skipping build"
    fi
fi

# Start Streamlit
echo "Starting Streamlit on port ${STREAMLIT_SERVER_PORT:-7860}..."
exec streamlit run app.py \
    --server.port=${STREAMLIT_SERVER_PORT:-7860} \
    --server.address=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0} \
    --server.headless=true
EOF

RUN chmod +x /app/start.sh

# Expose Streamlit port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

CMD ["/app/start.sh"]
