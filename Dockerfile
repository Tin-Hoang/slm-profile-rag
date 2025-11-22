# Dockerfile for HuggingFace Spaces with Ollama support
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p data/documents chroma_db

# Expose Streamlit port
EXPOSE 7860

# Expose Ollama port
EXPOSE 11434

# Start script that runs both Ollama and Streamlit
COPY <<'EOF' /app/start.sh
#!/bin/bash
set -e

# Start Ollama in the background
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
sleep 5

# Pull the model (use small model for HF Spaces)
MODEL=${OLLAMA_MODEL:-llama3.2:3b}
echo "Pulling model: $MODEL"
ollama pull $MODEL || echo "Model pull failed, will retry on app start"

# Build vector store if documents exist and DB doesn't
if [ -d "data/documents" ] && [ ! -d "chroma_db/chroma.sqlite3" ]; then
    echo "Building vector store..."
    python -m src.build_vectorstore || echo "Vector store build failed, will build on first use"
fi

# Start Streamlit
streamlit run app.py --server.port=7860 --server.address=0.0.0.0
EOF

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]

