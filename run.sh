#!/usr/bin/env bash
# Script to run the RAG chatbot application

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== SLM Profile RAG Chatbot ===${NC}"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python3 -m venv .venv
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source .venv/bin/activate

# Install dependencies if needed
if ! python -c "import slm_profile_rag" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -e ".[dev]"
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}No .env file found. Creating from .env.example...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env file with your settings.${NC}"
fi

# Check if Ollama is running
echo -e "${GREEN}Checking Ollama connection...${NC}"
OLLAMA_URL=$(grep OLLAMA_BASE_URL .env | cut -d '=' -f2)
if ! curl -s "${OLLAMA_URL:-http://localhost:11434}/api/tags" >/dev/null 2>&1; then
    echo -e "${RED}Warning: Cannot connect to Ollama at ${OLLAMA_URL:-http://localhost:11434}${NC}"
    echo -e "${YELLOW}Make sure Ollama is running: ollama serve${NC}"
    echo -e "${YELLOW}Continuing anyway...${NC}"
else
    echo -e "${GREEN}Ollama is running!${NC}"
fi

# Run the application
echo -e "${GREEN}Starting the application...${NC}"
python -m slm_profile_rag.app

# Deactivate virtual environment on exit
deactivate
