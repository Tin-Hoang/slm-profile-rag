# Makefile for SLM Profile RAG
# Run 'make help' to see available commands

.PHONY: help install install-dev test test-cov lint format clean \
        docker-up docker-down docker-build docker-logs docker-shell \
        docker-rebuild docker-ps ollama-pull index-build run

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
DOCKER_COMPOSE := docker compose
APP_PORT := 7860
OLLAMA_MODEL ?= llama3.2:3b

# =============================================================================
# Help
# =============================================================================
help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# =============================================================================
# Development Setup
# =============================================================================
install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

install-uv: ## Install dependencies using uv (faster)
	uv pip install -r requirements.txt

install-uv-dev: ## Install dev dependencies using uv
	uv pip install -e ".[dev]"

# =============================================================================
# Testing & Quality
# =============================================================================
test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint: ## Run linter (ruff)
	ruff check src/ tests/ app.py

lint-fix: ## Run linter and fix issues
	ruff check src/ tests/ app.py --fix

format: ## Format code (ruff)
	ruff format src/ tests/ app.py

format-check: ## Check code formatting
	ruff format src/ tests/ app.py --check

check: lint format-check ## Run all checks (lint + format)

# =============================================================================
# Local Development
# =============================================================================
run: ## Run the Streamlit app locally
	streamlit run app.py --server.port=$(APP_PORT)

index-build: ## Build vector and BM25 indexes
	$(PYTHON) -m src.build_vectorstore

index-clean: ## Remove existing indexes
	rm -rf chroma_db/ bm25_index/

index-rebuild: index-clean index-build ## Rebuild indexes from scratch

# =============================================================================
# Docker Compose
# =============================================================================
docker-up: ## Start all services
	$(DOCKER_COMPOSE) up -d

docker-down: ## Stop all services
	$(DOCKER_COMPOSE) down

docker-build: ## Build Docker images
	$(DOCKER_COMPOSE) build

docker-rebuild: ## Rebuild and restart services
	$(DOCKER_COMPOSE) up -d --build

docker-logs: ## Show logs (follow mode)
	$(DOCKER_COMPOSE) logs -f

docker-logs-app: ## Show app logs only
	$(DOCKER_COMPOSE) logs -f app

docker-ps: ## Show running containers
	$(DOCKER_COMPOSE) ps

docker-shell: ## Open shell in app container
	$(DOCKER_COMPOSE) exec app /bin/bash

docker-clean: ## Stop services and remove volumes
	$(DOCKER_COMPOSE) down -v

docker-restart: docker-down docker-up ## Restart all services

# =============================================================================
# Ollama Commands
# =============================================================================
ollama-pull: ## Pull the configured Ollama model
	$(DOCKER_COMPOSE) exec ollama ollama pull $(OLLAMA_MODEL)

ollama-list: ## List available Ollama models
	$(DOCKER_COMPOSE) exec ollama ollama list

ollama-shell: ## Open shell in Ollama container
	$(DOCKER_COMPOSE) exec ollama /bin/bash

# =============================================================================
# Setup & Utilities
# =============================================================================
setup: ## Initial project setup (copy env, install deps)
	@if [ ! -f .env ]; then cp .env.template .env && echo "Created .env from template"; fi
	@echo "Installing dependencies..."
	@make install-dev
	@echo "Setup complete! Edit .env if needed, then run 'make run' or 'make docker-up'"

clean: ## Clean up build artifacts and caches
	rm -rf __pycache__ .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean index-clean ## Clean everything including indexes
	rm -rf *.egg-info build dist

# =============================================================================
# Pre-commit
# =============================================================================
pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files
