---
title: Profile Q&A Assistant
emoji: 💼
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.0.0"
app_file: app.py
pinned: false
license: mit
---

# Profile Q&A Assistant

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about a candidate's profile using Small Language Models (SLM) via Ollama.

## Setup

1. This Space requires Ollama to be running. For local deployment:
   - Install [Ollama](https://ollama.ai/)
   - Pull a model: `ollama pull llama2`
   - Run Ollama: `ollama serve`

2. Add your profile documents to the `profile_docs/` directory:
   - Supported formats: PDF, HTML, TXT, MD
   - Examples: resume.pdf, linkedin_profile.html, project_reports/

3. Configure environment variables in Space Settings:
   - `OLLAMA_BASE_URL` (default: http://localhost:11434)
   - `OLLAMA_MODEL` (default: llama2)

## Usage

Ask questions about the candidate's:
- Educational background
- Work experience
- Skills and expertise
- Projects
- Achievements

The chatbot will retrieve relevant information from the profile documents and generate accurate answers.

## Configuration

See `config.yaml` for detailed configuration options.
