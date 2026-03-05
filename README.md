---
title: Multimodal RAG
emoji: 🧠
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Multimodal RAG — PDFs, scans, tables, charts.
---

# ⬡ Multimodal RAG System

A fully local, deployable **Multimodal Retrieval-Augmented Generation** system that answers questions **strictly** from your uploaded documents.

## Features

| Feature | Details |
|---|---|
| 📄 **Document types** | PDF (text + embedded images), scanned images (OCR), XLSX, DOCX, CSV, TXT |
| 🔍 **Multimodal** | Extracts text, tables, charts, and images from PDFs |
| 🧠 **Strict grounding** | Answers ONLY from documents — responds "I DON'T KNOW" otherwise |
| 💾 **Persistent vector store** | ChromaDB with cosine similarity (persists across restarts) |
| 🦙 **Local LLM** | Powered by Ollama (`llama3.2` by default) |
| 💬 **Conversation memory** | Remembers context; auto-summarizes when context window fills up |
| 📁 **Document management** | Add/remove documents via UI; index updates instantly |
| ⚡ **Streaming** | Token-by-token response streaming in chat |

## Setup

### Prerequisites

1. **Install Ollama**: https://ollama.com  
2. **Pull a model**: `ollama pull llama3.2`  
3. **Install Tesseract** (for OCR):  
   - macOS: `brew install tesseract`  
   - Ubuntu: `sudo apt-get install tesseract-ocr`  
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki

### Installation

```bash
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

- Gradio UI: http://localhost:7860  
- FastAPI docs: http://localhost:8000/docs

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `./data` | Directory for uploaded documents |
| `VECTORSTORE_DIR` | `./vectorstore` | ChromaDB persistence directory |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model to use |
| `API_BASE` | `http://localhost:8000` | FastAPI backend URL |

## Project Structure

```
multimodal-rag/
├── main.py                  # Entry point (runs API + UI)
├── api.py                   # FastAPI backend
├── app.py                   # Gradio UI
├── requirements.txt
├── data/                    # Uploaded documents
├── vectorstore/             # ChromaDB persistent storage
└── utils/
    ├── document_processor.py  # PDF/image/DOCX/XLSX extraction
    ├── vector_store.py        # ChromaDB manager
    ├── rag_engine.py          # RAG + Ollama integration
    └── memory.py              # Conversation memory manager
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/status` | System status, indexed docs, chunk count |
| POST | `/documents/upload` | Upload and index a document |
| DELETE | `/documents/{filename}` | Remove a document |
| POST | `/query` | Query the RAG system |
| POST | `/memory/clear` | Clear conversation memory |
| GET | `/memory/stats` | Memory usage stats |
| GET | `/models` | List available Ollama models |

## HuggingFace Spaces

For HuggingFace Spaces deployment, you need to add Ollama as a service or use an API-compatible endpoint. Set `OLLAMA_MODEL` and ensure the Ollama service is reachable at the configured host.

> ⚠️ **Note**: This system is designed for **local/private** deployments where your documents stay on your machine. For Spaces, ensure you have appropriate data privacy controls.
