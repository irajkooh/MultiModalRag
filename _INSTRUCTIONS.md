# 🧠 Multimodal RAG — Installation & Deployment Guide

A grounded, document-only question-answering system supporting PDFs (text, tables, embedded images/charts), scanned images, DOCX, XLSX, CSV, and TXT files.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Local Installation](#local-installation)
4. [Running the App](#running-the-app)
5. [Using the UI](#using-the-ui)
6. [Deploying to HuggingFace Spaces](#deploying-to-huggingface-spaces)
7. [Configuration Reference](#configuration-reference)
8. [Supported File Types](#supported-file-types)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)
11. [Project Structure](#project-structure)

> **Live Space:** https://huggingface.co/spaces/irajkoohi/MultiModalRag

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Gradio UI (port 7860)              │
│  • Document upload/remove   • Sample question pills  │
│  • Chat interface           • Memory stats           │
└───────────────────┬─────────────────────────────────┘
                    │ HTTP (REST)
┌───────────────────▼─────────────────────────────────┐
│               FastAPI Backend (port 8000)            │
│  /documents/upload   /documents/{name}  /query       │
│  /memory/clear       /memory/stats      /status      │
└──────┬────────────────────────┬────────────────────┘
       │                        │
┌──────▼──────┐        ┌────────▼────────────────────┐
│  ChromaDB   │        │   Ollama LLM                 │
│ (persists   │        │  (llama3.2 by default)       │
│  to disk)   │        │  OLLAMA_HOST configurable    │
└─────────────┘        └─────────────────────────────┘
       ▲
┌──────┴──────────────────────────────┐
│       Document Processor            │
│  PDF → text + OCR images + tables   │
│  Scanned images → Tesseract OCR     │
│  DOCX/XLSX/CSV → structured text    │
└─────────────────────────────────────┘
```

### Entry points

| File | Purpose |
|---|---|
| `app.py` | **Single entrypoint for both environments** — detects HF Spaces via `SPACE_ID` env var. Locally: kills stale ports, auto-starts Ollama, opens browser. On HF: skips all local helpers |
| `frontend.py` | Gradio UI — `build_ui()` and `CUSTOM_CSS` imported by `app.py` |
| `Dockerfile` | HF Spaces Docker build — inlines Ollama startup, model pull, and runs `app.py` |

**Key design principle:** The LLM is instructed to answer *only* from retrieved context. If the answer is not in the documents, it responds exactly: `I DON'T KNOW`.

---

## Prerequisites

### 1. Python 3.10+

```bash
python --version   # must be 3.10 or higher
```

Download from https://python.org if needed.

### 2. Ollama

Ollama runs LLMs locally. Install it from https://ollama.com and pull at least one model:

```bash
# Install Ollama (macOS / Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the default model
ollama pull llama3.2

# Or use a smaller/faster model
ollama pull llama3.2:1b
ollama pull mistral
ollama pull phi3
```

Confirm it's running:
```bash
ollama list    # should show pulled models
```

### 3. Tesseract OCR

Required for processing scanned images and image-based PDFs.

**macOS:**
```bash
brew install tesseract
```

**Ubuntu / Debian:**
```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```

**Windows:**
Download the installer from https://github.com/UB-Mannheim/tesseract/wiki  
Add the install directory (e.g. `C:\Program Files\Tesseract-OCR`) to your `PATH`.

Verify:
```bash
tesseract --version
```

### 4. Poppler (for `pdf2image`, optional but recommended)

Needed if you want better PDF image rendering.

**macOS:**
```bash
brew install poppler
```

**Ubuntu:**
```bash
sudo apt-get install -y poppler-utils
```

**Windows:** Download from https://github.com/oschwartz10612/poppler-windows

---

## Local Installation

### Step 1 — Clone / extract the project

```bash
unzip multimodal-rag.zip
cd multimodal-rag
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv

# Activate it:
# macOS / Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the embedding model (`all-MiniLM-L6-v2`, ~90 MB). This only happens once and is cached locally.

### Step 4 — (Optional) Configure environment

Copy the example env file and adjust as needed:

```bash
cp .env.example .env
```

Edit `.env`:

```env
DATA_DIR=./data
VECTORSTORE_DIR=./vectorstore
OLLAMA_MODEL=llama3.2
API_BASE=http://localhost:8000
```

---

## Running the App

### Start everything with one command

```bash
python app.py
```

This will:
1. **Auto-start Ollama** if `ollama serve` is not already running
2. **Auto-pull the model** (`llama3.2` by default) if not already downloaded — first pull is ~2 GB
3. Start the FastAPI backend on `http://localhost:8000`
4. Auto-index any documents already in the `data/` folder
5. Launch the Gradio UI on `http://localhost:7860`
6. Open `http://localhost:7860` in your browser automatically

> You no longer need to run `ollama serve` manually before starting the app.

### Run components separately (optional)

**Backend only:**
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend only** (after backend is running):
```bash
python frontend.py
```

**API docs** (Swagger UI):
```
http://localhost:8000/docs
```

---

## Using the UI

### Document Manager (left panel)

| Action | How |
|---|---|
| **Upload documents** | Click the file picker, select one or more files, then click **⬆ Upload & Index** |
| **Remove a document** | Select a document from the radio list, click **🗑 Remove** |
| **Refresh the list** | Click **↻ Refresh** |

The **Ask →** button and all sample question buttons are **automatically disabled** when no documents are indexed.

### Chat (right panel)

- Type your question in the text box and press **Enter** or **Ask →**
- Click any **sample question pill** to instantly send a pre-written query
- Responses are streamed character-by-character
- Sources are cited at the end of each answer
- If the answer is not in the documents, the system responds: **I DON'T KNOW**

### Memory Controls

| Control | Description |
|---|---|
| **Memory stats bar** | Shows message count, token usage, and `Summarized: Yes/No` |
| **🧹 Clear Memory** | Wipes conversation history so the next question starts fresh |
| **Context chunks slider** | Number of document chunks retrieved per query (1–10, default 5). Increase for broader questions |

### Auto Memory Compression

When the conversation history exceeds ~2,000 tokens, older messages are automatically compressed into a bullet-point summary. This keeps the context window manageable without losing important prior context. The stats bar shows `Summarized: Yes` when this has happened.

---

## Deploying to HuggingFace Spaces

This project is deployed at: **https://huggingface.co/spaces/irajkoohi/MultiModalRag**

The deployment uses a **Docker Space** with Ollama bundled inside the container.
Key files:

| File | Role |
|---|---|
| `Dockerfile` | Installs Python, Tesseract, Poppler, Ollama, CPU-only PyTorch, all deps. Inlines Ollama startup + model pull in `CMD` |
| `app.py` | Single entrypoint — auto-detects HF via `SPACE_ID` env var |
| `frontend.py` | Gradio UI imported by `app.py` |
| `README.md` | Must have `sdk: docker` in the YAML front-matter |
| `.dockerignore` | Excludes `.venv/`, `data/`, `vectorstore/`, `.DS_Store` from build |

### First-time setup

```bash
# 1. Log in
huggingface-cli login        # paste a Write token from hf.co/settings/tokens

# 2. Create the Docker Space (only needed once)
huggingface-cli repo create MultiModalRag --repo-type space --space_sdk docker

# 3. Add remote and push
git remote add space https://huggingface.co/spaces/irajkoohi/MultiModalRag
git push space main
```

### Subsequent deployments

```bash
git add -A
git commit -m "your change description"
git push space main
```

> `--force` is only needed if the remote has diverged (e.g. after HF auto-generates an initial commit).

### Build timeline

| Stage | Duration |
|---|---|
| Docker build + pip install | ~3–4 min |
| `ollama pull llama3.2` (~2 GB) | ~2–3 min (first cold start only) |
| App ready | ~30 s after model is pulled |

Subsequent restarts skip the pull if the model is already cached.

### Space Environment Variables (optional overrides)

Set in **Space Settings → Variables and secrets**:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | Any model available on Ollama — `llama3.2:1b` is faster/smaller |
| `OLLAMA_HOST` | `http://localhost:11434` | Override to point at an external Ollama server |
| `DATA_DIR` | `./data` | Document storage path |
| `VECTORSTORE_DIR` | `./vectorstore` | ChromaDB storage path |
| `API_BASE` | `http://localhost:8000` | FastAPI URL used by Gradio |

### Persistent Storage

By default, uploaded documents and the vector store reset when the Space restarts. To persist them, enable **Persistent Storage** in Space Settings (available on paid HF plans) and set `DATA_DIR=/data` and `VECTORSTORE_DIR=/data/vectorstore`.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `./data` | Folder where uploaded documents are stored |
| `VECTORSTORE_DIR` | `./vectorstore` | ChromaDB persistence directory |
| `OLLAMA_MODEL` | `llama3.2` | Model name as shown in `ollama list` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API URL — change to use a remote Ollama server |
| `API_BASE` | `http://localhost:8000` | URL of the FastAPI backend (used by Gradio) |
| `TORCH_DEVICE` | *(auto)* | Force embedding device: `mps`, `cuda`, or `cpu` |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model for embeddings |

To change the model, either set the env variable before running:

```bash
OLLAMA_MODEL=mistral python app.py
```

Or edit `.env`.

---

## GPU Acceleration (MPS, CUDA, CPU)

The embedding model (`sentence-transformers`) runs on the best available device, detected automatically at startup. The active device is shown in the UI status bar.

### Priority order
```
CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU
```

### Apple Silicon — MPS

MPS is supported on M1/M2/M3/M4 Macs running macOS 12.3+ with PyTorch 2.1+.

**Install PyTorch with MPS support:**
```bash
# Standard pip install already includes MPS on macOS arm64:
pip install torch>=2.1.0

# Verify MPS is available:
python -c "import torch; print(torch.backends.mps.is_available())"
# → True
```

**Force MPS explicitly** (optional — it's auto-detected):
```bash
TORCH_DEVICE=mps python app.py
```

Or set in `.env`:
```env
TORCH_DEVICE=mps
```

### NVIDIA GPU — CUDA

```bash
# Install CUDA-enabled PyTorch (adjust cu121 to match your CUDA version):
pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Verify:
python -c "import torch; print(torch.cuda.is_available())"
# → True
```

### Force CPU

```bash
TORCH_DEVICE=cpu python app.py
```

### Choosing a larger embedding model

The default `all-MiniLM-L6-v2` is fast and lightweight. On GPU/MPS you can use a more powerful model for better retrieval quality:

```env
# In .env:
EMBED_MODEL=all-mpnet-base-v2        # higher quality, ~420 MB
EMBED_MODEL=BAAI/bge-base-en-v1.5    # state-of-the-art retrieval
```

> ⚠️ If you change `EMBED_MODEL` after indexing documents, delete the `vectorstore/` directory and re-index — embeddings from different models are not compatible.

### Performance notes

| Device | Embedding speed (500 chunks) | Notes |
|---|---|---|
| CPU (M2) | ~12 s | Baseline |
| MPS (M2) | ~3 s | ~4× faster |
| CUDA (RTX 3080) | ~1.5 s | ~8× faster |

Ollama's LLM inference runs independently through its own process and already uses Metal (MPS) on Apple Silicon automatically — no configuration needed.

---

## Supported File Types

| Extension | Content Extracted |
|---|---|
| `.pdf` | Text per page, embedded images (OCR), table detection |
| `.png` `.jpg` `.jpeg` `.tiff` `.bmp` | Full OCR via Tesseract |
| `.docx` | Paragraphs + tables |
| `.xlsx` | All sheets as text tables |
| `.csv` | Full table as text |
| `.txt` | Raw text |

---

## API Reference

The FastAPI backend is self-documented at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | System status: indexed docs, chunk count, model |
| `POST` | `/documents/upload` | Upload + index a file (multipart/form-data) |
| `DELETE` | `/documents/{filename}` | Remove a document from index and disk |
| `POST` | `/documents/reindex` | Force re-index all files in DATA_DIR |
| `POST` | `/query` | RAG query: `{"question": "...", "n_results": 5}` |
| `POST` | `/memory/clear` | Clear conversation memory |
| `GET` | `/memory/stats` | Token count, message count, summary status |
| `GET` | `/models` | List available Ollama models |

**Example query via curl:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the documents?", "n_results": 5}'
```

---

## Troubleshooting

### "Ollama error: model not found"
```bash
ollama pull llama3.2   # or whichever model is set in OLLAMA_MODEL
```

### "Tesseract not found"
Make sure Tesseract is installed and on your `PATH`:
```bash
which tesseract       # macOS/Linux
where tesseract       # Windows
```

### Embedding model download hangs
The first run downloads `all-MiniLM-L6-v2` (~90 MB). This is cached in `~/.cache/huggingface/`. If it times out, run:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### ChromaDB / vectorstore errors
Delete and recreate the vectorstore (you'll need to re-upload documents):
```bash
rm -rf ./vectorstore
python app.py
```

### Upload fails for large PDFs
Increase the FastAPI timeout or upload timeout in `app.py`:
```python
resp = api_post("/documents/upload", ..., timeout=300)  # 5 minutes
```

### "API unavailable" in the UI
Make sure the FastAPI backend is running:
```bash
curl http://localhost:8000/status
```
The backend starts automatically when running `app.py`. Run it directly if needed:
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000
```

### Memory fills up quickly
Reduce `MAX_HISTORY_TOKENS` in `utils/memory.py` (default: 2000), or click **🧹 Clear Memory** regularly.

---

## Project Structure

```
multimodal-rag/
│
├── app.py                      # Single entrypoint (local + HF Spaces) — detects env via SPACE_ID
├── frontend.py                 # Gradio UI: chat, document manager, sample questions
├── backend.py                  # FastAPI REST backend
├── Dockerfile                  # HF Spaces Docker build (inlines Ollama startup in CMD)
├── .dockerignore               # Excludes .venv/, data/, vectorstore/ from image
├── requirements.txt            # Python dependencies
├── .gitignore
├── README.md                   # HuggingFace Spaces YAML metadata + overview
├── _INSTRUCTIONS.md            # This file
│
├── data/                       # Uploaded documents (auto-created, gitignored)
├── vectorstore/                # ChromaDB persistent storage (auto-created, gitignored)
│
└── utils/
    ├── __init__.py
    ├── document_processor.py   # Multimodal extraction: PDF, images, DOCX, XLSX
    ├── vector_store.py         # ChromaDB manager + sentence-transformers embeddings
    ├── rag_engine.py           # RAG pipeline: retrieval → prompt → Ollama (OLLAMA_HOST aware)
    └── memory.py               # Sliding-window conversation memory with auto-summary
```

---

## License

MIT — free to use, modify, and deploy.
