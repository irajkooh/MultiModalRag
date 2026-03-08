# ─── Base image ───────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ─── System packages ──────────────────────────────────────────────────────────
# poppler-utils  → pdf2image (PDF rendering)
# tesseract-ocr  → pytesseract (OCR)
# curl           → Ollama installer + health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    zstd \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# ─── Install Ollama ───────────────────────────────────────────────────────────
RUN curl -fsSL https://ollama.ai/install.sh | sh

# ─── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ─── Python dependencies ──────────────────────────────────────────────────────
# Install CPU-only torch first to avoid pulling the huge CUDA wheel.
# requirements.txt has `torch>=2.1.0`; pip will see 2.3.0+cpu already satisfies it.
RUN pip install --no-cache-dir \
    "torch==2.3.0+cpu" "torchvision==0.18.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Pre-download embedding model (cached before COPY . . so code changes don't bust this layer) ───
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ─── Application code ─────────────────────────────────────────────────────────
COPY . .

# ─── Runtime directories & permissions ───────────────────────────────────────
# /app/data        → uploaded documents
# /app/vectorstore → ChromaDB persistent store
# /app/.ollama     → Ollama model cache (survives within the same container)
RUN mkdir -p data vectorstore .ollama && \
    chmod -R 777 data vectorstore .ollama

# Tell Ollama where to store model weights
ENV OLLAMA_MODELS=/app/.ollama
# Default model — override via Space secret OLLAMA_MODEL
ENV OLLAMA_MODEL=llama3.2
ENV OLLAMA_NUM_CTX=8192

# ─── Pre-pull model at build time so restarts don't re-download ──────────────
RUN ollama serve & OLLAMA_PID=$! && \
    sleep 5 && \
    until curl -sf http://localhost:11434/api/version > /dev/null 2>&1; do sleep 1; done && \
    ollama pull ${OLLAMA_MODEL} && \
    kill $OLLAMA_PID 2>/dev/null || true

# ─── HuggingFace Spaces requires port 7860 ───────────────────────────────────
EXPOSE 7860

# ─── Startup: launch Ollama (model already cached) then start the app ────────
CMD ["/bin/bash", "-c", "\
ollama serve & \
echo '⏳ Waiting for Ollama to be ready...' && \
until curl -sf http://localhost:11434/api/version > /dev/null 2>&1; do sleep 1; done && \
echo '✅ Ollama ready (model pre-cached).' && \
exec python app.py\
"]
# backend.py = FastAPI REST API  |  frontend.py = Gradio UI  |  app.py = entrypoint
