# ─── Base image ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

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

# ─── HuggingFace Spaces requires port 7860 ───────────────────────────────────
EXPOSE 7860

# ─── Startup: launch Ollama, pull model, then start the app ──────────────────
CMD ["/bin/bash", "-c", "\
export OLLAMA_MODEL=${OLLAMA_MODEL:-llama3.2} && \
export OLLAMA_NUM_CTX=${OLLAMA_NUM_CTX:-8192} && \
echo '▶ Starting Ollama daemon...' && \
ollama serve & \
echo '⏳ Waiting for Ollama to be ready...' && \
until curl -sf http://localhost:11434/api/version > /dev/null 2>&1; do sleep 1; done && \
echo '✅ Ollama is ready.' && \
echo \"⬇  Pulling model: $OLLAMA_MODEL ...\" && \
ollama pull $OLLAMA_MODEL && \
echo '✅ Model ready.' && \
exec python app.py\
"]
# backend.py = FastAPI REST API  |  frontend.py = Gradio UI  |  app.py = entrypoint
