#!/bin/bash
# ─── HuggingFace Spaces startup ───────────────────────────────────────────────
# 1. Start the Ollama daemon
# 2. Wait until it is responsive
# 3. Pull the model (defaults to llama3.2, overridable via OLLAMA_MODEL secret)
# 4. Launch the Python application

set -e

export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
MODEL="$OLLAMA_MODEL"
export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-8192}"

echo "▶ Starting Ollama daemon..."
ollama serve &

echo "⏳ Waiting for Ollama to be ready..."
until curl -sf http://localhost:11434/api/version > /dev/null 2>&1; do
    sleep 1
done
echo "✅ Ollama is ready."

echo "⬇  Pulling model: $MODEL ..."
ollama pull "$MODEL"
echo "✅ Model ready."

echo "▶ Starting application..."
exec python hf_app.py
