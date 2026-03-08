"""
Application entrypoint (local dev + HuggingFace Spaces).

- Local dev      : ensures Ollama is running and the model is pulled, kills any
                   stale processes on ports 8000/7860, starts FastAPI in a
                   background thread, then launches Gradio on port 7860 and
                   opens the browser automatically.

                   Usage: python app.py

- HuggingFace    : same core flow without the local-only helpers (port killing,
                   browser open, coloured output, Ollama management).
                   Detection: HF Spaces sets the SPACE_ID environment variable.

                   Space : https://huggingface.co/spaces/irajkoohi/MultiModalRag
                   Deploy: git remote add space https://huggingface.co/spaces/irajkoohi/MultiModalRag
                           git push space main
"""
import os
import threading
import time
import logging
import asyncio
import uvicorn

IS_HF = bool(os.environ.get("SPACE_ID"))

GREEN      = "\033[92m"
DARK_GREEN = "\033[32m"
RED        = "\033[91m"
YELLOW     = "\033[93m"
BLUE       = "\033[94m"
RESET      = "\033[0m"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# ─── FastAPI ──────────────────────────────────────────────────────────────────

def run_api():
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        log_level="warning",
        reload=False,
    )


def wait_for_api(max_wait: int = None) -> bool:
    import requests
    if max_wait is None:
        max_wait = 180 if IS_HF else 30
    for i in range(max_wait):
        try:
            r = requests.get("http://localhost:8000/status", timeout=2)
            if r.status_code == 200:
                logger.info("FastAPI is ready.")
                return True
        except Exception:
            pass
        if IS_HF and i % 15 == 0 and i > 0:
            logger.info(f"Still waiting for FastAPI... ({i}s elapsed)")
        time.sleep(1)
    logger.warning("API did not become ready in time.")
    return False


# ─── Ollama (local dev only) ──────────────────────────────────────────────────

def ensure_ollama(model: str = None):
    """Start Ollama daemon if not running, then pull the model if missing."""
    import subprocess
    import requests
    model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    running = False
    for _ in range(3):
        try:
            if requests.get(f"{ollama_host}/api/version", timeout=2).status_code == 200:
                running = True
                break
        except Exception:
            pass
        time.sleep(1)

    if not running:
        print(f"{BLUE}Starting Ollama daemon...{RESET}")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(15):
            time.sleep(1)
            try:
                if requests.get(f"{ollama_host}/api/version", timeout=2).status_code == 200:
                    running = True
                    break
            except Exception:
                pass
        if not running:
            print(f"{RED}Could not start Ollama — make sure it is installed.{RESET}")
            return

    try:
        tags = requests.get(f"{ollama_host}/api/tags", timeout=5).json()
        pulled = [m["name"] for m in tags.get("models", [])]
        model_base = model.split(":")[0]
        if not any(model_base in p for p in pulled):
            print(f"{BLUE}Pulling model '{model}' (first run — this may take a few minutes)...{RESET}")
            subprocess.run(["ollama", "pull", model], check=False)
        else:
            print(f"{GREEN}Model '{model}' is ready.{RESET}")
    except Exception as e:
        logger.warning(f"Could not verify model: {e}")


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not IS_HF:
        import subprocess
        os.system('cls' if os.name == 'nt' else 'clear')
        ensure_ollama()
        subprocess.run(f"lsof -ti:{8000} | xargs kill -9", shell=True)
        subprocess.run(f"lsof -ti:{7860} | xargs kill -9", shell=True)
        time.sleep(1)

    # Start FastAPI backend in background thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    logger.info("Waiting for FastAPI to start...")
    wait_for_api()

    if not IS_HF:
        print(f"{YELLOW}FastAPI backend started on port 8000 --> open http://localhost:8000/docs{RESET}")

    # Ensure an asyncio event loop exists before building the Gradio UI.
    # Gradio's safe_get_lock() returns None (breaking the queue) if
    # asyncio.get_event_loop() raises RuntimeError (Python 3.10+ behaviour).
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    from frontend import build_ui
    ui = build_ui()

    if not IS_HF:
        import webbrowser

        def _open_browser():
            time.sleep(2)
            print(f"{YELLOW}Gradio UI started on port 7860 --> open http://localhost:7860{RESET}")
            webbrowser.open("http://localhost:7860")

        threading.Thread(target=_open_browser, daemon=True).start()

    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False,
    )
