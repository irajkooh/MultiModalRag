"""
Local development entrypoint.

Starts FastAPI (port 8000) in a background thread, then launches Gradio on
port 7860.  Also ensures Ollama is running locally and the required model is
pulled.

For HuggingFace Spaces deployment use hf_app.py (via startup.sh + Dockerfile).
"""
import threading
import time
import logging
import uvicorn

GREEN      = "\033[92m"
DARK_GREEN = "\033[32m"
RED        = "\033[91m"
YELLOW     = "\033[93m"
BLUE       = "\033[94m"  
RESET      = "\033[0m"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def run_api():
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        log_level="warning",
        reload=False,
    )


def wait_for_api(max_wait: int = 30):
    import requests
    for _ in range(max_wait):
        try:
            r = requests.get("http://localhost:8000/status", timeout=2)
            if r.status_code == 200:
                logger.info("API is ready.")
                return True
        except Exception:
            pass
        time.sleep(1)
    logger.warning("API did not become ready in time.")
    return False


def ensure_ollama(model: str = None):
    """Start Ollama daemon if not running, then pull the model if missing."""
    import subprocess, requests, os
    model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    # Check if Ollama is already responding
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
        subprocess.Popen(["ollama", "serve"],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        # Wait up to 15 s for it to come up
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

    # Check whether the model is already pulled
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


if __name__ == "__main__":
    import webbrowser
    import subprocess
    import time
    import os

    os.system('cls' if os.name == 'nt' else 'clear') # Clear terminal

    # Ensure Ollama is running and model is pulled before anything else
    ensure_ollama()

    subprocess.run(f"lsof -ti:{8000} | xargs kill -9", shell=True)
    subprocess.run(f"lsof -ti:{7860} | xargs kill -9", shell=True)
    time.sleep(1)  # Wait 1 second before starting Gradio

    # Start FastAPI backend in background thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    logger.info("Waiting for FastAPI to start...")
    wait_for_api()
    print(f"{YELLOW}FastAPI backend started on port 8000 --> open http://localhost:8000/docs{RESET}")
    # webbrowser.open("http://localhost:8000/docs")

    # Launch Gradio UI
    # Ensure an asyncio event loop exists in the main thread before building
    # the UI — Gradio's safe_get_lock() returns None (breaking the queue) if
    # asyncio.get_event_loop() raises RuntimeError (Python 3.10+ behaviour).
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    from app import build_ui, CUSTOM_CSS
    ui = build_ui()

    # Open Gradio UI in browser once the server is up
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
        css=CUSTOM_CSS,
    )
    
