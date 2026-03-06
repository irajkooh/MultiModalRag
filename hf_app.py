"""
HuggingFace Spaces entrypoint.

Starts FastAPI (port 8000, internal) in a background thread,
waits for it to be healthy, then launches Gradio on port 7860 (public).

This file intentionally omits the local-only helpers in main.py
(port killing, webbrowser.open, coloured terminal output).
"""
import threading
import time
import logging
import asyncio

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─── FastAPI background thread ────────────────────────────────────────────────

def run_api():
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        log_level="warning",
        reload=False,
    )


def wait_for_api(max_wait: int = 180) -> bool:
    import requests
    for i in range(max_wait):
        try:
            r = requests.get("http://localhost:8000/status", timeout=2)
            if r.status_code == 200:
                logger.info("FastAPI is ready.")
                return True
        except Exception:
            pass
        if i % 15 == 0 and i > 0:
            logger.info(f"Still waiting for FastAPI... ({i}s elapsed)")
        time.sleep(1)
    logger.warning("FastAPI did not become ready in time — continuing anyway.")
    return False


# ─── Start FastAPI ────────────────────────────────────────────────────────────

api_thread = threading.Thread(target=run_api, daemon=True)
api_thread.start()
logger.info("Waiting for FastAPI backend...")
wait_for_api()

# ─── Ensure asyncio event loop (Gradio queue requirement) ─────────────────────

try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# ─── Launch Gradio ────────────────────────────────────────────────────────────

from app import build_ui, CUSTOM_CSS  # noqa: E402

ui = build_ui()
ui.launch(
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True,
    share=False,
    css=CUSTOM_CSS,
)
