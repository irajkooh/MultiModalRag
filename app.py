"""
Gradio UI for the Multimodal RAG system.
Connects to the FastAPI backend for all operations.
"""
import os
import time
import requests
import gradio as gr
from pathlib import Path

GREEN      = "\033[92m"
DARK_GREEN = "\033[32m"
RED        = "\033[91m"
YELLOW     = "\033[93m"
BLUE       = "\033[94m"  
RESET      = "\033[0m"

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

# ─── API helpers ──────────────────────────────────────────────────────────────
def api_get(path: str, timeout: int = 10):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_post(path: str, json=None, files=None, timeout: int = 60, _retries: int = 3):
    import requests.exceptions
    for attempt in range(_retries):
        try:
            r = requests.post(f"{API_BASE}{path}", json=json, files=files, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError:
            if attempt < _retries - 1:
                time.sleep(3)
                continue
            return {"error": "⚠️ Backend not ready yet — please wait a moment and try again."}
        except Exception as e:
            return {"error": str(e)}
    return {"error": "⚠️ Backend unreachable after retries."}


def api_delete(path: str, timeout: int = 10):
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ─── UI Logic ──────────────────────────────────────────────────────────────────
def get_status():
    data = api_get("/status")
    if "error" in data:
        return [], [], "⚠️ API unavailable", "Unknown"
    docs = data.get("documents", [])
    files = data.get("data_dir_files", [])
    chunks = data.get("total_chunks", 0)
    model = data.get("model", "unknown")
    device = data.get("device", "CPU")
    status_msg = f"✅ {len(docs)} document(s) indexed | {chunks} chunks | Model: {model} | 🖥 {device}"
    return docs, files, status_msg, model


def upload_files(files):
    if not files:
        return "No files selected.", *refresh_ui()
    
    messages = []
    for file in files:
        path = Path(file.name)
        with open(path, "rb") as f:
            resp = api_post(
                "/documents/upload",
                files={"file": (path.name, f, "application/octet-stream")},
                timeout=120,
            )
        if "error" in resp:
            messages.append(f"❌ {path.name}: {resp['error']}")
        else:
            messages.append(f"✅ {path.name}: {resp['message']}")
    
    return "\n".join(messages), *refresh_ui()


def delete_document(filename: str):
    if not filename:
        return "Please select a document to delete.", *refresh_ui()
    resp = api_delete(f"/documents/{filename}")
    if "error" in resp:
        return f"❌ {resp['error']}", *refresh_ui()
    return f"🗑️ {resp['message']}", *refresh_ui()


def refresh_ui():
    docs, files, status_msg, model = get_status()
    has_docs = len(docs) > 0
    # Return: doc_list choices, status_msg, submit_interactive
    return gr.update(choices=docs, value=None), status_msg, gr.update(interactive=has_docs)


def chat_fn(message, history, n_results):
    """Send query to API, stream response token by token."""
    if not message.strip():
        yield history, ""
        return
    
    resp = api_post("/query", json={"question": message, "n_results": n_results}, timeout=480)
    if "error" in resp:
        answer = f"⚠️ {resp['error']}"
    else:
        answer = resp.get("answer", "I DON'T KNOW")
        sources = resp.get("sources", [])
        if sources:
            answer += f"\n\n📄 *Sources: {', '.join(sources)}*"

    history = list(history) if history else []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    # Simulate streaming for better UX
    displayed = ""
    for char in answer:
        displayed += char
        history[-1] = {"role": "assistant", "content": displayed}
        yield history, ""
        time.sleep(0.005)
    
    yield history, ""


def clear_memory():
    resp = api_post("/memory/clear")
    stats = api_get("/memory/stats")
    if "error" in resp:
        return f"⚠️ {resp['error']}", []
    msg = f"🧹 Memory cleared. {stats.get('message_count', 0)} messages in memory."
    return msg, []


def _status_html(text: str) -> str:
    """Wrap a status string in a styled HTML block with white text."""
    import html
    lines = html.escape(str(text)).replace("\n", "<br>")
    return (
        '<div style="background:#1c2030;color:#ffffff;padding:10px 14px;'
        'border:1px solid #2a2f40;border-radius:6px;'
        'font-family:\'IBM Plex Mono\',monospace;font-size:0.85rem;'
        'line-height:1.6;min-height:2.5em;">'
        + lines + "</div>"
    )


def get_memory_stats():
    stats = api_get("/memory/stats")
    if "error" in stats:
        return f"⚠️ {stats['error']}"
    return (
        f"💬 Messages: {stats.get('message_count', 0)} | "
        f"🔢 Tokens: {stats.get('total_tokens', 0)}/{stats.get('max_tokens', 2000)} | "
        f"📝 Ctx summarized: {'Yes' if stats.get('has_summary') else 'No'}"
    )


# ─── Sample Questions ─────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    # General document understanding
    ("📋 Summarize", "What is the main topic or purpose of the uploaded documents?"),
    ("📊 Tables", "What data or figures are presented in any tables within the documents?"),
    ("🖼 Images", "Describe any charts or images found in the documents."),
    # Content extraction
    ("🔑 Key points", "What are the key findings or conclusions mentioned in the documents?"),
    ("📅 Dates", "Are there any important dates or timelines referenced in the documents?"),
    ("👤 Entities", "What people, organizations, or places are mentioned in the documents?"),
    # Analytical
    ("💡 Recommendations", "What recommendations or action items are stated in the documents?"),
    ("⚠️ Risks", "Are there any risks, warnings, or caveats mentioned in the documents?"),
    ("💰 Numbers", "What numerical values, statistics, or metrics appear in the documents?"),
]


# ─── Custom CSS ───────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg: #0d0f14;
  --surface: #141720;
  --surface2: #1c2030;
  --border: #2a2f40;
  --accent: #4fffb0;
  --accent2: #7c5cfc;
  --accent3: #ff6b6b;
  --text: #e8eaf0;
  --text-muted: #8890a4;
  --radius: 10px;
  --font-head: 'Syne', sans-serif;
  --font-mono: 'IBM Plex Mono', monospace;
}

* { box-sizing: border-box; }

html, body {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
  margin: 0 !important;
  padding: 0 !important;
}

body, .gradio-container {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
}

.gradio-container {
  max-width: 860px !important;
  margin: 0 auto !important;
  padding: 8px 16px 24px 16px !important;
}

/* Row/form gaps */
.gap { gap: 6px !important; padding: 0 !important; }
.form { gap: 4px !important; }
.block { overflow: visible !important; }

/* Tab bar */
.tab-nav {
  border-bottom: 2px solid var(--border) !important;
  margin-bottom: 14px !important;
}
.tab-nav button {
  font-family: var(--font-head) !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  color: #ffffff !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 3px solid transparent !important;
  padding: 8px 22px !important;
  margin-bottom: -2px !important;
  transition: all 0.2s !important;
}
.tab-nav button.selected {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
}
.tab-nav button:hover:not(.selected) {
  color: var(--accent) !important;
}



/* Header */
.app-header {
  background: linear-gradient(135deg, #141720 0%, #1c2030 100%);
  border: 1px solid var(--border);
  border-bottom: 2px solid var(--accent);
  border-radius: var(--radius);
  padding: 6px 16px;
  margin-bottom: 6px;
  display: flex;
  align-items: baseline;
  gap: 12px;
  flex-wrap: wrap;
}

.app-header h1 {
  font-family: var(--font-head) !important;
  font-size: 1.2rem !important;
  font-weight: 800 !important;
  background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0 !important;
  letter-spacing: -0.5px;
  white-space: nowrap;
}

.app-header p {
  color: var(--text-muted) !important;
  margin: 0 !important;
  font-size: 0.78rem !important;
}

/* Panels */
.panel-box {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 8px !important;
  overflow-y: auto !important;
  max-height: calc(100vh - 80px) !important;
}

.panel-label {
  font-family: var(--font-head) !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  color: var(--accent) !important;
  margin-bottom: 6px !important;
  margin-top: 0 !important;
}

/* Buttons */
button.primary-btn {
  background: var(--accent) !important;
  color: #0d0f14 !important;
  border: none !important;
  font-family: var(--font-head) !important;
  font-weight: 700 !important;
  letter-spacing: 0.5px !important;
  border-radius: 6px !important;
  transition: all 0.2s !important;
}

button.primary-btn:hover:not(:disabled) {
  background: #2dffa0 !important;
  transform: translateY(-1px) !important;
}

button.primary-btn:disabled {
  opacity: 0.35 !important;
  cursor: not-allowed !important;
}

button.danger-btn {
  background: transparent !important;
  color: var(--accent3) !important;
  border: 1px solid var(--accent3) !important;
  font-family: var(--font-head) !important;
  font-weight: 700 !important;
  border-radius: 6px !important;
  transition: all 0.2s !important;
}

button.danger-btn:hover:not(:disabled) {
  background: var(--accent3) !important;
  color: #0d0f14 !important;
}

button.secondary-btn {
  background: transparent !important;
  color: var(--accent2) !important;
  border: 1px solid var(--accent2) !important;
  font-family: var(--font-head) !important;
  font-weight: 700 !important;
  border-radius: 6px !important;
  transition: all 0.2s !important;
}

button.secondary-btn:hover {
  background: var(--accent2) !important;
  color: white !important;
}

/* Inputs — Gradio Svelte components use many internal wrappers */
.gr-input, textarea, input[type="text"], .gr-box,
[data-testid="textbox"] textarea,
[data-testid="textbox"] input,
.block textarea, .block input[type="text"],
div.svelte-1f354aw textarea,
div.svelte-1f354aw input {
  background: #1c2030 !important;
  border: 1px solid var(--border) !important;
  color: #ffffff !important;
  border-radius: 6px !important;
  font-family: var(--font-mono) !important;
}

/* Read-only / non-interactive Textbox — Gradio renders these as divs not textareas */
[data-testid="textbox"] .prose,
[data-testid="textbox"] .output-class,
[data-testid="textbox"] > div > div,
[data-testid="textbox"] span,
[data-testid="textbox"] p,
[data-testid="textbox"] div {
  color: #ffffff !important;
  background: #1c2030 !important;
}

/* Textbox container/wrapper backgrounds */
[data-testid="textbox"],
[data-testid="textbox"] > div,
[data-testid="textbox"] .wrap,
.block .wrap,
.block .wrap-inner,
.block label span {
  color: #ffffff !important;
}

/* Placeholder text */
textarea::placeholder, input::placeholder {
  color: #8890a4 !important;
}

.gr-input:focus, textarea:focus {
  border-color: var(--accent) !important;
  outline: none !important;
  box-shadow: 0 0 0 2px rgba(79, 255, 176, 0.15) !important;
}

/* File upload widget */
.gr-file, [data-testid="file"],
[data-testid="file"] > div, [data-testid="file"] .wrap,
.file-preview, .file-preview-title,
.file-preview span, label.svelte-1b53jlb,
.upload-container, .upload-container span, .upload-container p,
.dndzone, .dndzone span, .dndzone p, .dndzone label,
.dndzone > div, [data-testid="file"] span,
[data-testid="file"] p,
/* uploaded file list items */
[data-testid="file"] .file-preview li,
[data-testid="file"] .file-preview a,
[data-testid="file"] .file-preview .file-name,
[data-testid="file"] ul li,
[data-testid="file"] ul li span,
[data-testid="file"] .upload-container *,
.file-preview *, [data-testid="file"] * {
  color: #ffffff !important;
}

/* Chatbot outer container + scrollable area */
.chatbot-wrap,
.chatbot-wrap > div,
.chatbot-wrap .wrap,
.chatbot-wrap .bubble-wrap,
.chatbot-wrap .message-wrap,
[data-testid="chatbot"],
[data-testid="chatbot"] > div,
[data-testid="chatbot"] .wrap {
  background: #1c2030 !important;
  border: none !important;
}

/* ── Nuclear border removal: strip every border/outline/shadow
   from every element inside the chatbot ────────────────────── */
#rag-chatbot *,
.chatbot-wrap *,
[data-testid="chatbot"] * {
  border: none !important;
  border-top: none !important;
  border-right: none !important;
  border-bottom: none !important;
  border-left: none !important;
  outline: none !important;
  box-shadow: none !important;
}

/* Chatbot bubble backgrounds & colours */
.chatbot-wrap [data-testid="bot"],
#rag-chatbot [data-testid="bot"] {
  background: #242a3d !important;
  border-radius: 10px !important;
  color: #ffffff !important;
}

.chatbot-wrap [data-testid="user"],
#rag-chatbot [data-testid="user"] {
  background: linear-gradient(135deg, #1a2545 0%, #1c1a35 100%) !important;
  border-radius: 10px !important;
  color: #ffed00 !important;
  font-weight: 700 !important;
}

.chatbot-wrap [data-testid="bot"] * {
  color: #ffffff !important;
}

.chatbot-wrap [data-testid="user"],
.chatbot-wrap [data-testid="user"] * {
  color: #ffed00 !important;
  font-weight: 700 !important;
}

/* Status bar */
.status-bar {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 6px;
  padding: 5px 10px;
  font-size: 0.78rem;
  color: var(--text-muted);
}

/* Memory stats markdown */
#memory-stats,
#memory-stats p,
#memory-stats span,
#memory-stats div {
  color: #ffffff !important;
  font-size: 0.82rem !important;
}

/* Document list */
.doc-list .gr-radio-row {
  background: var(--surface2) !important;
  border-radius: 6px !important;
  margin-bottom: 4px !important;
  padding: 8px 12px !important;
  border: 1px solid var(--border) !important;
  transition: border-color 0.2s !important;
}

.doc-list .gr-radio-row:hover {
  border-color: var(--accent2) !important;
}

/* Gradio 4.x radio items inside doc-list */
.doc-list [data-testid="radio"] label,
.doc-list [data-testid="radio"] span,
.doc-list fieldset label,
.doc-list fieldset span,
.doc-list .wrap label,
.doc-list .wrap span,
.doc-list label {
  background: var(--surface2) !important;
  color: var(--text) !important;
}

.doc-list [data-testid="radio"] label:hover,
.doc-list fieldset label:hover {
  background: #1e2240 !important;
  color: var(--accent) !important;
}

/* Tabs */
.tab-nav button {
  font-family: var(--font-head) !important;
  font-weight: 700 !important;
  letter-spacing: 0.5px !important;
  color: #ffffff !important;
  background: transparent !important;
  border-bottom: 2px solid transparent !important;
}

.tab-nav button.selected {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
}

/* ── Fixed-size Documents tab — prevent layout shifts ──────────── */

/* Lock file upload zone to a strict height regardless of selection state */
#file-upload,
#file-upload .wrap,
#file-upload > div,
[data-testid="file"]#file-upload {
  height: 120px !important;
  min-height: 120px !important;
  max-height: 120px !important;
  overflow: hidden !important;
}

/* Status boxes: strictly fixed height — no layout shift on content change */
#upload-status,
#delete-status {
  height: 62px !important;
  min-height: 62px !important;
  max-height: 62px !important;
  overflow: hidden !important;
  overflow-y: auto !important;
  flex-shrink: 0 !important;
  contain: strict !important;
}
#upload-status > *,
#delete-status > * {
  min-height: unset !important;
  max-height: 62px !important;
  overflow: hidden !important;
}
#upload-status > * > *,
#delete-status > * > * {
  min-height: unset !important;
  overflow: hidden !important;
}

/* Doc list: fixed height so adding entries doesn't push buttons down */
.doc-list {
  min-height: 120px !important;
  max-height: 200px !important;
  overflow-y: auto !important;
  flex-shrink: 0 !important;
}

/* Slider */
input[type="range"] {
  accent-color: var(--accent) !important;
}

/* Slider label and value */
[data-testid="slider"] label,
[data-testid="slider"] span,
[data-testid="slider"] .wrap span,
[data-testid="slider"] > div > span {
  color: #e8eaf0 !important;
  background: transparent !important;
}

/* Slider number input — must have explicit light text or it disappears */
[data-testid="slider"] input[type="number"] {
  color: #ffffff !important;
  background: #242a3d !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
}

[data-testid="slider"],
[data-testid="slider"] > div,
[data-testid="slider"] .wrap {
  background: var(--surface2) !important;
}

/* Chatbot processing status ("processing | 18.5s") */
#rag-chatbot .status,
#rag-chatbot .pending,
#rag-chatbot .generating,
#rag-chatbot [class*="status"],
#rag-chatbot [class*="pending"],
[data-testid="chatbot"] .status,
[data-testid="chatbot"] .pending,
[data-testid="chatbot"] [class*="status"],
[data-testid="chatbot"] [class*="pending"],
[data-testid="chatbot"] + div,
[data-testid="chatbot"] ~ div {
  color: #ffffff !important;
}

/* ── Smaller text inside chat bubbles ──────────────────────────── */
.chatbot-wrap [data-testid="bot"],
.chatbot-wrap [data-testid="user"] {
  font-size: 0.82rem !important;
  line-height: 1.55 !important;
}
.chatbot-wrap [data-testid="bot"] p,
.chatbot-wrap [data-testid="bot"] li,
.chatbot-wrap [data-testid="bot"] span,
.chatbot-wrap [data-testid="user"] p,
.chatbot-wrap [data-testid="user"] li,
.chatbot-wrap [data-testid="user"] span,
.chatbot-wrap .message,
.chatbot-wrap .message * {
  font-size: 0.82rem !important;
  line-height: 1.55 !important;
}

/* ── Global label / component-title contrast fixes ─────────────── */

/* All block labels visible against dark surface */
.block > label > span,
.block .label-wrap > span,
.block .label-wrap span,
.block label span,
fieldset legend,
fieldset > div > span,
.wrap > span,
.form > div > label span {
  color: #e8eaf0 !important;
  background: transparent !important;
}

/* Markdown blocks — ensure prose text is visible */
.prose, .prose p, .prose li, .prose span,
.md p, .md li, .md span,
[data-testid="markdown"] p,
[data-testid="markdown"] span,
[data-testid="markdown"] li,
[data-testid="markdown"] div {
  color: #e8eaf0 !important;
}

/* status-bar markdown inner text */
.status-bar p, .status-bar span, .status-bar div,
.status-bar .prose, .status-bar .prose p {
  color: var(--text-muted) !important;
  background: transparent !important;
}

/* Accordion label contrast */
.sample-accordion .label-wrap span,
.sample-accordion summary span,
.sample-accordion > button span {
  color: #c0c4d0 !important;
}

/* File upload drop-zone — force dark background everywhere */
[data-testid="file"],
[data-testid="file"] > div,
[data-testid="file"] .wrap,
[data-testid="file"] .upload-container,
[data-testid="file"] .dndzone,
[data-testid="file"] .file-preview,
[data-testid="file"] .pending {
  background: #1c2030 !important;
  color: #e8eaf0 !important;
}

/* Upload progress overlay — prevent white-on-white */
[data-testid="file"] .uploading,
[data-testid="file"] .progress,
[data-testid="file"] .progress-bar,
[data-testid="file"] .progress-level,
[data-testid="file"] .progress-level-inner,
[data-testid="file"] .loading,
[data-testid="file"] [class*="progress"],
[data-testid="file"] [class*="uploading"],
[data-testid="file"] [class*="loading"] {
  background: #1c2030 !important;
  color: #e8eaf0 !important;
}
[data-testid="file"] [class*="progress"] span,
[data-testid="file"] [class*="uploading"] span,
[data-testid="file"] [class*="loading"] span {
  color: #e8eaf0 !important;
}

/* File upload inner text (drop zone description & file names) */
[data-testid="file"] .wrap > span,
[data-testid="file"] .file-preview span,
[data-testid="file"] .file-name,
[data-testid="file"] button span,
[data-testid="file"] .upload-container span,
[data-testid="file"] .upload-container p,
[data-testid="file"] .upload-container div,
[data-testid="file"] .label-wrap span {
  color: #e8eaf0 !important;
  background: transparent !important;
}

/* Textbox label / wrapper bg fix */
[data-testid="textbox"] .label-wrap span {
  color: #e8eaf0 !important;
  background: transparent !important;
}

/* Radio group: label text must not be invisible on dark bg */
[data-testid="radio"] label span,
[data-testid="radio"] span.svelte-1p9xokt,
.doc-list [data-testid="radio"] input + span {
  color: #e8eaf0 !important;
  background: transparent !important;
}

/* Sample questions accordion */
.sample-accordion {
  margin: 2px 0 4px 0 !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  background: var(--surface2) !important;
}
.sample-accordion > .label-wrap {
  padding: 4px 10px !important;
  min-height: 28px !important;
}
.sample-accordion > .label-wrap span {
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
}

.sample-q-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
}

.sample-q-btn {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
  padding: 4px 8px !important;
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
  text-align: left !important;
  cursor: pointer !important;
  transition: all 0.18s ease !important;
  line-height: 1.3 !important;
  white-space: normal !important;
  height: auto !important;
  min-height: 36px !important;
}

.sample-q-btn:hover:not(:disabled) {
  border-color: var(--accent2) !important;
  background: #1e2240 !important;
  color: var(--accent) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 12px rgba(124, 92, 252, 0.2) !important;
}

.sample-q-btn:disabled {
  opacity: 0.3 !important;
  cursor: not-allowed !important;
}

.sample-q-btn .emoji {
  display: block;
  font-size: 0.7rem;
  color: var(--accent2);
  margin-bottom: 2px;
  font-style: normal;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent2); }
"""


# ─── Build UI ──────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="Multimodal RAG") as demo:
        
        # Header
        _model_name = os.environ.get("OLLAMA_MODEL", "llama3.2")
        gr.HTML(f"""
        <div class="app-header">
          <h1>🧠 MULTIMODAL RAG</h1><p>Query your PDFs, scanned images, tables and charts — grounded answers only. Powered by Ollama (<b>{_model_name}</b>) + ChromaDB.</p>
        </div>
        """)

        # Global inline overrides — beat Gradio 6 Svelte-scoped styles
        gr.HTML("""
        <style>
          /* Tab labels: always white; accent when selected */
          .tab-nav button,
          div[role="tablist"] button,
          button[role="tab"] {
            color: #ffffff !important;
            opacity: 1 !important;
          }
          .tab-nav button.selected,
          div[role="tablist"] button[aria-selected="true"],
          button[role="tab"][aria-selected="true"] {
            color: #4fffb0 !important;
            border-bottom-color: #4fffb0 !important;
          }
          .tab-nav button:hover,
          div[role="tablist"] button:hover,
          button[role="tab"]:hover {
            color: #4fffb0 !important;
          }
          /* File upload zone — dark background to prevent white-on-white */
          [data-testid="file"],
          [data-testid="file"] > div,
          [data-testid="file"] .wrap,
          [data-testid="file"] .upload-container,
          [data-testid="file"] .dndzone,
          [data-testid="file"] .uploading,
          [data-testid="file"] .pending,
          [data-testid="file"] [class*="progress"],
          [data-testid="file"] [class*="uploading"],
          [data-testid="file"] [class*="loading"] {
            background: #1c2030 !important;
            color: #e8eaf0 !important;
          }
          [data-testid="file"] * {
            color: #e8eaf0 !important;
          }
          /* Any leftover white surface elements */
          .gradio-container [class*="wrap"],
          .gradio-container .block > div {
            background-color: transparent;
          }
        </style>
        """)

        with gr.Tabs(selected="chat"):

            # ── TAB 1 — Documents ──────────────────────────────────────────
            with gr.Tab("📁 Documents", id="documents"):

                status_text = gr.Markdown(
                    value="⏳ Loading...",
                    elem_classes="status-bar",
                )

                gr.HTML('<div class="panel-label" style="margin-top:12px">⬆ Upload Documents</div>')
                file_upload = gr.File(
                    label="",
                    file_count="multiple",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff",
                                ".docx", ".xlsx", ".csv", ".txt"],
                    interactive=True,
                    height=120,
                    elem_id="file-upload",
                )
                upload_btn = gr.Button("⬆ Upload & Index", elem_classes="primary-btn", size="sm")
                gr.HTML('<div style="font-size:0.75rem;color:#8890a4;margin-top:6px;margin-bottom:2px;letter-spacing:1px;">UPLOAD STATUS</div>')
                upload_status = gr.HTML(value=_status_html(""), elem_id="upload-status")

                gr.HTML('<hr style="border-color:#2a2f40; margin:16px 0">')
                gr.HTML('<div class="panel-label">🗂 Indexed Documents</div>')

                doc_list = gr.Radio(
                    choices=[],
                    label="",
                    elem_classes="doc-list",
                    interactive=True,
                )

                with gr.Row():
                    delete_btn = gr.Button("🗑 Remove selected", elem_classes="danger-btn", size="sm")
                    refresh_btn = gr.Button("↻ Refresh list", elem_classes="secondary-btn", size="sm")

                gr.HTML('<div style="font-size:0.75rem;color:#8890a4;margin-top:8px;margin-bottom:2px;letter-spacing:1px;">ACTION</div>')
                delete_status = gr.HTML(value=_status_html(""), elem_id="delete-status")

            # ── TAB 2 — Chat ───────────────────────────────────────────────
            with gr.Tab("💬 Chat", id="chat"):

                sample_btns = []
                with gr.Accordion("✦ Sample questions", open=False, elem_classes="sample-accordion"):
                    with gr.Row(elem_classes="sample-q-grid"):
                        for col_idx in range(3):
                            with gr.Column(scale=1, min_width=0):
                                col_btns = []
                                for row_idx in range(3):
                                    idx = row_idx * 3 + col_idx
                                    if idx < len(SAMPLE_QUESTIONS):
                                        label, question = SAMPLE_QUESTIONS[idx]
                                        btn = gr.Button(
                                            f"{label}\n{question}",
                                            elem_classes="sample-q-btn",
                                            interactive=False,
                                            size="sm",
                                        )
                                        col_btns.append((btn, question))
                                sample_btns.extend(col_btns)

                chatbot = gr.Chatbot(
                    label="",
                    height=460,
                    layout="bubble",
                    show_label=False,
                    elem_id="rag-chatbot",
                    elem_classes="chatbot-wrap",
                    buttons=["copy", "copy_all"],
                )

                # Inline style override — wins over Gradio 6 Svelte-scoped styles
                gr.HTML('''
                <style>
                  /* strip every border/shadow from every chatbot descendant */
                  #rag-chatbot *, .chatbot-wrap * {
                    border: none !important;
                    border-top: none !important;
                    border-right: none !important;
                    border-bottom: none !important;
                    border-left: none !important;
                    outline: none !important;
                    box-shadow: none !important;
                  }
                  /* match bubble bg to chat window bg */
                  #rag-chatbot .bubble-wrap,
                  #rag-chatbot .message-wrap,
                  #rag-chatbot .message,
                  #rag-chatbot [class$="-bubble"],
                  #rag-chatbot [class*=" bubble"],
                  .chatbot-wrap .bubble-wrap,
                  .chatbot-wrap .message-wrap {
                    background: #1c2030 !important;
                  }
                  /* bot bubble */
                  #rag-chatbot [data-testid="bot"],
                  .chatbot-wrap [data-testid="bot"] {
                    background: #242a3d !important;
                    border-radius: 10px !important;
                    color: #ffffff !important;
                  }
                  #rag-chatbot [data-testid="bot"] *,
                  .chatbot-wrap [data-testid="bot"] * {
                    color: #ffffff !important;
                  }
                  /* user bubble */
                  #rag-chatbot [data-testid="user"],
                  .chatbot-wrap [data-testid="user"] {
                    background: linear-gradient(135deg,#1a2545 0%,#1c1a35 100%) !important;
                    border-radius: 10px !important;
                    color: #ffed00 !important;
                    font-weight: 700 !important;
                  }
                  #rag-chatbot [data-testid="user"] *,
                  .chatbot-wrap [data-testid="user"] * {
                    color: #ffed00 !important;
                    font-weight: 700 !important;
                  }
                </style>
                ''')

                # Auto-scroll
                gr.HTML('''
                <script>
                (function() {
                  var _scrollEl = null, _rafPending = false;
                  function findScrollable(root) {
                    if (!root) return null;
                    var all = root.querySelectorAll("*");
                    for (var i = 0; i < all.length; i++) {
                      var ov = window.getComputedStyle(all[i]).overflowY;
                      if ((ov==="auto"||ov==="scroll") && all[i].scrollHeight > all[i].clientHeight) return all[i];
                    }
                    return null;
                  }
                  function scheduleScroll() {
                    if (_rafPending) return; _rafPending = true;
                    requestAnimationFrame(function() { requestAnimationFrame(function() {
                      _rafPending = false;
                      var root = document.getElementById("rag-chatbot") || document.querySelector(".chatbot-wrap");
                      if (!_scrollEl || !document.contains(_scrollEl)) _scrollEl = findScrollable(root);
                      if (_scrollEl) _scrollEl.scrollTop = _scrollEl.scrollHeight;
                    }); });
                  }
                  function attach() {
                    var root = document.getElementById("rag-chatbot") || document.querySelector(".chatbot-wrap");
                    if (!root) { setTimeout(attach, 200); return; }
                    new MutationObserver(scheduleScroll).observe(root, {childList:true,subtree:true,characterData:true});
                    scheduleScroll();
                  }
                  if (document.readyState==="loading") document.addEventListener("DOMContentLoaded", attach);
                  else attach();
                })();
                </script>
                ''')

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask a question about your documents...",
                        label="",
                        scale=5,
                        lines=1,
                        autofocus=True,
                    )
                    submit_btn = gr.Button(
                        "Ask →",
                        scale=1,
                        elem_classes="primary-btn",
                        interactive=False,
                    )

                with gr.Row():
                    with gr.Column(scale=3, min_width=0):
                        gr.HTML('<div style="color:#ffffff;font-size:0.8rem;font-family:\'IBM Plex Mono\',monospace;margin-bottom:4px;">Top K — Context chunks to retrieve</div>')
                        n_results_slider = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="", show_label=False,
                            elem_id="topk-slider",
                        )
                    memory_stats_text = gr.Markdown(value="", elem_id="memory-stats")
                    clear_memory_btn = gr.Button(
                        "🧹 Clear Memory", scale=1, elem_classes="secondary-btn", size="sm"
                    )

        # ─── Event Handlers ───────────────────────────────────────────────
        all_sample_btn_components = [b for b, _ in sample_btns]

        def on_load():
            docs, files, status_msg, model = get_status()
            has_docs = len(docs) > 0
            mem_stat = get_memory_stats()
            sample_updates = [gr.update(interactive=has_docs) for _ in sample_btns]
            return (
                gr.update(choices=docs, value=None),
                status_msg,
                gr.update(interactive=has_docs),
                mem_stat,
                *sample_updates,
            )

        demo.load(
            fn=on_load,
            outputs=[doc_list, status_text, submit_btn, memory_stats_text, *all_sample_btn_components],
        )

        def refresh_and_update_samples():
            docs, files, status_msg, model = get_status()
            has_docs = len(docs) > 0
            mem_stat = get_memory_stats()
            sample_updates = [gr.update(interactive=has_docs) for _ in sample_btns]
            return (
                gr.update(choices=docs, value=None),
                status_msg,
                gr.update(interactive=has_docs),
                mem_stat,
                *sample_updates,
            )

        upload_btn.click(
            fn=lambda files: (
                _status_html(upload_files(files)[0]),
                *refresh_and_update_samples(),
            ),
            inputs=[file_upload],
            outputs=[upload_status, doc_list, status_text, submit_btn, memory_stats_text, *all_sample_btn_components],
        )

        delete_btn.click(
            fn=lambda doc: (
                _status_html(delete_document(doc)[0]),
                *refresh_and_update_samples(),
            ),
            inputs=[doc_list],
            outputs=[delete_status, doc_list, status_text, submit_btn, memory_stats_text, *all_sample_btn_components],
        )

        refresh_btn.click(
            fn=refresh_and_update_samples,
            outputs=[doc_list, status_text, submit_btn, memory_stats_text, *all_sample_btn_components],
        )

        def on_submit(message, history, n):
            history = history or []
            for updated_history, _ in chat_fn(message, history, n):
                yield updated_history, "", ""
            yield updated_history, "", get_memory_stats()

        submit_btn.click(
            fn=on_submit,
            inputs=[msg_input, chatbot, n_results_slider],
            outputs=[chatbot, msg_input, memory_stats_text],
        )

        msg_input.submit(
            fn=on_submit,
            inputs=[msg_input, chatbot, n_results_slider],
            outputs=[chatbot, msg_input, memory_stats_text],
        )

        # Wire each sample question button → prefill input and auto-submit
        def make_sample_handler(question_text):
            def handler(history, n):
                for updated_history, _ in chat_fn(question_text, history or [], n):
                    yield updated_history, "", ""
                yield updated_history, "", get_memory_stats()
            return handler

        for btn, question_text in sample_btns:
            btn.click(
                fn=make_sample_handler(question_text),
                inputs=[chatbot, n_results_slider],
                outputs=[chatbot, msg_input, memory_stats_text],
            )

        def on_clear_memory():
            status, history = clear_memory()
            return history, _status_html(status), get_memory_stats()

        clear_memory_btn.click(
            fn=on_clear_memory,
            outputs=[chatbot, delete_status, memory_stats_text],
        )

    demo.queue()
    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        css=CUSTOM_CSS,
    )
