"""
Modern Gradio UI for Multimodal RAG system.
Chat and document management with clean, responsive layout.
"""
import os
import time
import requests as _requests
import requests
import gradio as gr
from pathlib import Path


def _ping_self(url: str):
    """Lightweight GET to keep the HF Space from going to sleep."""
    try:
        _requests.get(url, timeout=10)
    except Exception:
        pass


def start_keep_alive_scheduler(space_url: str):
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger

    scheduler = BackgroundScheduler(timezone="UTC", daemon=True)
    scheduler.add_job(
        _ping_self,
        trigger=IntervalTrigger(minutes=20),
        args=[space_url],
        id="keep_alive_20m",
        replace_existing=True,
    )
    scheduler.start()
    return scheduler

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

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

def get_status():
  data = api_get("/status")
  if not data or "error" in data:
    # Always return a tuple of the right length
    return [], [], "⚠️ API unavailable", "Unknown", "Unknown"
  docs = data.get("documents", [])
  files = data.get("data_dir_files", [])
  chunks = data.get("total_chunks", 0)
  model = data.get("model", "unknown")
  device = data.get("device", "CPU")
  status_msg = f"✅ {len(docs)} document(s) indexed | {chunks} chunks"
  return docs, files, status_msg, model, device


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
      messages.append(f"<span style='color:#ff4d4f'>❌ {path.name}: {resp['error']}</span>")
    else:
      messages.append(f"<span style='color:#fff'>✅ {path.name}: {resp['message']}</span>")
  status_html = '<br>'.join(messages)
  return status_html, *refresh_ui()


def delete_document(filenames):
    if not filenames:
        return "Please select at least one document.", *refresh_ui()
    messages = []
    for filename in filenames:
        resp = api_delete(f"/documents/{filename}")
        if "error" in resp:
            messages.append(f"❌ {filename}: {resp['error']}")
        else:
            messages.append(f"🗑️ {resp['message']}")
    return "\n".join(messages), *refresh_ui()


def delete_all_embeddings():
    resp = api_delete("/documents")
    if "error" in resp:
        return f"❌ {resp['error']}", *refresh_ui()
    return f"🗑️ {resp['message']}", *refresh_ui()


def refresh_ui():
    docs, files, status_msg, model, _ = get_status()
    has_docs = len(docs) > 0 if docs is not None else False
    return gr.update(choices=docs or [], value=None), status_msg, gr.update(interactive=has_docs)


def chat_fn(message, history, n_results, temperature):
  """Send query to API, stream response token by token."""
  if not message.strip():
    yield history, ""
    return
  resp = api_post("/query", json={"question": message, "n_results": n_results, "temperature": temperature}, timeout=480)
  tokens_user = resp.get("tokens_user", 0)
  tokens_assistant = resp.get("tokens_assistant", 0)
  if "error" in resp:
    answer = f"⚠️ {resp['error']}"
  else:
    answer = resp.get("answer", "I DON'T KNOW")
    sources = resp.get("sources", [])
    if sources:
      answer += f"\n\n📄 *Sources: {', '.join(sources)}*"
  # Gradio 6.x expects: list of dicts with 'role' and 'content' keys
  history = list(history) if history else []
  if history and isinstance(history[0], tuple):
    new_hist = []
    for user, bot in history:
      if user is not None:
        new_hist.append({"role": "user", "content": user})
      if bot is not None:
        new_hist.append({"role": "assistant", "content": bot})
    history = new_hist
  history.append({"role": "user", "content": message})
  history.append({"role": "assistant", "content": ""})
  displayed = ""
  for char in answer:
    displayed += char
    history[-1]["content"] = displayed
    yield history, {"tokens_user": tokens_user, "tokens_assistant": tokens_assistant}
    time.sleep(0.005)
  yield history, {"tokens_user": tokens_user, "tokens_assistant": tokens_assistant}


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
    msgs = stats.get('message_count', 0)
    used = stats.get('total_tokens', 0)
    max_tok = stats.get('max_tokens', 2000)
    pct = int(used / max_tok * 100) if max_tok else 0
    bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
    return (
      f"💬 **{msgs}** messages · "
      f"🔢 **{used}** / {max_tok} tokens used ({pct}%)  `{bar}`"
    )


def get_last_answer(history):
    if not history:
        return ""
    last = history[-1]
    if isinstance(last, dict):
        return last.get("content", "") if last.get("role") == "assistant" else ""
    return last[1] or ""


def format_chat_history(history):
    if not history:
        return ""
    lines = []
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
        else:
            role = "User" if msg[0] else "Assistant"
            content = msg[0] or msg[1] or ""
        if content:
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


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
/* Ensure chat and documents tab panels have the same height */
.panel-box, #chat > div, #chat .panel-box {
  min-height: 480px !important;
  max-height: 600px !important;
  box-sizing: border-box !important;
}
/* Ensure chat and documents tab panels have the same height */
.panel-box, #chat > div, #chat .panel-box {
  min-height: 340px !important;
  max-height: 420px !important;
  box-sizing: border-box !important;
}
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

html, body, .gradio-container {
  min-height: unset !important;
  height: auto !important;
}

* { box-sizing: border-box; }

html, body {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow-x: hidden !important;
}

body, .gradio-container {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
}

.gradio-container {
  max-width: 1100px !important;
  margin: 0 auto !important;
  padding: 0 8px 0 8px !important;
  box-sizing: border-box !important;
  zoom: 0.82;
  transform-origin: top center;
}

/* Undo zoom effect on html/body so the scaled container is centered */
html, body {
  overflow-x: hidden !important;
}

/* Row/form gaps */
.gap { gap: 6px !important; padding: 0 !important; }
.form { gap: 4px !important; }
.block { overflow: visible !important; }

/* Tab bar */
.tab-nav {
  border-bottom: 2px solid var(--border) !important;
  margin-bottom: 4px !important;
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
  background: linear-gradient(135deg, #141720 0%, #1c2030 100%);
  border: 1px solid var(--border);
  border-bottom: 2px solid var(--accent);
  border-radius: var(--radius);
  padding: 6px 14px !important;
  margin-bottom: 6px !important;
  display: flex;
  align-items: baseline;
  gap: 8px !important;
  flex-wrap: wrap;
  max-width: 100%;
  overflow: hidden;
  box-sizing: border-box;
}
.app-header h1, .app-header p {
  max-width: 100%;
  overflow-wrap: break-word;
  word-break: break-word;
  white-space: normal;
  box-sizing: border-box;
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
  padding: 10px !important;
  overflow-y: auto !important;
  max-height: 42vh !important;
  min-height: 40px !important;
  height: 40px !important;
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

/* ...existing code... */
"""


# ─── Build UI ──────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="Multimodal RAG", theme=gr.themes.Soft(), css="""
    .main-col {max-width: 900px; margin: 0 auto;}
    .chatbot-wrap {background: #181c24; border-radius: 12px;}
    .gradio-container {background: #10131a;}
    .gr-button.primary-btn {background: #4fffb0; color: #181c24; font-weight: bold;}
    .gr-button.secondary-btn {background: #23263a; color: #4fffb0;}
    .gr-button.danger-btn {background: #ff6b6b; color: #fff;}
    .gr-chatbot .message.user {color: #ffe066; font-weight: bold;}
    .gr-chatbot .message.bot {color: #fff;}
    /* #token-stats color override removed to allow inline color */
    """) as demo:
        with gr.Row():
            with gr.Column(elem_classes="main-col"):
                docs, files, status_msg, model, device = get_status()
                gr.Markdown(f"""
                # 🧠 <span style='color:#3b82f6;'>Multimodal RAG</span>
                <span style='color:#7c5cfc;font-size:1.1em;'>Chat with your pdf, word, excel, csv, txt, image, chart, and table documents.</span>
                <br><span style='color:#ff9800;font-weight:bold;'>| LLM: {model} | Device: {device} |</span> <span style='color:#7c5cfc;font-weight:bold;'>Powered by Ollama + ChromaDB</span>
                """, elem_id="header")
        with gr.Tabs(selected=0) as tabs:
          with gr.TabItem("💬 Chat"):
            chatbot = gr.Chatbot(
              label="Chat",
              height=380,
              elem_classes="chatbot-wrap",
            )
            with gr.Row():
              msg_input = gr.Textbox(
                placeholder="Ask a question about your documents...",
                show_label=False,
                scale=5,
                lines=1,
                autofocus=True,
              )
              submit_btn = gr.Button("Ask →", elem_classes="primary-btn", scale=1)
            with gr.Row():
              n_results_slider = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="Top K (context chunks)",
                elem_id="topk-slider",
              )
              temperature_slider = gr.Slider(
                minimum=0.0, maximum=2.0, value=0.0, step=0.1,
                label="Temperature (0 = deterministic)",
                elem_id="temperature-slider",
              )
              with gr.Column():
                with gr.Row():
                    read_btn      = gr.Button("🔊 Read",      elem_classes="secondary-btn")
                    copy_btn      = gr.Button("📋 Copy Chat", elem_classes="secondary-btn")
                    clear_chat_btn = gr.Button("🗑 Clear Chat", elem_classes="danger-btn")
                token_stats_text = gr.Markdown(
                  value="<span style='color:#3b82f6; font-weight:600;'>Tokens sent: 0 &nbsp;&nbsp; Tokens received: 0</span>",
                    elem_id="token-stats",
                    visible=True,
                )
            memory_stats_text = gr.Markdown(value="", elem_id="memory-stats")
          with gr.TabItem("📁 Documents"):
            status_text = gr.Markdown(value="⏳ Loading...", elem_classes="status-bar")
            file_upload = gr.UploadButton(
              label="⬆ Upload & Index Files",
              file_count="multiple",
              file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".docx", ".xlsx", ".csv", ".txt"],
              elem_classes="primary-btn",
              elem_id="file-upload",
            )
            upload_status = gr.HTML(value="", elem_id="upload-status")
            doc_list = gr.CheckboxGroup(
              choices=[],
              label="Indexed Documents",
              elem_classes="doc-list",
              interactive=True,
            )
            with gr.Row():
              delete_btn = gr.Button("🗑 Remove selected", elem_classes="danger-btn")
              delete_all_btn = gr.Button("🗑 Remove ALL", elem_classes="danger-btn")
              refresh_btn = gr.Button("↻ Refresh list", elem_classes="secondary-btn")
            # Confirmation row for Remove ALL
            with gr.Row(visible=False) as confirm_row:
              gr.Markdown('<span style="font-size:0.95em;color:#f87171;">⚠️ Remove ALL embeddings? This cannot be undone.</span>')
              confirm_yes_btn = gr.Button("✔ Yes, remove all", elem_classes="danger-btn")
              confirm_no_btn = gr.Button("✖ Cancel", elem_classes="secondary-btn")
            delete_status = gr.Markdown(value="", elem_id="delete-status")

        def on_load():
          docs, files, status_msg, model, device = get_status()
          has_docs = len(docs) > 0 if docs is not None else False
          return (
            gr.update(choices=docs or [], value=None),
            status_msg,
            gr.update(interactive=has_docs),
          )
        tts_state  = gr.State("")
        copy_state = gr.State("")

        demo.load(
            fn=on_load,
            outputs=[doc_list, status_text, submit_btn],
        )
        def refresh_and_update():
          docs, files, status_msg, model, device = get_status()
          has_docs = len(docs) > 0 if docs is not None else False
          return (
            gr.update(choices=docs or [], value=None),
            status_msg,
            gr.update(interactive=has_docs),
          )
        file_upload.upload(
            fn=lambda files: (upload_files(files)[0], *refresh_and_update()),
            inputs=[file_upload],
            outputs=[upload_status, doc_list, status_text, submit_btn],
        )
        delete_btn.click(
          fn=lambda doc: (delete_document(doc)[0], *refresh_and_update()),
          inputs=[doc_list],
          outputs=[delete_status, doc_list, status_text, submit_btn],
        )
        # Show confirmation row when Remove ALL is clicked
        delete_all_btn.click(
          fn=lambda: gr.update(visible=True),
          outputs=[confirm_row],
        )
        # Confirm: execute delete, hide confirmation row
        confirm_yes_btn.click(
          fn=lambda: (gr.update(visible=False), delete_all_embeddings()[0], *refresh_and_update()),
          outputs=[confirm_row, delete_status, doc_list, status_text, submit_btn],
        )
        # Cancel: just hide the confirmation row
        confirm_no_btn.click(
          fn=lambda: gr.update(visible=False),
          outputs=[confirm_row],
        )
        refresh_btn.click(
          fn=refresh_and_update,
          outputs=[doc_list, status_text, submit_btn],
        )
        def on_submit(message, history, n, temp):
          history = history or []
          last_resp = None
          tokens_user = 0
          tokens_assistant = 0
          for updated_history, _ in chat_fn(message, history, n, temp):
            if isinstance(_, dict):
              tokens_user = _.get("tokens_user", 0)
              tokens_assistant = _.get("tokens_assistant", 0)
            last_resp = updated_history
            yield updated_history, "", f"<span style='color:#3b82f6; font-weight:600;'>Tokens sent: {tokens_user} &nbsp;&nbsp; Tokens received: {tokens_assistant}</span>"
          yield last_resp, "", f"<span style='color:#3b82f6; font-weight:600;'>Tokens sent: {tokens_user} &nbsp;&nbsp; Tokens received: {tokens_assistant}</span>"
        submit_btn.click(
          fn=on_submit,
          inputs=[msg_input, chatbot, n_results_slider, temperature_slider],
          outputs=[chatbot, msg_input, token_stats_text],
        )
        msg_input.submit(
          fn=on_submit,
          inputs=[msg_input, chatbot, n_results_slider, temperature_slider],
          outputs=[chatbot, msg_input, token_stats_text],
        )
        def on_clear_chat():
          status, history = clear_memory()
          if history and isinstance(history[0], tuple):
            new_hist = []
            for user, bot in history:
              if user is not None:
                new_hist.append({"role": "user", "content": user})
              if bot is not None:
                new_hist.append({"role": "assistant", "content": bot})
            history = new_hist
          return history, status, "<span style='color:#3b82f6; font-weight:600;'>Tokens sent: 0 &nbsp;&nbsp; Tokens received: 0</span>"

        read_btn.click(
          fn=get_last_answer,
          inputs=[chatbot],
          outputs=[tts_state],
        ).then(
          fn=None,
          inputs=[tts_state],
          js="(text) => { if(text && 'speechSynthesis' in window){ window.speechSynthesis.cancel(); window.speechSynthesis.speak(new SpeechSynthesisUtterance(text)); } }",
        )
        copy_btn.click(
          fn=format_chat_history,
          inputs=[chatbot],
          outputs=[copy_state],
        ).then(
          fn=None,
          inputs=[copy_state],
          js="(text) => { if(text) navigator.clipboard.writeText(text).catch(()=>{}); }",
        )
        clear_chat_btn.click(
          fn=on_clear_chat,
          outputs=[chatbot, memory_stats_text, token_stats_text],
        )
    demo.queue()
    return demo

if __name__ == "__main__":
    if os.environ.get("SPACE_ID"):
        _space_url = "https://irajkoohi-multimodalrag.hf.space"
        start_keep_alive_scheduler(_space_url)
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
