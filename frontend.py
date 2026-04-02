"""
Modern Gradio UI for Multimodal RAG system.
Chat and document management with clean, responsive layout.
"""
import os
import time
import requests
import gradio as gr
from pathlib import Path


def _ping_self(url: str):
    """Lightweight GET to keep the HF Space from going to sleep."""
    try:
        requests.get(url, timeout=10)
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

API_BASE     = os.environ.get("API_BASE", "http://localhost:8000")
_LLM_BACKEND = "Groq" if os.environ.get("GROQ_API_KEY") else "Ollama"

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
  status_msg = (
      f'✅ <span style="color:#facc15;">'
      f'<span style="color:#4ade80;font-weight:bold;">{len(docs)}</span>'
      f' document(s) indexed | '
      f'<span style="color:#4ade80;font-weight:bold;">{chunks}</span>'
      f' chunks</span>'
  )
  return docs, files, status_msg, model, device


def upload_files(files):
  if not files:
    return "No files selected."
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
  return '<br>'.join(messages)


def delete_document(filenames):
    if not filenames:
        return "Please select at least one document."
    messages = []
    for filename in filenames:
        resp = api_delete(f"/documents/{filename}")
        if "error" in resp:
            messages.append(f"❌ {filename}: {resp['error']}")
        else:
            messages.append(f"🗑️ {resp['message']}")
    return "\n".join(messages)


def delete_all_embeddings():
    resp = api_delete("/documents")
    if "error" in resp:
        return f"❌ {resp['error']}"
    return f"🗑️ {resp['message']}"


def add_url(url: str) -> str:
    """
    Start a background crawl, then poll until done (or error).
    Returns an HTML status string.
    """
    url = url.strip()
    if not url:
        return "<span style='color:#f87171'>Please enter a URL.</span>"
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # Kick off the background crawl
    resp = api_post("/documents/url", json={"url": url}, timeout=30)
    if "error" in resp:
        return f"<span style='color:#ff4d4f'>❌ {resp['error']}</span>"

    # Poll until the crawl finishes (max 10 min)
    import urllib.parse
    encoded = urllib.parse.quote(url, safe="")
    deadline = time.time() + 600
    while time.time() < deadline:
        time.sleep(5)
        status = api_get(f"/documents/url/status?url={encoded}", timeout=10)
        if "error" in status:
            break
        if status.get("status") == "done":
            return f"<span style='color:#4ade80'>✅ {status['message']}</span>"
        if status.get("status") == "error":
            return f"<span style='color:#ff4d4f'>❌ {status.get('message', 'Crawl failed.')}</span>"

    return "<span style='color:#f87171'>⚠️ Crawl is taking longer than expected — refresh the document list in a moment.</span>"


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


import re as _re

def _extract_text(content):
    """Handle Gradio 6.x content: str or [{'text': '...', 'type': 'text'}, ...]"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item["text"] if isinstance(item, dict) and "text" in item else str(item)
            for item in content
        )
    return str(content) if content else ""

def get_last_answer(history):
    if not history:
        return ""
    last = history[-1]
    if isinstance(last, dict):
        return _extract_text(last.get("content", "")) if last.get("role") == "assistant" else ""
    return _extract_text(last[1]) if last[1] else ""


def format_chat_history(history):
    if not history:
        return ""
    lines = []
    for msg in history:
        if isinstance(msg, dict):
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = _extract_text(msg.get("content", ""))
        else:
            role = "User" if msg[0] else "Assistant"
            content = _extract_text(msg[0] or msg[1] or "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


_EMOJI_RE = _re.compile(
    "[\U0001F300-\U0001F9FF\U00002702-\U000027B0\U000024C2-\U0001F251]+",
    flags=_re.UNICODE,
)

def _clean_for_tts(text: str) -> str:
    text = _re.sub(r"\n\n📄 \*Sources:[^\n]*", "", text)   # remove sources line
    text = _re.sub(r"\*+([^*]*)\*+", r"\1", text)          # remove bold/italic
    text = _re.sub(r"`[^`]*`", "", text)                    # remove inline code
    text = _re.sub(r"#+\s", "", text)                       # remove headers
    text = _EMOJI_RE.sub("", text)                          # remove emojis
    return text.strip()



_copy_counter = [0]

def get_chat_for_copy(history):
    _copy_counter[0] += 1
    return f"{_copy_counter[0]}\n{format_chat_history(history)}"


_UI_THEME = gr.themes.Soft()
_UI_CSS = """
    .main-col {max-width: 900px; margin: 0 auto;}
    .chatbot-wrap {background: #181c24; border-radius: 12px;}
    .gradio-container {background: #10131a;}
    /* Read button: font-size:0 hides Svelte-managed text; ::before shows our label */
    #read-btn button {
        position: relative !important;
        font-size: 0 !important;
        background: linear-gradient(135deg,#2563eb 0%,#60a5fa 100%) !important;
        box-shadow: 0 4px 16px rgba(37,99,235,0.55) !important;
    }
    #read-btn button::before {
        content: '🔊 Read';
        font-size: 14px; font-weight: 700; color: #fff;
        position: absolute; left: 0; right: 0; top: 50%;
        transform: translateY(-50%);
        text-align: center; pointer-events: none;
    }
    #read-btn button[data-speaking="1"] {
        background: linear-gradient(135deg,#ea580c 0%,#fb923c 100%) !important;
        box-shadow: 0 4px 18px rgba(234,88,12,0.6) !important;
    }
    #read-btn button[data-speaking="1"]::before { content: '⏹ Stop'; }
"""


# ─── Build UI ──────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="Multimodal RAG") as demo:
        with gr.Row():
            with gr.Column(elem_classes="main-col"):
                docs, files, status_msg, model, device = get_status()
                gr.Markdown(f"""
                # 🧠 <span style='color:#3b82f6;'>Multimodal RAG</span>
                <span style='color:#7c5cfc;font-size:1.1em;'>Chat with your pdf, word, excel, csv, txt, image, chart, and table documents.</span>
                <br><span style='color:#ff9800;font-weight:bold;'>| LLM: {model} | Device: {device} |</span> <span style='color:#7c5cfc;font-weight:bold;'>Powered by {_LLM_BACKEND} + ChromaDB</span>
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
                max_lines=1,
                autofocus=True,
              )
              submit_btn = gr.Button("Ask →", elem_id="ask-btn", elem_classes="primary-btn", scale=1)
            with gr.Row():
              read_btn       = gr.Button("🔊 Read",      elem_id="read-btn",       elem_classes=["btn-read"],  scale=1)
              copy_btn       = gr.Button("📋 Copy Chat", elem_id="copy-btn",       elem_classes=["btn-copy"],  scale=1)
              clear_chat_btn = gr.Button("🗑 Clear Chat", elem_id="clear-chat-btn", elem_classes=["btn-clear"], scale=1)
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
            with gr.Row():
              url_input = gr.Textbox(
                placeholder="https://example.com  —  crawls up to 2 levels deep + linked PDFs",
                show_label=False,
                scale=5,
                elem_id="url-input",
              )
              add_url_btn = gr.Button("🌐 Add URL", scale=1, elem_id="add-url-btn")
            doc_list = gr.CheckboxGroup(
              choices=[],
              label="Indexed Documents",
              elem_classes="doc-list",
              interactive=True,
            )
            with gr.Row():
              delete_btn     = gr.Button("🗑 Remove selected", elem_id="delete-btn")
              delete_all_btn = gr.Button("🗑 Remove ALL",      elem_id="delete-all-btn")
              refresh_btn    = gr.Button("↻ Refresh list",    elem_id="refresh-btn")
            # Confirmation row for Remove ALL
            with gr.Row(visible=False) as confirm_row:
              gr.Markdown('<span style="font-size:0.95em;color:#f87171;">⚠️ Remove ALL embeddings? This cannot be undone.</span>')
              confirm_yes_btn = gr.Button("✔ Yes, remove all", elem_id="confirm-yes-btn")
              confirm_no_btn  = gr.Button("✖ Cancel",          elem_id="confirm-no-btn")
            delete_status = gr.Markdown(value="", elem_id="delete-status")

        tts_audio_box= gr.Textbox(value="", visible=False, elem_id="tts-ready-box")
        copy_box     = gr.Textbox(value="", visible=False, elem_id="copy-box")

        def refresh_and_update():
          docs, files, status_msg, model, device = get_status()
          return (
            gr.update(choices=docs or [], value=None),
            status_msg,
            gr.update(interactive=True),
          )

        demo.load(
            fn=refresh_and_update,
            outputs=[doc_list, status_text, submit_btn],
        )
        # Unlock Web Speech API for mobile (iOS Safari blocks speechSynthesis
        # from async callbacks unless speak() is called once in a direct user gesture first)
        _JS_UNLOCK_TTS = """() => {
            function unlock() {
                if (!window.speechSynthesis) return;
                var u = new SpeechSynthesisUtterance('');
                window.speechSynthesis.speak(u);
                window.speechSynthesis.cancel();
            }
            document.addEventListener('click',    unlock, {once: true});
            document.addEventListener('touchend', unlock, {once: true});
        }"""
        demo.load(fn=None, js=_JS_UNLOCK_TTS)
        demo.load(
            fn=None,
            js="""() => {
                // 1. Hover CSS (inline styles handle base colors; CSS handles hover)
                const s = document.createElement('style');
                s.textContent = `
                    #read-btn button:hover       { background:#2563eb!important; }
                    #copy-btn button:hover       { background:#059669!important; }
                    #clear-chat-btn button:hover { background:#dc2626!important; }
                    #delete-btn button:hover     { background:#dc2626!important; }
                    #delete-all-btn button:hover { background:#991b1b!important; }
                    #refresh-btn button:hover    { background:#4f46e5!important; }
                    button { transition: transform .08s ease, opacity .08s ease !important; }
                `;
                document.head.appendChild(s);

                // 2. Artistic gradient + glow styles per button
                const STYLE_RULES = [
                    {
                        match: t => t === 'Ask →',
                        bg:  'linear-gradient(135deg,#7c5cfc 0%,#a78bfa 100%)',
                        sh:  '0 4px 18px rgba(124,92,252,0.55)',
                    },
                    {
                        match: t => t.includes('Upload') && t.includes('Files'),
                        bg:  'linear-gradient(135deg,#0ea5e9 0%,#38bdf8 100%)',
                        sh:  '0 4px 18px rgba(14,165,233,0.55)',
                    },
                    {
                        match: t => t.includes('Copy Chat'),
                        bg:  'linear-gradient(135deg,#059669 0%,#34d399 100%)',
                        sh:  '0 4px 16px rgba(5,150,105,0.5)',
                    },
                    {
                        match: t => t.includes('Clear Chat'),
                        bg:  'linear-gradient(135deg,#dc2626 0%,#f87171 100%)',
                        sh:  '0 4px 16px rgba(220,38,38,0.5)',
                    },
                    {
                        match: t => t.includes('Remove') && t.includes('selected'),
                        bg:  'linear-gradient(135deg,#ef4444 0%,#fca5a5 100%)',
                        sh:  '0 4px 16px rgba(239,68,68,0.5)',
                    },
                    {
                        match: t => t.includes('Remove ALL') || t.includes('Yes, remove all'),
                        bg:  'linear-gradient(135deg,#7f1d1d 0%,#b91c1c 100%)',
                        sh:  '0 4px 18px rgba(127,29,29,0.65)',
                    },
                    {
                        match: t => t.includes('Refresh'),
                        bg:  'linear-gradient(135deg,#4338ca 0%,#818cf8 100%)',
                        sh:  '0 4px 16px rgba(67,56,202,0.5)',
                    },
                    {
                        match: t => t.includes('Add URL'),
                        bg:  'linear-gradient(135deg,#0d9488 0%,#2dd4bf 100%)',
                        sh:  '0 4px 16px rgba(13,148,136,0.5)',
                    },
                    {
                        match: t => t.includes('Cancel'),
                        bg:  'linear-gradient(135deg,#374151 0%,#6b7280 100%)',
                        sh:  '0 2px 10px rgba(107,114,128,0.4)',
                    },
                ];
                function styleEl(el, rule) {
                    el.style.setProperty('background',   rule.bg, 'important');
                    el.style.setProperty('box-shadow',   rule.sh, 'important');
                    el.style.setProperty('color',        '#fff',  'important');
                    el.style.setProperty('border',       'none',  'important');
                    el.style.setProperty('font-weight',  '700',   'important');
                    el.style.setProperty('border-radius','8px',   'important');
                    el.style.setProperty('letter-spacing','0.4px','important');
                    el.style.setProperty('text-shadow',  '0 1px 3px rgba(0,0,0,0.35)', 'important');
                }
                function applyColors() {
                    document.querySelectorAll('button').forEach(el => {
                        if (el.closest('#read-btn')) return; // fully managed by CSS + JS
                        const text = el.textContent.trim();
                        for (const rule of STYLE_RULES) {
                            if (rule.match(text)) { styleEl(el, rule); break; }
                        }
                    });
                    // UploadButton may render as <label> — handle by ID
                    const upWrap = document.getElementById('file-upload');
                    if (upWrap) {
                        const upEl = (upWrap.tagName === 'BUTTON' || upWrap.tagName === 'LABEL')
                            ? upWrap : upWrap.querySelector('button, label');
                        if (upEl) styleEl(upEl, {
                            bg: 'linear-gradient(135deg,#0ea5e9 0%,#38bdf8 100%)',
                            sh: '0 4px 18px rgba(14,165,233,0.55)',
                        });
                    }
                }
                setTimeout(applyColors, 150);
                setTimeout(applyColors, 700);
                setInterval(applyColors, 2000);  // permanent safety net

                // 3. MutationObserver: re-apply colors immediately after any Gradio re-render
                let _t = null;
                const obs = new MutationObserver(() => {
                    if (_t) clearTimeout(_t);
                    _t = setTimeout(applyColors, 50);
                });
                obs.observe(document.body, { childList: true, subtree: true });

                // 4. Click feedback via mousedown/mouseup (CSS :active unreliable in Svelte)
                document.addEventListener('mousedown', (e) => {
                    const btn = e.target.closest('button');
                    if (!btn) return;
                    btn.style.setProperty('transform', 'scale(0.93)', 'important');
                    btn.style.setProperty('opacity', '0.82', 'important');
                    const reset = () => {
                        btn.style.removeProperty('transform');
                        btn.style.removeProperty('opacity');
                        btn.removeEventListener('mouseup', reset);
                        btn.removeEventListener('mouseleave', reset);
                    };
                    btn.addEventListener('mouseup', reset);
                    btn.addEventListener('mouseleave', reset);
                }, true);

                // 5. Auto-scroll chatbot to bottom as tokens stream in
                function attachChatScroller() {
                    // Gradio renders the scrollable chatbot as a div with overflow-y:auto/scroll
                    const chatWrap = document.querySelector('.chatbot-wrap .overflow-y-auto')
                                  || document.querySelector('.chatbot-wrap [data-testid="bot"]')
                                  || document.querySelector('.chatbot-wrap');
                    if (!chatWrap) return false;
                    const chatObs = new MutationObserver(() => {
                        chatWrap.scrollTop = chatWrap.scrollHeight;
                    });
                    chatObs.observe(chatWrap, { childList: true, subtree: true, characterData: true });
                    return true;
                }
                // Try immediately, then retry until the chatbot is rendered
                if (!attachChatScroller()) {
                    let tries = 0;
                    const t = setInterval(() => {
                        if (attachChatScroller() || ++tries > 20) clearInterval(t);
                    }, 300);
                }

                // 6. TTS state — used by read_btn pure-JS click handler (read_btn.click fn=None)
                window._ttsText    = null;   // clean text of latest answer
                window._ttsPlaying = false;  // true while speechSynthesis is speaking
            }"""
        )
        file_upload.upload(
            fn=lambda files: (upload_files(files), *refresh_and_update()),
            inputs=[file_upload],
            outputs=[upload_status, doc_list, status_text, submit_btn],
        )
        add_url_btn.click(
            fn=lambda url: (add_url(url), *refresh_and_update(), gr.update(value="")),
            inputs=[url_input],
            outputs=[upload_status, doc_list, status_text, submit_btn, url_input],
        )
        url_input.submit(
            fn=lambda url: (add_url(url), *refresh_and_update(), gr.update(value="")),
            inputs=[url_input],
            outputs=[upload_status, doc_list, status_text, submit_btn, url_input],
        )
        delete_btn.click(
          fn=lambda doc: (delete_document(doc), *refresh_and_update()),
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
          fn=lambda: (gr.update(visible=False), delete_all_embeddings(), *refresh_and_update()),
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
          first_yield = True
          for updated_history, _ in chat_fn(message, history, n, temp):
            if isinstance(_, dict):
              tokens_user = _.get("tokens_user", 0)
              tokens_assistant = _.get("tokens_assistant", 0)
            last_resp = updated_history
            tok_html = f"<span style='color:#3b82f6; font-weight:600;'>Tokens sent: {tokens_user} &nbsp;&nbsp; Tokens received: {tokens_assistant}</span>"
            if first_yield:
              first_yield = False
              yield updated_history, "", tok_html, ""  # clear _ttsText for new question
            else:
              yield updated_history, "", tok_html, gr.update()
          # Send clean text immediately — speechSynthesis in browser handles it, no server TTS delay
          tts_text = _clean_for_tts(get_last_answer(last_resp)) if last_resp else ""
          tok_html = f"<span style='color:#3b82f6; font-weight:600;'>Tokens sent: {tokens_user} &nbsp;&nbsp; Tokens received: {tokens_assistant}</span>"
          yield last_resp, "", tok_html, tts_text
        submit_btn.click(
          fn=on_submit,
          inputs=[msg_input, chatbot, n_results_slider, temperature_slider],
          outputs=[chatbot, msg_input, token_stats_text, tts_audio_box],
        )
        msg_input.submit(
          fn=on_submit,
          inputs=[msg_input, chatbot, n_results_slider, temperature_slider],
          outputs=[chatbot, msg_input, token_stats_text, tts_audio_box],
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

        # Pure JS read button — no Python roundtrip, instant response on click/tap.
        # fn=None means Gradio calls the JS synchronously in the click event (user gesture),
        # so speechSynthesis.speak/cancel work on iOS Safari without any unlock tricks.
        _READ_BTN_JS = """() => {
            if (!window.speechSynthesis) return;
            const btn = document.getElementById('read-btn')?.querySelector('button');
            if (window._ttsPlaying) {
                // STOP — delete data-speaking → CSS switches back to blue "🔊 Read"
                window.speechSynthesis.cancel();
                window._ttsPlaying = false;
                if (btn) delete btn.dataset.speaking;
            } else if (window._ttsText) {
                // START — set data-speaking="1" → CSS switches to orange "⏹ Stop"
                window.speechSynthesis.cancel();
                const utt = new SpeechSynthesisUtterance(window._ttsText);
                window._ttsPlaying = true;
                if (btn) btn.dataset.speaking = '1';
                utt.onend = () => {
                    window._ttsPlaying = false;
                    const eb = document.getElementById('read-btn')?.querySelector('button');
                    if (eb) delete eb.dataset.speaking;
                };
                window.speechSynthesis.speak(utt);
            }
        }"""
        read_btn.click(fn=None, js=_READ_BTN_JS)
        tts_audio_box.change(
          fn=None,
          inputs=[tts_audio_box],
          js="""(val) => {
            // New answer arrived — stop any reading, store text, reset to "🔊 Read".
            if (window.speechSynthesis && window.speechSynthesis.speaking)
                window.speechSynthesis.cancel();
            window._ttsPlaying = false;
            window._ttsText = val || null;
            const rb = document.getElementById('read-btn')?.querySelector('button');
            if (rb) delete rb.dataset.speaking; // CSS handles blue color automatically
          }""",
        )
        copy_btn.click(fn=get_chat_for_copy, inputs=[chatbot], outputs=[copy_box])
        copy_box.change(
          fn=None,
          inputs=[copy_box],
          js="""(val) => {
            const text = val.split('\\n').slice(1).join('\\n');
            if (text.trim()) navigator.clipboard.writeText(text).catch(() => {});
          }""",
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
    ui.launch(server_name="0.0.0.0", server_port=7860, show_error=True,
              theme=_UI_THEME, css=_UI_CSS)
