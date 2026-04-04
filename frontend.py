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
_LLM_BACKEND = "HuggingFace" if os.environ.get("HF_TOKEN") else ("Groq" if os.environ.get("GROQ_API_KEY") else "Ollama")

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
    import urllib.parse
    if not filenames:
        return "Please select at least one document."
    messages = []
    for filename in filenames:
        resp = api_delete(f"/documents/{urllib.parse.quote(filename, safe='')}")
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
  """Send query to API, return complete answer (no character streaming)."""
  if not message.strip():
    return history, ""
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
  history.append({"role": "assistant", "content": answer})
  return history, {"tokens_user": tokens_user, "tokens_assistant": tokens_assistant}


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
    .main-col  { max-width: 900px; margin: 0 auto; }
    .chatbot-wrap { background: #181c24; border-radius: 12px; }
    .gradio-container { background: #10131a; }
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
                scale=8,
                lines=1,
                max_lines=1,
                autofocus=True,
              )
              submit_btn = gr.Button("Ask", elem_id="ask-btn", elem_classes="primary-btn", scale=1)
            with gr.Row():
              read_btn       = gr.Button("Read", elem_id="read-btn",       scale=2)
              copy_btn       = gr.Button("📋 Copy Chat", elem_id="copy-btn",       elem_classes=["btn-copy"],  scale=1)
              clear_chat_btn = gr.Button("🗑 Clear Chat", elem_id="clear-chat-btn", elem_classes=["btn-clear"], scale=1)
            with gr.Row():
              n_results_slider = gr.Slider(
                minimum=1, maximum=15, value=8, step=1,
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
                // ── 0. Inject CSS for user bubbles ──
                var _css = document.createElement('style');
                _css.textContent = ''
                    + '.chatbot-wrap .message-row:not(.bot-row) .message-bubble,'
                    + '.chatbot-wrap .message-row:not(.bot-row) .bubble-wrap > *,'
                    + '.chatbot-wrap [data-testid="user"] > div,'
                    + '.chatbot-wrap .role-user .message,'
                    + '.chatbot-wrap .user-row .message-bubble'
                    + ' { background:#3b82f6!important; color:#000!important; border-radius:12px!important; }'
                    + '.chatbot-wrap .message-row:not(.bot-row) .message-bubble p,'
                    + '.chatbot-wrap .message-row:not(.bot-row) .message-bubble span,'
                    + '.chatbot-wrap .message-row:not(.bot-row) .prose,'
                    + '.chatbot-wrap .message-row:not(.bot-row) .prose p,'
                    + '.chatbot-wrap [data-testid="user"] p,'
                    + '.chatbot-wrap .role-user p,'
                    + '.chatbot-wrap .user-row p,'
                    + '.chatbot-wrap .user-row span'
                    + ' { color:#000!important; }';
                document.head.appendChild(_css);

                // ── 1. Button gradient colors ──
                var READ_BLUE  = {bg:'linear-gradient(135deg,#2563eb 0%,#60a5fa 100%)', sh:'0 4px 16px rgba(37,99,235,0.55)'};
                var READ_ORANGE = {bg:'linear-gradient(135deg,#ea580c 0%,#fb923c 100%)', sh:'0 4px 18px rgba(234,88,12,0.6)'};
                var STYLE_RULES = [
                    { id:'ask-btn',       bg:'linear-gradient(135deg,#7c5cfc 0%,#a78bfa 100%)', sh:'0 4px 18px rgba(124,92,252,0.55)' },
                    { id:'file-upload',   bg:'linear-gradient(135deg,#0ea5e9 0%,#38bdf8 100%)', sh:'0 4px 18px rgba(14,165,233,0.55)' },
                    { id:'copy-btn',      bg:'linear-gradient(135deg,#059669 0%,#34d399 100%)', sh:'0 4px 16px rgba(5,150,105,0.5)' },
                    { id:'clear-chat-btn',bg:'linear-gradient(135deg,#dc2626 0%,#f87171 100%)', sh:'0 4px 16px rgba(220,38,38,0.5)' },
                    { id:'delete-btn',    bg:'linear-gradient(135deg,#ef4444 0%,#fca5a5 100%)', sh:'0 4px 16px rgba(239,68,68,0.5)' },
                    { id:'delete-all-btn',bg:'linear-gradient(135deg,#7f1d1d 0%,#b91c1c 100%)', sh:'0 4px 18px rgba(127,29,29,0.65)' },
                    { id:'confirm-yes-btn',bg:'linear-gradient(135deg,#7f1d1d 0%,#b91c1c 100%)',sh:'0 4px 18px rgba(127,29,29,0.65)' },
                    { id:'confirm-no-btn',bg:'linear-gradient(135deg,#374151 0%,#6b7280 100%)', sh:'0 2px 10px rgba(107,114,128,0.4)' },
                    { id:'refresh-btn',   bg:'linear-gradient(135deg,#4338ca 0%,#818cf8 100%)', sh:'0 4px 16px rgba(67,56,202,0.5)' },
                    { id:'add-url-btn',   bg:'linear-gradient(135deg,#0d9488 0%,#2dd4bf 100%)', sh:'0 4px 16px rgba(13,148,136,0.5)' },
                    { id:'read-btn',      bg:READ_BLUE.bg, sh:READ_BLUE.sh },
                ];
                function styleEl(el, bg, sh) {
                    el.style.setProperty('background',    bg,  'important');
                    el.style.setProperty('box-shadow',    sh,  'important');
                    el.style.setProperty('color',         '#fff', 'important');
                    el.style.setProperty('border',        'none', 'important');
                    el.style.setProperty('font-weight',   '700',  'important');
                    el.style.setProperty('border-radius', '8px',  'important');
                    el.style.setProperty('letter-spacing','0.4px','important');
                    el.style.setProperty('text-shadow',   '0 1px 3px rgba(0,0,0,0.35)', 'important');
                }
                var _applyingColors = false;
                function applyColors() {
                    if (_applyingColors) return;
                    _applyingColors = true;
                    for (var i = 0; i < STYLE_RULES.length; i++) {
                        var rule = STYLE_RULES[i];
                        var wrap = document.getElementById(rule.id);
                        if (!wrap) continue;
                        var el = wrap.querySelector('button') || wrap.querySelector('label') || wrap;
                        if (rule.id === 'read-btn') {
                            var c = window._ttsPlaying ? READ_ORANGE : READ_BLUE;
                            styleEl(el, c.bg, c.sh);
                            var want = window._ttsPlaying ? 'Stop' : 'Read';
                            if (el.textContent.trim() !== want) el.textContent = want;
                        } else {
                            styleEl(el, rule.bg, rule.sh);
                        }
                    }
                    _applyingColors = false;
                }
                setTimeout(applyColors, 150);
                setTimeout(applyColors, 700);
                setInterval(applyColors, 2000);
                var _mt = null;
                new MutationObserver(function() {
                    if (_mt) clearTimeout(_mt);
                    _mt = setTimeout(applyColors, 80);
                }).observe(document.body, { childList: true, subtree: true });

                // ── 2. Click press feedback ──
                document.addEventListener('mousedown', function(e) {
                    var btn = e.target.closest('button');
                    if (!btn) return;
                    btn.style.setProperty('transform', 'scale(0.93)', 'important');
                    btn.style.setProperty('opacity', '0.82', 'important');
                    function reset() {
                        btn.style.removeProperty('transform');
                        btn.style.removeProperty('opacity');
                        btn.removeEventListener('mouseup', reset);
                        btn.removeEventListener('mouseleave', reset);
                    }
                    btn.addEventListener('mouseup', reset);
                    btn.addEventListener('mouseleave', reset);
                }, true);

                // ── 3. Scroll to bottom — called once when answer is complete ──
                window._scrollChatToBottom = function() {
                    var chatEl = document.querySelector('.chatbot-wrap');
                    if (!chatEl) return;
                    var scrollEl = null;
                    var divs = chatEl.querySelectorAll('div');
                    for (var i = 0; i < divs.length; i++) {
                        if (divs[i].scrollHeight > divs[i].clientHeight + 10) scrollEl = divs[i];
                    }
                    if (scrollEl) {
                        setTimeout(function() { scrollEl.scrollTop = scrollEl.scrollHeight; }, 50);
                    }
                };

                // ── 4. TTS — event delegation on document (immune to re-renders) ──
                window._ttsText    = null;
                window._ttsPlaying = false;
                function _getReadBtn() {
                    var wrap = document.getElementById('read-btn');
                    if (!wrap) return null;
                    return wrap.querySelector('button') || wrap;
                }
                window._ttsSetBtn = function(playing) {
                    var btn = _getReadBtn();
                    if (!btn) return;
                    var c = playing ? READ_ORANGE : READ_BLUE;
                    styleEl(btn, c.bg, c.sh);
                    btn.textContent = playing ? 'Stop' : 'Read';
                };
                window._ttsToggle = function() {
                    if (!window.speechSynthesis) return;
                    if (window._ttsPlaying) {
                        window.speechSynthesis.cancel();
                        window._ttsPlaying = false;
                        window._ttsSetBtn(false);
                    } else if (window._ttsText) {
                        window.speechSynthesis.cancel();
                        var utt = new SpeechSynthesisUtterance(window._ttsText);
                        window._ttsPlaying = true;
                        window._ttsSetBtn(true);
                        utt.onend = function() {
                            window._ttsPlaying = false;
                            window._ttsSetBtn(false);
                        };
                        window.speechSynthesis.speak(utt);
                    }
                };
                // Event delegation — click anywhere, check if it's the read button
                document.addEventListener('click', function(e) {
                    var wrap = document.getElementById('read-btn');
                    if (!wrap) return;
                    if (wrap.contains(e.target)) {
                        if (window._ttsToggle) window._ttsToggle();
                    }
                }, true);
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
          updated_history, stats = chat_fn(message, history, n, temp)
          tokens_user = stats.get("tokens_user", 0) if isinstance(stats, dict) else 0
          tokens_assistant = stats.get("tokens_assistant", 0) if isinstance(stats, dict) else 0
          tts_text = _clean_for_tts(get_last_answer(updated_history)) if updated_history else ""
          tok_html = f"<span style='color:#3b82f6; font-weight:600;'>Tokens sent: {tokens_user} &nbsp;&nbsp; Tokens received: {tokens_assistant}</span>"
          return updated_history, "", tok_html, tts_text
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

        tts_audio_box.change(
          fn=None,
          inputs=[tts_audio_box],
          js="""(val) => {
            if (window.speechSynthesis && window.speechSynthesis.speaking)
                window.speechSynthesis.cancel();
            window._ttsPlaying = false;
            window._ttsText = val || null;
            if (window._ttsSetBtn) window._ttsSetBtn(false);
            if (window._scrollChatToBottom) window._scrollChatToBottom();
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
