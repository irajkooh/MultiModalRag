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
            messages.append(f"❌ {path.name}: {resp['error']}")
        else:
            messages.append(f"✅ {path.name}: {resp['message']}")
    
    return "\n".join(messages), *refresh_ui()


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
    has_docs = len(docs) > 0
    # Return: doc_list choices, status_msg, submit_interactive
    return gr.update(choices=docs, value=None), status_msg, gr.update(interactive=has_docs)


def chat_fn(message, history, n_results, temperature):
    """Send query to API, stream response token by token."""
    if not message.strip():
        yield history, ""
        return
    
    resp = api_post("/query", json={"question": message, "n_results": n_results, "temperature": temperature}, timeout=480)
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
    msgs = stats.get('message_count', 0)
    used = stats.get('total_tokens', 0)
    max_tok = stats.get('max_tokens', 2000)
    pct = int(used / max_tok * 100) if max_tok else 0
    bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
    return (
      f"💬 **{msgs}** messages · "
      f"🔢 **{used}** / {max_tok} tokens used ({pct}%)  `{bar}`"
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


# ─── Build UI ──────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="Multimodal RAG") as demo:
        
        # Header (dynamic — device info injected on load)
        _model_name = os.environ.get("OLLAMA_MODEL", "llama3.2")
        def _header_html(model, device=""):
                dev = f" | 🖥 {device}" if device else ""
                return (f'<div class="app-header">'
                    f'<h1>🧠 MULTIMODAL RAG</h1>'
                    f'<p>Query your PDFs, scanned images, tables and charts — grounded answers only. '
                    f'Powered by Ollama (<b style="color:#000000;font-weight:bold">{model}</b>) + ChromaDB{dev}.</p>'
                    f'</div>')
        header_html = gr.HTML(value=_header_html(_model_name))

        # ...existing code...


        with gr.Tabs(selected="chat"):
            # ── TAB 1 — Documents ──────────────────────────────────────────
            with gr.Tab("📁 Documents", id="documents"):

                status_text = gr.Markdown(
                    value="⏳ Loading...",
                    elem_classes="status-bar",
                )

                gr.HTML('<div class="panel-label" style="margin-top:2px;margin-bottom:2px;">⬆ Upload Documents</div>')
                file_upload = gr.UploadButton(
                    label="⬆ Click or Drop Files to Upload & Index",
                    file_count="multiple",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff",
                                ".docx", ".xlsx", ".csv", ".txt"],
                    elem_classes="primary-btn",
                    elem_id="file-upload",
                    size="sm",
                )
                gr.HTML('<div style="font-size:0.75rem;color:#8890a4;margin-top:2px;margin-bottom:2px;letter-spacing:1px;">UPLOAD STATUS</div>')
                upload_status = gr.HTML(value=_status_html(""), elem_id="upload-status")

                gr.HTML('<hr style="border-color:#2a2f40; margin:6px 0">')
                gr.HTML('<div class="panel-label" style="margin-bottom:2px;">🗂 Indexed Documents</div>')

                doc_list = gr.CheckboxGroup(
                    choices=[],
                    label="",
                    elem_classes="doc-list",
                    interactive=True,
                )

                with gr.Row():
                    delete_btn = gr.Button("🗑 Remove selected", elem_classes="danger-btn", size="sm")
                    delete_all_btn = gr.Button("🗑 Remove ALL", elem_classes="danger-btn", size="sm")
                    refresh_btn = gr.Button("↻ Refresh list", elem_classes="secondary-btn", size="sm")

                with gr.Row(visible=False) as confirm_row:
                    gr.HTML('<span style="font-size:0.82rem;color:#f87171;align-self:center;">⚠️ Remove ALL embeddings?</span>')
                    confirm_yes_btn = gr.Button("✔ Yes, remove all", elem_classes="danger-btn", size="sm")
                    confirm_no_btn = gr.Button("✖ Cancel", elem_classes="secondary-btn", size="sm")

                gr.HTML('<div style="font-size:0.75rem;color:#8890a4;margin-top:2px;margin-bottom:2px;letter-spacing:1px;">ACTION</div>')
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
                    height=240,
                    layout="bubble",
                    show_label=False,
                    elem_id="rag-chatbot",
                    elem_classes="chatbot-wrap",
                    buttons=["copy", "copy_all"],
                )


                # Auto-scroll to bottom on every new message
                gr.HTML('''
                <script>
                (function() {
                  function scrollBottom() {
                    // Try the chatbot element itself first
                    var candidates = [
                      document.getElementById("rag-chatbot"),
                      document.querySelector(".chatbot-wrap"),
                      document.querySelector("#rag-chatbot .bubble-wrap"),
                      document.querySelector("#rag-chatbot .messages"),
                      document.querySelector("#rag-chatbot > div"),
                    ];
                    for (var i = 0; i < candidates.length; i++) {
                      var el = candidates[i];
                      if (!el) continue;
                      // Try the element itself
                      var ov = window.getComputedStyle(el).overflowY;
                      if (ov === "auto" || ov === "scroll") {
                        el.scrollTop = el.scrollHeight; continue;
                      }
                      // Try all its descendants
                      var all = el.querySelectorAll("*");
                      for (var j = 0; j < all.length; j++) {
                        var dov = window.getComputedStyle(all[j]).overflowY;
                        if ((dov === "auto" || dov === "scroll") && all[j].scrollHeight > all[j].clientHeight + 4) {
                          all[j].scrollTop = all[j].scrollHeight;
                        }
                      }
                    }
                  }
                  function attach() {
                    var root = document.getElementById("rag-chatbot") || document.querySelector(".chatbot-wrap");
                    if (!root) { setTimeout(attach, 300); return; }
                    // Watch for any DOM change inside the chatbot
                    new MutationObserver(function() {
                      requestAnimationFrame(function() { requestAnimationFrame(scrollBottom); });
                    }).observe(root, {childList:true, subtree:true, characterData:true});
                  }
                  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", attach);
                  else attach();
                })();
                </script>
                ''')

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask a question about your documents...",
                        show_label=False,
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
                        gr.HTML('<div style="color:#8890a4;font-size:0.72rem;font-family:\'IBM Plex Mono\',monospace;margin-bottom:0px;">Top K &nbsp;<span style="opacity:0.6">— context chunks</span></div>')
                        n_results_slider = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="", show_label=False,
                            elem_id="topk-slider",
                        )
                    with gr.Column(scale=3, min_width=0):
                        gr.HTML('<div style="color:#8890a4;font-size:0.72rem;font-family:\'IBM Plex Mono\',monospace;margin-bottom:0px;">Temperature &nbsp;<span style="opacity:0.6">— 0 = deterministic</span></div>')
                        temperature_slider = gr.Slider(
                            minimum=0.0, maximum=2.0, value=0.0, step=0.1,
                            label="", show_label=False,
                            elem_id="temperature-slider",
                        )
                    clear_memory_btn = gr.Button(
                        "🧹 Clear Memory", scale=1, elem_classes="secondary-btn", size="sm"
                    )

                memory_stats_text = gr.Markdown(value="", elem_id="memory-stats")

        # ─── Event Handlers ───────────────────────────────────────────────
        all_sample_btn_components = [b for b, _ in sample_btns]

        def on_load():
            docs, files, status_msg, model, device = get_status()
            has_docs = len(docs) > 0
            mem_stat = get_memory_stats()
            sample_updates = [gr.update(interactive=has_docs) for _ in sample_btns]
            return (
                gr.update(value=_header_html(model, device)),
                gr.update(choices=docs, value=None),
                status_msg,
                gr.update(interactive=has_docs),
                mem_stat,
                *sample_updates,
            )

        demo.load(
            fn=on_load,
            outputs=[header_html, doc_list, status_text, submit_btn, memory_stats_text, *all_sample_btn_components],
        )

        def refresh_and_update_samples():
            docs, files, status_msg, model, device = get_status()
            has_docs = len(docs) > 0
            mem_stat = get_memory_stats()
            sample_updates = [gr.update(interactive=has_docs) for _ in sample_btns]
            return (
                gr.update(value=_header_html(model, device)),
                gr.update(choices=docs, value=None),
                status_msg,
                gr.update(interactive=has_docs),
                mem_stat,
                *sample_updates,
            )

        file_upload.upload(
            fn=lambda files: (
                _status_html(upload_files(files)[0]),
                *refresh_and_update_samples(),
            ),
            inputs=[file_upload],
            outputs=[upload_status, header_html, doc_list, status_text, submit_btn, memory_stats_text, *all_sample_btn_components],
        )

        delete_btn.click(
            fn=lambda doc: (
                _status_html(delete_document(doc)[0]),
                *refresh_and_update_samples(),
            ),
            inputs=[doc_list],
            outputs=[delete_status, header_html, doc_list, status_text, submit_btn, memory_stats_text, *all_sample_btn_components],
        )

        # First click on Remove ALL — show confirmation row
        delete_all_btn.click(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[confirm_row],
        )

        # Confirm: execute delete, hide confirmation row
        confirm_yes_btn.click(
            fn=lambda: (
                gr.update(visible=False),
                _status_html(delete_all_embeddings()[0]),
                *refresh_and_update_samples(),
            ),
            inputs=[],
            outputs=[confirm_row, delete_status, header_html, doc_list, status_text, submit_btn, memory_stats_text, *all_sample_btn_components],
        )

        # Cancel: just hide the confirmation row
        confirm_no_btn.click(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[confirm_row],
        )

        refresh_btn.click(
            fn=refresh_and_update_samples,
            outputs=[header_html, doc_list, status_text, submit_btn, memory_stats_text, *all_sample_btn_components],
        )

        def on_submit(message, history, n, temp):
            history = history or []
            for updated_history, _ in chat_fn(message, history, n, temp):
                yield updated_history, "", ""
            yield updated_history, "", get_memory_stats()

        submit_btn.click(
            fn=on_submit,
            inputs=[msg_input, chatbot, n_results_slider, temperature_slider],
            outputs=[chatbot, msg_input, memory_stats_text],
        )

        msg_input.submit(
            fn=on_submit,
            inputs=[msg_input, chatbot, n_results_slider, temperature_slider],
            outputs=[chatbot, msg_input, memory_stats_text],
        )

        # Wire each sample question button → prefill input and auto-submit
        def make_sample_handler(question_text):
            def handler(history, n, temp):
                for updated_history, _ in chat_fn(question_text, history or [], n, temp):
                    yield updated_history, "", ""
                yield updated_history, "", get_memory_stats()
            return handler

        for btn, question_text in sample_btns:
            btn.click(
                fn=make_sample_handler(question_text),
                inputs=[chatbot, n_results_slider, temperature_slider],
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
    )
