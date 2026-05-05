"""
FastAPI backend for the Multimodal RAG system.
Exposes endpoints for document management and querying.
"""
import os
import asyncio
import logging
import shutil
from pathlib import Path
from typing import List, Optional

import threading
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from utils.document_processor import process_document_chunked, SUPPORTED_EXTENSIONS
from utils.vector_store import VectorStoreManager
from utils.rag_engine import RAGEngine, BACKEND
from utils.memory import ConversationMemory, estimate_tokens
from utils.device import device_info

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "./data")
VECTORSTORE_DIR = os.environ.get("VECTORSTORE_DIR", "./vectorstore")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

# HF Hub Dataset used for persistent user-uploaded file storage.
# Set HF_DATASET_REPO (e.g. "yourname/MyApp-data") and HF_TOKEN as Space secrets.
# Files uploaded via the app are pushed here and re-downloaded on every cold start,
# so they survive container restarts and redeployments.
HF_DATASET_REPO = os.environ.get("MultiModalRag_dataset", "")
HF_TOKEN = os.environ.get("MultiModalRag_Token", "")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)


# ─── HF Hub Persistent Storage Helpers ───────────────────────────────────────

def _hf_api():
    """Return a configured HfApi instance, or None if not set up."""
    if HF_DATASET_REPO and HF_TOKEN:
        from huggingface_hub import HfApi
        return HfApi(token=HF_TOKEN)
    return None


def sync_from_hf_hub():
    """Download user-uploaded files from HF Hub dataset to data dir on startup.
    Only downloads files that don't already exist locally (committed files win).
    """
    api = _hf_api()
    if not api:
        return
    try:
        import huggingface_hub
        files = list(api.list_repo_files(HF_DATASET_REPO, repo_type="dataset"))
        for path_in_repo in files:
            # We store uploaded files under "data/<filename>" in the dataset repo
            if not path_in_repo.startswith("data/"):
                continue
            basename = Path(path_in_repo).name
            if not basename or Path(basename).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            local_path = Path(DATA_DIR) / basename
            if local_path.exists():
                logger.info(f"HF Hub sync: '{basename}' already present locally — skipping.")
                continue
            downloaded = huggingface_hub.hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=path_in_repo,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            shutil.copy2(downloaded, str(local_path))
            logger.info(f"HF Hub sync: downloaded '{basename}'")
    except Exception as e:
        logger.warning(f"HF Hub sync (download) failed: {e}")


def push_to_hf_hub(filename: str):
    """Push a single file from data dir to the HF Hub dataset repo."""
    api = _hf_api()
    if not api:
        return
    try:
        api.upload_file(
            path_or_fileobj=str(Path(DATA_DIR) / filename),
            path_in_repo=f"data/{filename}",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Upload {filename}",
        )
        logger.info(f"HF Hub: pushed '{filename}'")
    except Exception as e:
        logger.warning(f"HF Hub push failed for '{filename}': {e}")


def delete_from_hf_hub(filename: str):
    """Delete a single file from the HF Hub dataset repo."""
    api = _hf_api()
    if not api:
        return
    try:
        api.delete_file(
            path_in_repo=f"data/{filename}",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Delete {filename}",
        )
        logger.info(f"HF Hub: deleted '{filename}'")
    except Exception as e:
        logger.warning(f"HF Hub delete failed for '{filename}': {e}")


def sync_vectorstore_from_hf_hub():
    """Download persisted vectorstore from HF Hub dataset.
    Must be called BEFORE VectorStoreManager is initialized so ChromaDB can
    load existing embeddings and avoid re-indexing on cold start.
    """
    if not (HF_DATASET_REPO and HF_TOKEN):
        return
    try:
        import huggingface_hub
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        files = list(api.list_repo_files(HF_DATASET_REPO, repo_type="dataset"))
        vs_files = [f for f in files if f.startswith("vectorstore/")]
        if not vs_files:
            logger.info("HF Hub: no persisted vectorstore found — will build from scratch.")
            return
        for path_in_repo in vs_files:
            rel = path_in_repo[len("vectorstore/"):]
            if not rel:
                continue
            local = Path(VECTORSTORE_DIR) / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            dl = huggingface_hub.hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=path_in_repo,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            shutil.copy2(dl, str(local))
            logger.info(f"HF Hub vectorstore: restored '{rel}'")
    except Exception as e:
        logger.warning(f"HF Hub vectorstore sync failed: {e}")


def push_vectorstore_to_hf_hub():
    """Push the entire vectorstore directory to HF Hub dataset.
    Called after every index or delete operation so embeddings survive restarts.
    """
    api = _hf_api()
    if not api:
        return
    try:
        api.upload_folder(
            folder_path=VECTORSTORE_DIR,
            path_in_repo="vectorstore",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message="Update vectorstore",
            ignore_patterns=["*.lock", ".DS_Store"],
        )
        logger.info("HF Hub: pushed vectorstore")
    except Exception as e:
        logger.warning(f"HF Hub vectorstore push failed: {e}")


def _copy_committed_files():
    """Copy baseline PDFs committed to the Space repo under _secrets/data/ into DATA_DIR.
    These are always available in the Space even without HF Hub Dataset configured.
    """
    secrets_data = Path("_secrets/data")
    if not secrets_data.exists():
        return
    for fp in secrets_data.iterdir():
        if fp.suffix.lower() in SUPPORTED_EXTENSIONS:
            dest = Path(DATA_DIR) / fp.name
            if not dest.exists():
                shutil.copy2(str(fp), str(dest))
                logger.info(f"Copied committed file '{fp.name}' → data/")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Pre-init: restore persisted data so ChromaDB loads existing embeddings ──
# This runs BEFORE VectorStoreManager is created. Restoring the vectorstore here
# means ChromaDB will open the existing DB — no re-indexing needed on cold start.
logger.info("Pre-init: copying committed baseline files...")
_copy_committed_files()
logger.info("Pre-init: restoring vectorstore from HF Hub (if configured)...")
sync_vectorstore_from_hf_hub()
logger.info("Pre-init: restoring data files from HF Hub (if configured)...")
sync_from_hf_hub()

# ─── Singletons ───────────────────────────────────────────────────────────────
vs = VectorStoreManager(persist_dir=VECTORSTORE_DIR)
rag = RAGEngine(vector_store=vs, model=OLLAMA_MODEL)
memory = ConversationMemory()


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Multimodal RAG API", version="1.0.0")

# CORS is a browser security mechanism that blocks web pages from making requests to a different domain than the one that served the page. For example, if your Gradio frontend runs on localhost:7860 and tries to call your FastAPI backend on localhost:8000, the browser would normally block that request.
app.add_middleware( 
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Models ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    n_results: int = 8
    temperature: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    tokens_user: int = 0  # tokens in user message
    tokens_assistant: int = 0  # tokens in assistant response


class StatusResponse(BaseModel):
    documents: List[str]
    total_chunks: int
    data_dir_files: List[str]
    model: str
    device: str


class URLIndexRequest(BaseModel):
    url: str
    max_depth: int = 2
    max_pages: int = 50


# ─── Helper ───────────────────────────────────────────────────────────────────
def index_file(filepath: str) -> int:
    """Process and index a file into the vector store."""
    chunks = process_document_chunked(filepath)
    source_name = Path(filepath).name
    # Remove old version first (re-index)
    vs.remove_document(source_name)
    return vs.add_documents(chunks, source_name)


def index_all_data_dir():
    """Index all supported files in DATA_DIR on startup."""
    indexed_sources = set(vs.list_sources())
    for fp in Path(DATA_DIR).iterdir():
        if fp.suffix.lower() in SUPPORTED_EXTENSIONS and fp.name not in indexed_sources:
            try:
                n = index_file(str(fp))
                logger.info(f"Indexed '{fp.name}': {n} chunks")
            except Exception as e:
                logger.error(f"Failed to index '{fp.name}': {e}")


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    # Files and vectorstore were already restored at module load (pre-init).
    # index_all_data_dir() is a no-op for already-indexed docs; it only indexes
    # any files that arrived in data/ but aren't in the vectorstore yet.
    logger.info("Indexing any new documents not yet in vectorstore...")
    index_all_data_dir()
    logger.info(f"Ready. Vector store has {vs.total_chunks()} chunks.")


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/status", response_model=StatusResponse)
async def get_status():
    data_files = [
        f.name for f in Path(DATA_DIR).iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return StatusResponse(
        documents=vs.list_sources(),
        total_chunks=vs.total_chunks(),
        data_dir_files=data_files,
        model=rag.model,
        device=device_info()["label"],
    )


@app.post("/documents/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Save a document to disk and start background indexing. Returns immediately."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Supported: {SUPPORTED_EXTENSIONS}")

    save_path = Path(DATA_DIR) / file.filename
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    with _upload_lock:
        _upload_jobs[file.filename] = {"status": "processing"}

    background_tasks.add_task(_index_background, file.filename, str(save_path))
    return {
        "message": f"⏳ Indexing started for '{file.filename}' — polling for status.",
        "status": "processing",
        "filename": file.filename,
    }


@app.get("/documents/upload/status")
async def upload_status(filename: str):
    """Poll the indexing status of a background file upload."""
    with _upload_lock:
        job = _upload_jobs.get(filename)
    if job is None:
        raise HTTPException(404, f"No upload job found for '{filename}'")
    return job


@app.delete("/documents/{filename:path}")
async def delete_document(filename: str):
    """Remove a document's embeddings from the vector store only. File is kept on disk."""
    removed_chunks = vs.remove_document(filename)
    # Persist the updated vectorstore so the removal survives restarts
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, push_vectorstore_to_hf_hub)
    if removed_chunks > 0:
        return {"message": f"Removed {removed_chunks} indexed chunks for '{filename}'. File kept on disk."}
    else:
        raise HTTPException(404, f"No indexed chunks found for '{filename}'.")


@app.delete("/documents")
async def delete_all_documents():
    """Remove ALL embeddings from the vector store only. Files are kept on disk."""
    filenames = [f.name for f in Path(DATA_DIR).iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    removed = vs.clear_all()
    # Persist the cleared vectorstore
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, push_vectorstore_to_hf_hub)
    return {"message": f"Removed all {removed} indexed chunks. {len(filenames)} file(s) kept on disk.", "chunks_removed": removed}


# ─── Upload job tracker ──────────────────────────────────────────────────────
_upload_jobs: dict = {}   # filename → {"status": "processing"|"done"|"error", ...}
_upload_lock = threading.Lock()


def _index_background(filename: str, save_path: str):
    """Runs in a background thread: index file, persist to HF Hub, update job status."""
    try:
        n_chunks = index_file(save_path)
        push_to_hf_hub(filename)
        push_vectorstore_to_hf_hub()  # persist embeddings so they survive restarts
        with _upload_lock:
            _upload_jobs[filename] = {
                "status": "done",
                "message": f"Uploaded and indexed '{filename}' ({n_chunks} chunks).",
                "chunks": n_chunks,
            }
        logger.info(f"Background index done: '{filename}' — {n_chunks} chunks")
    except Exception as e:
        logger.error(f"Background index failed for '{filename}': {e}", exc_info=True)
        # File is kept on disk even if indexing fails — user can retry
        with _upload_lock:
            _upload_jobs[filename] = {"status": "error", "message": str(e)}


# ─── URL crawl job tracker ────────────────────────────────────────────────────
_crawl_jobs: dict = {}   # url → {"status": "crawling"|"done"|"error", ...}
_crawl_lock = threading.Lock()


def _crawl_background(url: str, max_depth: int, max_pages: int):
    """Runs in a background thread: crawl + index, then update job status."""
    from utils.url_processor import crawl_url
    try:
        vs.remove_document(url)
        chunks, crawled_urls = crawl_url(url, max_depth=max_depth, max_pages=max_pages)
        if not chunks:
            with _crawl_lock:
                _crawl_jobs[url] = {"status": "error", "message": "No content extracted."}
            return
        n_chunks = vs.add_documents(chunks, url)
        with _crawl_lock:
            _crawl_jobs[url] = {
                "status": "done",
                "message": (
                    f"Indexed {len(crawled_urls)} page(s) and file(s) "
                    f"({n_chunks} chunks) from {url}"
                ),
                "pages": len(crawled_urls),
                "chunks": n_chunks,
            }
        logger.info(f"Crawl done: {url} — {len(crawled_urls)} pages, {n_chunks} chunks")
    except Exception as e:
        logger.error(f"Crawl failed for {url}: {e}", exc_info=True)
        with _crawl_lock:
            _crawl_jobs[url] = {"status": "error", "message": str(e)}


@app.post("/documents/url")
async def index_url(req: URLIndexRequest, background_tasks: BackgroundTasks):
    """Start a background crawl of a URL (2 levels deep). Returns immediately."""
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "URL must start with http:// or https://")

    with _crawl_lock:
        _crawl_jobs[url] = {"status": "crawling"}

    background_tasks.add_task(_crawl_background, url, req.max_depth, req.max_pages)
    return {
        "message": f"⏳ Crawling started for {url} — refresh the document list in ~30 s.",
        "status": "crawling",
        "url": url,
    }


@app.get("/documents/url/status")
async def url_crawl_status(url: str):
    """Poll the status of a background URL crawl."""
    with _crawl_lock:
        job = _crawl_jobs.get(url)
    if job is None:
        raise HTTPException(404, f"No crawl job found for {url}")
    return job


@app.post("/documents/reindex")
async def reindex_all():
    """Force re-index all documents in data dir."""
    for fp in Path(DATA_DIR).iterdir():
        if fp.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                index_file(str(fp))
            except Exception as e:
                logger.error(f"Reindex failed for {fp.name}: {e}")
    return {"message": "Reindexed all documents.", "total_chunks": vs.total_chunks()}


_GREETING_RESPONSES = {
    "hi": "Hi! Ask me anything about your uploaded documents.",
    "hello": "Hello! Ask me anything about your uploaded documents.",
    "hey": "Hey! Ask me anything about your uploaded documents.",
    "hiya": "Hi there! Ask me anything about your uploaded documents.",
    "howdy": "Howdy! Ask me anything about your uploaded documents.",
    "greetings": "Greetings! Ask me anything about your uploaded documents.",
    "sup": "Hey! Ask me anything about your uploaded documents.",
    "yo": "Hey! Ask me anything about your uploaded documents.",
    "good morning": "Good morning! Ask me anything about your uploaded documents.",
    "good afternoon": "Good afternoon! Ask me anything about your uploaded documents.",
    "good evening": "Good evening! Ask me anything about your uploaded documents.",
    "good day": "Good day! Ask me anything about your uploaded documents.",
    "how are you": "I'm doing well, thank you! Ask me anything about your uploaded documents.",
    "how are you doing": "I'm doing well, thank you! Ask me anything about your uploaded documents.",
    "how do you do": "I'm doing well, thank you! How can I help with your documents?",
    "what's up": "Not much! Ready to answer questions about your documents.",
    "whats up": "Not much! Ready to answer questions about your documents.",
    "what is up": "Not much! Ready to answer questions about your documents.",
    "thanks": "You're welcome! Let me know if you have more questions.",
    "thank you": "You're welcome! Let me know if you have more questions.",
    "thx": "You're welcome! Let me know if you have more questions.",
    "ty": "You're welcome! Let me know if you have more questions.",
    "bye": "Goodbye! Feel free to come back anytime.",
    "goodbye": "Goodbye! Feel free to come back anytime.",
    "see you": "See you! Feel free to come back anytime.",
    "cya": "See you later! Feel free to come back anytime.",
    "ok": "Let me know if you have any questions about your documents.",
    "okay": "Let me know if you have any questions about your documents.",
    "cool": "Glad to help! Let me know if you have more questions.",
    "great": "Glad to help! Let me know if you have more questions.",
    "nice": "Thanks! Let me know if you have more questions.",
}

_META_PATTERNS = [
    "how can you help", "how you can help", "what can you do",
    "what do you do", "who are you", "what are you",
    "help me", "how does this work", "how do you work",
]

_META_ANSWER = (
    "I'm your document assistant. Here's how I can help:\n\n"
    "1. **Upload documents** (PDF, Word, Excel, CSV, TXT, images) or **add URLs** in the Documents tab\n"
    "2. **Ask questions** about your uploaded documents and I'll answer based on their content\n"
    "3. I can handle text, tables, charts, and scanned images\n"
    "4. Use the **Read** button to hear answers aloud\n\n"
    "Upload some documents and start asking questions!"
)

def _chitchat_response(text: str) -> str | None:
    """Return a context-appropriate response for chitchat, or None if not chitchat."""
    normalized = text.strip().lower().rstrip("!?.,")
    if normalized in _GREETING_RESPONSES:
        return _GREETING_RESPONSES[normalized]
    for pattern in _META_PATTERNS:
        if pattern in normalized:
            return _META_ANSWER
    return None


@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """Query the RAG system."""
    try:
        # Short-circuit chitchat / greetings — don't pollute with RAG results
        chitchat_answer = _chitchat_response(req.question)
        if chitchat_answer:
            return QueryResponse(answer=chitchat_answer, sources=[])

        if vs.total_chunks() == 0:
            return QueryResponse(answer="No documents are indexed yet. Please upload some documents first.", sources=[])

        # Run all blocking work (embedding + LLM) in a thread executor
        def _run_query():
            results = vs.query(req.question, n_results=req.n_results)
            sources = list({r["metadata"].get("source", "") for r in results})
            parts = []
            for token in rag.query(req.question, memory, n_results=req.n_results, temperature=req.temperature, stream=False):
                parts.append(token)
            answer = "".join(parts)
            tokens_user = estimate_tokens(req.question)
            tokens_assistant = estimate_tokens(answer)
            return answer, sources, tokens_user, tokens_assistant

        loop = asyncio.get_running_loop()
        answer, sources, tokens_user, tokens_assistant = await loop.run_in_executor(None, _run_query)

        return QueryResponse(answer=answer, sources=sources, tokens_user=tokens_user, tokens_assistant=tokens_assistant)
    except Exception as e:
        logger.error(f"Query endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/clear")
async def clear_memory():
    memory.clear()
    return {"message": "Conversation memory cleared."}


@app.get("/memory/stats")
async def memory_stats():
    from utils.memory import estimate_tokens
    total_tokens = sum(estimate_tokens(m.content) for m in memory.messages)
    summary_tokens = estimate_tokens(memory.summary) if memory.summary else 0
    return {
        "message_count": len(memory.messages),
        "total_tokens": total_tokens + summary_tokens,
        "has_summary": memory.summary is not None,
        "max_tokens": memory.max_tokens,
    }


@app.get("/models")
async def list_models():
    return {"models": rag.list_available_models(), "current": OLLAMA_MODEL}
