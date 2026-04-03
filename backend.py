"""
FastAPI backend for the Multimodal RAG system.
Exposes endpoints for document management and querying.
"""
import os
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
from utils.rag_engine import RAGEngine, USE_GROQ
from utils.memory import ConversationMemory, estimate_tokens
from utils.device import device_info

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "./data")
VECTORSTORE_DIR = os.environ.get("VECTORSTORE_DIR", "./vectorstore")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

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
    n_results: int = 5
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
    logger.info("Indexing existing documents in data dir...")
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
async def upload_document(file: UploadFile = File(...)):
    """Upload a document, save to data dir, and index it."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Supported: {SUPPORTED_EXTENSIONS}")
    
    save_path = Path(DATA_DIR) / file.filename
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        n_chunks = await loop.run_in_executor(None, index_file, str(save_path))
        return {"message": f"Uploaded and indexed '{file.filename}' ({n_chunks} chunks).", "chunks": n_chunks}
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Indexing failed: {str(e)}")


@app.delete("/documents/{filename:path}")
async def delete_document(filename: str):
    """Remove a document's embeddings from the vector store only (file kept on disk)."""
    removed_chunks = vs.remove_document(filename)
    if removed_chunks > 0:
        return {"message": f"Removed {removed_chunks} indexed chunks for '{filename}'. File kept on disk."}
    else:
        raise HTTPException(404, f"No indexed chunks found for '{filename}'.")


@app.delete("/documents")
async def delete_all_documents():
    """Remove ALL embeddings from the vector store. Files are kept on disk."""
    removed = vs.clear_all()
    return {"message": f"Removed all {removed} indexed chunks. Files kept on disk.", "chunks_removed": removed}


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


_GREETINGS = {
    "hi", "hello", "hey", "hiya", "howdy", "greetings", "sup", "yo",
    "good morning", "good afternoon", "good evening", "good day",
    "how are you", "how are you doing", "how do you do",
    "what's up", "whats up", "what is up",
    "thanks", "thank you", "thx", "ty",
    "bye", "goodbye", "see you", "cya",
    "ok", "okay", "cool", "great", "nice",
}

def _is_chitchat(text: str) -> bool:
    """Return True if the query is conversational chitchat that shouldn't hit the vector store."""
    normalized = text.strip().lower().rstrip("!?.,")
    return normalized in _GREETINGS or len(normalized.split()) <= 1 and normalized in _GREETINGS


@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """Query the RAG system."""
    import asyncio
    try:
        # Short-circuit chitchat / greetings — don't pollute with RAG results
        if _is_chitchat(req.question):
            return QueryResponse(
                answer="Hello! I'm your document assistant. Ask me anything about your uploaded documents.",
                sources=[],
            )

        if vs.total_chunks() == 0:
            return QueryResponse(answer="No documents are indexed yet. Please upload some documents first.", sources=[])

        # Run all blocking work (embedding + Ollama) in a thread executor

        def _run_query():
            results = vs.query(req.question, n_results=req.n_results)
            sources = list({r["metadata"].get("source", "") for r in results})
            # Build context for token count
            context = rag._build_context(results)
            user_message = f"[CONTEXT]\n{context}\n\n[QUESTION]\n{req.question}\n\nRemember: Answer ONLY from the context above. If not found, say \"I DON'T KNOW\"."
            # Compose full prompt for LLM
            system_prompt = getattr(rag, "SYSTEM_PROMPT", "You are a document assistant. Answer questions using ONLY the [CONTEXT] provided. If the answer is not in the context, respond: 'I DON'T KNOW'. Be concise and factual. Cite source and page when available.")
            prompt = f"{system_prompt}\n{user_message}"
            parts = []
            for token in rag.query(req.question, memory, n_results=req.n_results, temperature=req.temperature, stream=False):
                parts.append(token)
            answer = "".join(parts)
            # Per-message token counts
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
