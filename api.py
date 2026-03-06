"""
FastAPI backend for the Multimodal RAG system.
Exposes endpoints for document management and querying.
"""
import os
import logging
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from utils.document_processor import process_document_chunked, SUPPORTED_EXTENSIONS
from utils.vector_store import VectorStoreManager
from utils.rag_engine import RAGEngine
from utils.memory import ConversationMemory
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


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


class StatusResponse(BaseModel):
    documents: List[str]
    total_chunks: int
    data_dir_files: List[str]
    model: str
    device: str


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
        model=OLLAMA_MODEL,
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


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Remove a document's embeddings from the vector store only (file kept on disk)."""
    removed_chunks = vs.remove_document(filename)
    if removed_chunks > 0:
        return {"message": f"Removed {removed_chunks} indexed chunks for '{filename}'. File kept on disk."}
    else:
        raise HTTPException(404, f"No indexed chunks found for '{filename}'.")


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


@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """Query the RAG system."""
    import asyncio
    try:
        if vs.total_chunks() == 0:
            return QueryResponse(answer="I DON'T KNOW", sources=[])

        # Run all blocking work (embedding + Ollama) in a thread executor
        def _run_query():
            results = vs.query(req.question, n_results=req.n_results)
            sources = list({r["metadata"].get("source", "") for r in results})
            parts = []
            for token in rag.query(req.question, memory, n_results=req.n_results, stream=False):
                parts.append(token)
            return "".join(parts), sources

        loop = asyncio.get_running_loop()
        answer, sources = await loop.run_in_executor(None, _run_query)

        return QueryResponse(answer=answer, sources=sources)
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
