"""
FastAPI backend for the Multimodal RAG system.
Exposes endpoints for document management and querying.
"""
import os

# On local dev (no SPACE_ID env var) default the embedding model to CPU.
# MPS (Apple Silicon GPU) can segfault after a previous crash leaves the Metal
# driver in a bad state. CPU is fast enough for local dev.
if not os.environ.get("SPACE_ID"):
    os.environ.setdefault("TORCH_DEVICE", "cpu")

import asyncio
import logging
import re
import shutil
from pathlib import Path
from typing import List, Optional

import threading
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from utils.document_processor import process_document_chunked, SUPPORTED_EXTENSIONS, extract_dataframes, extract_images
from utils.vector_store import VectorStoreManager
from utils.rag_engine import RAGEngine, BACKEND, RELEVANCE_THRESHOLD
from utils.memory import ConversationMemory, estimate_tokens
from utils.device import device_info
from utils.table_store import TableStore
from utils.image_store import ImageStore

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
        print("[STARTUP] sync_data: SKIPPED — HF_DATASET_REPO or HF_TOKEN not set", flush=True)
        return
    try:
        import huggingface_hub
        files = list(api.list_repo_files(HF_DATASET_REPO, repo_type="dataset"))
        data_files = [f for f in files if f.startswith("data/") and
                      Path(f).suffix.lower() in SUPPORTED_EXTENSIONS and Path(f).name]
        print(f"[STARTUP] sync_data: {len(data_files)} supported file(s) in HF Hub", flush=True)
        downloaded_count = 0
        for path_in_repo in data_files:
            basename = Path(path_in_repo).name
            local_path = Path(DATA_DIR) / basename
            if local_path.exists():
                continue
            dl = huggingface_hub.hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=path_in_repo,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            shutil.copy2(dl, str(local_path))
            downloaded_count += 1
            print(f"[STARTUP] sync_data: downloaded '{basename}'", flush=True)
        print(f"[STARTUP] sync_data: {downloaded_count} new file(s) downloaded", flush=True)
    except Exception as e:
        print(f"[STARTUP] sync_data: FAILED — {e}", flush=True)
        logger.warning(f"HF Hub sync (download) failed: {e}")


def push_to_hf_hub(filename: str):
    """Push a single file from data dir to the HF Hub dataset repo."""
    if not (HF_DATASET_REPO and HF_TOKEN):
        return
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
        print("[STARTUP] sync_vectorstore: SKIPPED — HF_DATASET_REPO or HF_TOKEN not set", flush=True)
        return
    try:
        import huggingface_hub
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        files = list(api.list_repo_files(HF_DATASET_REPO, repo_type="dataset"))
        vs_files = [f for f in files if f.startswith("vectorstore/")]
        if not vs_files:
            print("[STARTUP] sync_vectorstore: no vectorstore in HF Hub — will build from scratch", flush=True)
            return
        print(f"[STARTUP] sync_vectorstore: downloading {len(vs_files)} file(s)...", flush=True)
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
        print(f"[STARTUP] sync_vectorstore: restored {len(vs_files)} file(s) OK", flush=True)
    except Exception as e:
        print(f"[STARTUP] sync_vectorstore: FAILED — {e}", flush=True)
        logger.warning(f"HF Hub vectorstore sync failed: {e}")


def push_vectorstore_to_hf_hub():
    """Push the entire vectorstore directory to HF Hub dataset.
    Called after every index or delete operation so embeddings survive restarts.
    """
    if not (HF_DATASET_REPO and HF_TOKEN):
        return
    api = _hf_api()
    if not api:
        return
    try:
        # Compact SQLite before uploading to keep file sizes small.
        # ChromaDB accumulates free pages over time; VACUUM reclaims them.
        _sqlite_path = Path(VECTORSTORE_DIR) / "chroma.sqlite3"
        if _sqlite_path.exists():
            try:
                import sqlite3 as _sqlite3
                _conn = _sqlite3.connect(str(_sqlite_path), timeout=5)
                _conn.execute("VACUUM")
                _conn.close()
                logger.info("Vectorstore SQLite compacted before push")
            except Exception as _ve:
                logger.warning(f"SQLite VACUUM skipped: {_ve}")
        api.upload_folder(
            folder_path=VECTORSTORE_DIR,
            path_in_repo="vectorstore",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message="Update vectorstore",
            ignore_patterns=["*.lock", ".DS_Store", "*.wal", "*.shm"],
        )
        logger.info("HF Hub: pushed vectorstore")
    except Exception as e:
        logger.warning(f"HF Hub vectorstore push failed: {e}")


def push_tables_to_hf_hub():
    if not (HF_DATASET_REPO and HF_TOKEN):
        return
    api = _hf_api()
    if not api:
        return
    try:
        api.upload_folder(
            folder_path=str(Path(DATA_DIR) / "tables"),
            path_in_repo="tables",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message="Update tables",
            ignore_patterns=["*.lock", ".DS_Store"],
        )
        logger.info("HF Hub: pushed tables")
    except Exception as e:
        logger.warning(f"HF Hub tables push failed: {e}")


def sync_tables_from_hf_hub():
    if not (HF_DATASET_REPO and HF_TOKEN):
        print("[STARTUP] sync_tables: SKIPPED — HF_DATASET_REPO or HF_TOKEN not set", flush=True)
        return
    try:
        import huggingface_hub
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        files = list(api.list_repo_files(HF_DATASET_REPO, repo_type="dataset"))
        table_files = [f for f in files if f.startswith("tables/")]
        if not table_files:
            print("[STARTUP] sync_tables: no tables found on HF Hub — will rely on on-demand extraction", flush=True)
            return
        print(f"[STARTUP] sync_tables: downloading {len(table_files)} file(s)...", flush=True)
        tables_dir = Path(DATA_DIR) / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        for path_in_repo in table_files:
            rel = path_in_repo[len("tables/"):]
            if not rel:
                continue
            local = tables_dir / rel
            dl = huggingface_hub.hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=path_in_repo,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            shutil.copy2(dl, str(local))
            logger.info(f"HF Hub tables: restored '{rel}'")
        print(f"[STARTUP] sync_tables: restored {len(table_files)} file(s) OK", flush=True)
    except Exception as e:
        print(f"[STARTUP] sync_tables: FAILED — {e}", flush=True)
        logger.warning(f"HF Hub tables sync failed: {e}")


def push_images_to_hf_hub():
    if not (HF_DATASET_REPO and HF_TOKEN):
        return
    api = _hf_api()
    if not api:
        return
    try:
        api.upload_folder(
            folder_path=str(Path(DATA_DIR) / "images"),
            path_in_repo="images",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message="Update images",
            ignore_patterns=["*.lock", ".DS_Store"],
        )
        logger.info("HF Hub: pushed images")
    except Exception as e:
        logger.warning(f"HF Hub images push failed: {e}")


def sync_images_from_hf_hub():
    if not (HF_DATASET_REPO and HF_TOKEN):
        return
    try:
        import huggingface_hub
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        files = list(api.list_repo_files(HF_DATASET_REPO, repo_type="dataset"))
        image_files = [f for f in files if f.startswith("images/")]
        if not image_files:
            return
        images_dir = Path(DATA_DIR) / "images"
        for path_in_repo in image_files:
            rel = path_in_repo[len("images/"):]
            if not rel:
                continue
            local = images_dir / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            dl = huggingface_hub.hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=path_in_repo,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            shutil.copy2(dl, str(local))
            logger.info(f"HF Hub images: restored '{rel}'")
    except Exception as e:
        logger.warning(f"HF Hub images sync failed: {e}")


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
# This runs BEFORE VectorStoreManager is created. On HF Spaces (SPACE_ID is set)
# it downloads the latest data + vectorstore from HF Hub. Locally, files are
# already on disk so we skip the network calls for a fast startup.
_IS_HF_SPACE = bool(os.environ.get("SPACE_ID"))

print(
    f"[STARTUP] IS_HF_SPACE={_IS_HF_SPACE} | "
    f"HF_DATASET_REPO={'SET' if HF_DATASET_REPO else 'NOT SET'} | "
    f"HF_TOKEN={'SET' if HF_TOKEN else 'NOT SET'}",
    flush=True,
)
_copy_committed_files()
if _IS_HF_SPACE:
    sync_vectorstore_from_hf_hub()
    sync_from_hf_hub()
    sync_tables_from_hf_hub()
    sync_images_from_hf_hub()
else:
    logger.info("Pre-init: local run — skipping HF Hub sync (files already on disk).")

# ─── Singletons ───────────────────────────────────────────────────────────────
vs = VectorStoreManager(persist_dir=VECTORSTORE_DIR)
_vs_chunks = vs.total_chunks()
_vs_sources = vs.list_sources()
_data_files = [f.name for f in Path(DATA_DIR).iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
print(
    f"[STARTUP] VS loaded: {_vs_chunks} chunks, {len(_vs_sources)} source(s): {_vs_sources}",
    flush=True,
)
print(f"[STARTUP] DATA_DIR files: {_data_files}", flush=True)
rag = RAGEngine(vector_store=vs, model=OLLAMA_MODEL)
memory = ConversationMemory()
ts = TableStore()
img_store = ImageStore()


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
    source_filter: List[str] = []


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    tokens_user: int = 0
    tokens_assistant: int = 0
    chunks_used: int = 0
    sql_query: str = ""
    answer_method: str = "rag"  # "rag" | "table_query"


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
    """Process and index a file into the vector store, and extract tables into TableStore."""
    chunks = process_document_chunked(filepath)
    source_name = Path(filepath).name
    vs.remove_document(source_name)
    n = vs.add_documents(chunks, source_name)
    try:
        dfs = _extract_tables(filepath)
        if dfs:
            ts.save(source_name, dfs)
    except Exception as e:
        logger.warning(f"Table extraction failed for '{source_name}': {e}")
    return n


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
    # ChromaDB PersistentClient already restores the vectorstore from disk on init.
    # Do NOT auto-index data/ here — that would re-index files the user deleted
    # from the index, making them reappear after every restart.
    # On HF Spaces, sync_from_hf_hub() + sync_vectorstore_from_hf_hub() run at
    # module load (before this), so the vectorstore is already fully restored.

    def _backfill_tables():
        for fp in Path(DATA_DIR).iterdir():
            if fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            src = fp.name
            if ts.was_attempted(src):
                continue
            try:
                dfs = _extract_tables(str(fp))
                ts.save(src, dfs)
                if dfs:
                    logger.info(f"Startup backfill: {len(dfs)} table(s) for '{src}'")
            except Exception as e:
                logger.warning(f"Startup backfill failed for '{src}': {e}")
                ts.save(src, [])

    def _backfill_images():
        for fp in Path(DATA_DIR).iterdir():
            if fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            src = fp.name
            if img_store.was_attempted(src):
                continue
            try:
                images = extract_images(str(fp))
                img_store.save(src, images)
                if images:
                    logger.info(f"Startup backfill: {len(images)} image(s) for '{src}'")
            except Exception as e:
                logger.warning(f"Startup image backfill failed for '{src}': {e}")
                img_store.save(src, [])

    def _index_missing_files():
        """Index any file in data/ that is not yet in the vectorstore.
        Runs on every startup to self-heal after a stale or failed vectorstore restore.
        Safe to run because deleted files are now removed from disk too.
        """
        indexed = set(vs.list_sources())
        missing = [
            fp for fp in Path(DATA_DIR).iterdir()
            if fp.suffix.lower() in SUPPORTED_EXTENSIONS and fp.name not in indexed
        ]
        if not missing:
            return
        logger.warning(
            "%d file(s) in data/ not in vectorstore — indexing: %s",
            len(missing), [f.name for f in missing],
        )
        failed = []
        for fp in missing:
            try:
                n = index_file(str(fp))
                logger.warning(f"Startup index OK: '{fp.name}' — {n} chunks")
            except Exception as e:
                logger.error(f"Startup index FAILED: '{fp.name}': {e}", exc_info=True)
                failed.append(fp.name)
        logger.warning(
            "Startup index complete: %d OK, %d failed. VS now has %d chunks.",
            len(missing) - len(failed), len(failed), vs.total_chunks(),
        )
        if _IS_HF_SPACE:
            if failed:
                logger.warning(f"Skipping vectorstore push — {len(failed)} file(s) failed: {failed}")
            else:
                push_vectorstore_to_hf_hub()

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _index_missing_files)
    loop.run_in_executor(None, _backfill_tables)
    loop.run_in_executor(None, _backfill_images)
    logger.info("HTTP server ready.")


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


@app.get("/debug")
async def get_debug():
    """Diagnostic endpoint — shows startup config and current VS/data state."""
    data_files = [
        f.name for f in Path(DATA_DIR).iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    vs_files = []
    try:
        vs_files = list(Path(VECTORSTORE_DIR).rglob("*"))
        vs_files = [str(p.relative_to(VECTORSTORE_DIR)) for p in vs_files if p.is_file()]
    except Exception:
        pass
    return {
        "is_hf_space": _IS_HF_SPACE,
        "hf_dataset_repo_set": bool(HF_DATASET_REPO),
        "hf_token_set": bool(HF_TOKEN),
        "vs_chunks": vs.total_chunks(),
        "vs_sources": vs.list_sources(),
        "data_dir_files": data_files,
        "vectorstore_files": vs_files,
    }


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
    """Remove a document's embeddings and delete the file from disk and HF Hub."""
    removed_chunks = vs.remove_document(filename)
    ts.remove(filename)
    img_store.remove(filename)
    file_path = Path(DATA_DIR) / filename
    file_path.unlink(missing_ok=True)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, delete_from_hf_hub, filename)
    await loop.run_in_executor(None, push_vectorstore_to_hf_hub)
    await loop.run_in_executor(None, push_tables_to_hf_hub)
    await loop.run_in_executor(None, push_images_to_hf_hub)
    if removed_chunks > 0:
        return {"message": f"Removed '{filename}' ({removed_chunks} chunks)."}
    else:
        raise HTTPException(404, f"No indexed chunks found for '{filename}'.")


@app.post("/reextract")
async def reextract_all():
    """Re-run table and image extraction on all files in DATA_DIR. No re-embedding."""
    files = [f for f in Path(DATA_DIR).iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    results = {}
    for f in files:
        source_name = f.name
        tables_saved = 0
        images_saved = 0
        try:
            dfs = _extract_tables(str(f))
            ts.save(source_name, dfs)
            tables_saved = len(dfs)
        except Exception as e:
            logger.warning(f"Table reextract failed for '{source_name}': {e}")
        try:
            images = extract_images(str(f))
            img_store.save(source_name, images)
            images_saved = len(images)
        except Exception as e:
            logger.warning(f"Image reextract failed for '{source_name}': {e}")
        results[source_name] = {"tables": tables_saved, "images": images_saved}
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, push_tables_to_hf_hub)
    loop.run_in_executor(None, push_images_to_hf_hub)
    return {"results": results}


@app.delete("/documents")
async def delete_all_documents():
    """Remove ALL embeddings and delete all user files from disk and HF Hub."""
    filenames = [f.name for f in Path(DATA_DIR).iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    removed = vs.clear_all()
    ts.clear_all()
    img_store.clear_all()
    for name in filenames:
        (Path(DATA_DIR) / name).unlink(missing_ok=True)
    loop = asyncio.get_running_loop()
    for name in filenames:
        await loop.run_in_executor(None, delete_from_hf_hub, name)
    await loop.run_in_executor(None, push_vectorstore_to_hf_hub)
    await loop.run_in_executor(None, push_tables_to_hf_hub)
    await loop.run_in_executor(None, push_images_to_hf_hub)
    return {"message": f"Removed {removed} indexed chunks and {len(filenames)} file(s).", "chunks_removed": removed}


# ─── Upload job tracker ──────────────────────────────────────────────────────
_upload_jobs: dict = {}   # filename → {"status": "processing"|"done"|"error", ...}
_upload_lock = threading.Lock()


def _index_background(filename: str, save_path: str):
    """Runs in a background thread: index file, persist to HF Hub, update job status."""
    def _set_phase(msg: str):
        with _upload_lock:
            _upload_jobs[filename]["phase"] = msg
    try:
        _set_phase("parsing document…")
        chunks = process_document_chunked(save_path)
        total = len(chunks)
        source_name = Path(save_path).name
        vs.remove_document(source_name)

        # Embed and upsert in batches of 150 so we can report live progress and
        # avoid one massive blocking encode() call (critical on HF Space CPU).
        EMBED_BATCH = 150
        done = 0
        for i in range(0, max(total, 1), EMBED_BATCH):
            batch = chunks[i : i + EMBED_BATCH]
            end = min(i + EMBED_BATCH, total)
            _set_phase(f"embedding chunks {i + 1}–{end} of {total}…")
            vs.add_documents(batch, source_name, chunk_offset=i)
            done += len(batch)

        n_chunks = done

        _set_phase("extracting tables…")
        try:
            dfs = _extract_tables(save_path)
            ts.save(source_name, dfs)
            if dfs:
                logger.info(f"TableStore: saved {len(dfs)} table(s) for '{source_name}'")
        except Exception as e:
            logger.warning(f"Table extraction failed for '{source_name}': {e}")

        _set_phase("extracting images…")
        try:
            images = extract_images(save_path)
            img_store.save(source_name, images)
            if images:
                logger.info(f"ImageStore: saved {len(images)} image(s) for '{source_name}'")
        except Exception as e:
            logger.warning(f"Image extraction failed for '{source_name}': {e}")

        # Mark done IMMEDIATELY so the frontend poll resolves without waiting for
        # the (potentially slow) HF Hub push that follows.
        with _upload_lock:
            _upload_jobs[filename] = {
                "status": "done",
                "message": f"Uploaded and indexed '{filename}' ({n_chunks} chunks).",
                "chunks": n_chunks,
            }
        logger.info(f"Background index done: '{filename}' — {n_chunks} chunks")
        # Push to HF Hub after marking done so a slow upload doesn't block status.
        push_to_hf_hub(filename)
        push_vectorstore_to_hf_hub()
        push_tables_to_hf_hub()
        push_images_to_hf_hub()
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

_ALL_DOCS_SUMMARY_PATTERNS = [
    "summarize each doc", "summarize all doc", "summarize every doc",
    "summarize each file", "summarize all file", "summarize every file",
    "summary of each", "summary of all doc", "summary of every doc",
    "overview of each", "overview of all doc",
    "describe each doc", "describe all doc", "describe every doc",
    "bullet point each", "bullet points for each",
    "summarize the doc", "summarize the file",
    "summarize all the doc", "summarize all the file",
    "give me a summary of each", "give me a summary of all",
]

_DOCS_LIST_PATTERNS = [
    "how many doc", "how many file", "list doc", "list file",
    "what doc", "what file", "which doc", "which file",
    "show doc", "show file", "what are the doc", "what are the file",
    "what is indexed", "what is uploaded", "what have you indexed",
    "what have you uploaded", "what documents do you have",
    "what files do you have", "tell me the doc", "tell me the file",
    "name the doc", "name the file",
    # file-type specific
    "how many docx", "how many xlsx", "how many csv", "how many pdf",
    "how many image", "how many txt", "how many png", "how many jpg",
    "list docx", "list xlsx", "list csv", "list pdf", "list image", "list txt",
    "show docx", "show xlsx", "show csv", "show pdf", "show image", "show txt",
    "what docx", "what xlsx", "what csv", "what pdf",
    "which docx", "which xlsx", "which csv", "which pdf",
    "any docx", "any xlsx", "any csv", "any pdf", "any image",
    "excel file", "word file", "spreadsheet", "word document",
]


def _docs_list_response() -> str | None:
    """Return a formatted list of indexed documents, or None if none are indexed."""
    sources = vs.list_sources()
    if not sources:
        return None
    lines = "\n".join(f"- {s}" for s in sources)
    # type breakdown
    from collections import Counter
    ext_counts = Counter(Path(s).suffix.lower() for s in sources)
    breakdown = ", ".join(f"{cnt} {ext}" for ext, cnt in sorted(ext_counts.items()))
    return (
        f"There are **{len(sources)}** indexed document(s) ({breakdown}):\n\n"
        f"{lines}"
    )

def _is_all_docs_summary(text: str) -> bool:
    normalized = text.strip().lower().rstrip("!?.,")
    return any(p in normalized for p in _ALL_DOCS_SUMMARY_PATTERNS)

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
    for pattern in _DOCS_LIST_PATTERNS:
        if pattern in normalized:
            return _docs_list_response()
    return None



def _llm_extract_dataframes(filepath: str) -> list:
    """LLM-based table extraction fallback for images and PDFs where the heuristic fails."""
    import io as _io
    import pandas as _pd
    from pathlib import Path as _Path
    ext = _Path(filepath).suffix.lower()
    ocr_text = ""
    try:
        if ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}:
            from utils.document_processor import ocr_image
            from PIL import Image
            ocr_text = ocr_image(Image.open(filepath).convert("RGB"))
        elif ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            ocr_text = "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        logger.warning(f"LLM table fallback OCR failed for '{filepath}': {e}")
        return []

    if not ocr_text.strip():
        return []

    prompt = (
        "The following is text extracted from a document. "
        "If it contains a table, output ONLY a valid CSV representation — "
        "no explanation, no markdown fences, just raw CSV rows. "
        "Use clean column names in the header row. Strip leading row/line numbers. "
        "If no table is present, output exactly: NO_TABLE\n\n"
        f"{ocr_text}"
    )
    try:
        csv_text = _call_llm([{"role": "user", "content": prompt}]).strip()
        if not csv_text or csv_text.startswith("NO_TABLE"):
            return []
        if csv_text.startswith("```"):
            csv_text = "\n".join(l for l in csv_text.splitlines() if not l.startswith("```"))
        df = _pd.read_csv(_io.StringIO(csv_text.strip()))
        return [df] if not df.empty else []
    except Exception as e:
        logger.warning(f"LLM table fallback parse failed for '{filepath}': {e}")
        return []


def _extract_tables(filepath: str) -> list:
    """Extract DataFrames from a file, falling back to LLM parsing for images/PDFs."""
    dfs = extract_dataframes(filepath)
    if not dfs:
        ext = Path(filepath).suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".pdf"}:
            dfs = _llm_extract_dataframes(filepath)
    return dfs


def _call_llm(messages) -> str:
    """Direct LLM call with no RAG context. Falls back to HF Inference on Groq rate-limit."""
    if BACKEND == "groq":
        try:
            resp = rag._client.chat.completions.create(
                model=rag.model, messages=messages, temperature=0.0,
            )
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e).lower()
            if ("429" in msg or "rate_limit" in msg or "rate limit" in msg) and HF_TOKEN:
                logger.warning("Groq rate limit in _call_llm — falling back to HF Inference")
                from huggingface_hub import InferenceClient
                from utils.rag_engine import DEFAULT_HF_MODEL
                client = InferenceClient(token=HF_TOKEN)
                resp = client.chat_completion(
                    model=os.environ.get("HF_MODEL", DEFAULT_HF_MODEL),
                    messages=messages, temperature=0.01, max_tokens=512,
                )
                return resp.choices[0].message.content
            raise
    elif BACKEND == "hf":
        resp = rag._client.chat_completion(
            model=rag.model, messages=messages, temperature=0.01, max_tokens=2048,
        )
        return resp.choices[0].message.content
    else:
        response = rag._client.chat(
            model=rag.model, messages=messages, options={"temperature": 0.0},
        )
        return response["message"]["content"]


def _strip_sql_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:])
        if "```" in raw:
            raw = raw[: raw.rfind("```")]
    return raw.strip()


def _select_relevant_tables(question: str, schema_info: list) -> list:
    """Return the most relevant table(s) based on keyword overlap with column names and sample data."""
    import re as _re
    if len(schema_info) <= 1:
        return schema_info
    q_words = set(_re.sub(r"[^a-z0-9]", " ", question.lower()).split())
    scores = []
    for s in schema_info:
        col_words = set(_re.sub(r"[^a-z0-9]", " ", " ".join(s["numeric_cols"] + s["text_cols"]).lower()).split())
        sample_words = set(_re.sub(r"[^a-z0-9]", " ", s["sample"].lower()).split())
        score = len(q_words & col_words) + 0.3 * len(q_words & sample_words)
        scores.append(score)
    max_score = max(scores)
    if max_score == 0:
        return schema_info
    sorted_scores = sorted(scores, reverse=True)
    # Narrow to best table only when it's clearly dominant
    if len(sorted_scores) >= 2 and sorted_scores[0] >= 2 * sorted_scores[1] + 0.01:
        return [schema_info[scores.index(max_score)]]
    return [s for s, sc in zip(schema_info, scores) if sc >= max_score * 0.5]


def _question_to_sql(question: str, schema_info: list, conn) -> tuple[str, list, list] | None:
    """Convert a natural language question to SQL, execute it, and return (sql, rows, col_names).

    Returns None if SQL generation or execution fails after retries.
    Uses the table schema to build a structured prompt that prevents column confusion.
    """
    import pandas as pd

    # Narrow to the most relevant table(s) to prevent cross-table column confusion
    relevant = _select_relevant_tables(question, schema_info)

    schema_text = "\n\n".join(
        f"Table `{s['table_name']}` (source: {s['source']}, {s['nrows']} rows)\n"
        f"  Numeric columns (REAL — safe for SUM/AVG/MIN/MAX): {', '.join(s['numeric_cols']) or 'none'}\n"
        f"  Text columns (strings only — NEVER use in SUM/AVG/MIN/MAX): {', '.join(s['text_cols'])}\n"
        f"  Sample rows:\n{s['sample']}"
        for s in relevant
    )
    table_col_blocks = "\n".join(
        f"  `{s['table_name']}` valid columns: "
        + ", ".join(f"`{c}`" for c in s["numeric_cols"] + s["text_cols"])
        for s in relevant
    )
    # Build dynamic warnings for columns that look like OCR-garbage text columns
    text_col_warnings = []
    for s in relevant:
        bad_cols = [c for c in s["text_cols"] if c.lower() in ("balance", "total", "amount", "price")]
        if bad_cols:
            text_col_warnings.append(
                f"WARNING: {', '.join(bad_cols)} in `{s['table_name']}` are TEXT (OCR output with symbols like '$', '€', ',') — "
                f"they CANNOT be summed. Use {', '.join(s['numeric_cols']) or 'numeric columns'} instead."
            )
    warnings_block = ("\n".join(text_col_warnings) + "\n\n") if text_col_warnings else ""

    sql_prompt = (
        "SQLite database tables:\n\n"
        + schema_text
        + "\n\nCOLUMN CONSTRAINTS (only these column names are valid — no others exist):\n"
        + table_col_blocks
        + "\n\n"
        + warnings_block
        + "RULES:\n"
        "  1. NEVER use text columns in SUM/AVG/MIN/MAX — they contain strings, not numbers.\n"
        "  2. For date comparisons ALWAYS wrap with DATE(): WHERE DATE(Date)='2022-01-02'\n"
        "     WRONG: WHERE Date='2022-01-02'   CORRECT: WHERE DATE(Date)='2022-01-02'\n"
        "  3. For spending/expense queries on a bank table: filter Credit < 0 (negative = debit/spending)\n"
        "  4. ONLY filter on columns explicitly mentioned in the question. Do NOT infer extra filters from sample data.\n"
        "     WRONG (question says 'all items'): WHERE LOWER(Item)='apple' AND LOWER(Sales_Rep)='william'\n"
        "     CORRECT:                           WHERE LOWER(Sales_Rep)='william'\n"
        "  5. Always use LOWER() for text column comparisons.\n"
        "     For exact names: WHERE LOWER(Sales_Rep)='william'\n"
        "     For categories/keywords: WHERE LOWER(Description) LIKE '%grocery%'\n"
        "     NEVER use bare equality for text without LOWER().\n"
        "\n"
        "Query patterns:\n"
        "  Specific day:           WHERE DATE(Date)='2022-01-02'\n"
        "  Month filter:           WHERE strftime('%Y-%m', Date)='2026-03'\n"
        "  Exact text filter:      WHERE LOWER(Sales_Rep)='william'\n"
        "  Category/keyword:       WHERE LOWER(Description) LIKE '%grocery%'\n"
        "  All items for person:   SELECT SUM(Sales) FROM tbl WHERE LOWER(Sales_Rep)='william'\n"
        "  Numeric aggregation:    SELECT SUM(Sales) FROM tbl WHERE ...\n"
        "  Spending total:         SELECT SUM(Credit) FROM tbl WHERE strftime('%Y-%m', Date)='2026-03' AND Credit < 0\n"
        "  Spending by category:   SELECT SUM(Credit) FROM tbl WHERE strftime('%Y-%m', Date)='2026-03' AND LOWER(Description) LIKE '%grocery%' AND Credit < 0\n"
        "  Spending list:          SELECT Date, Description, Credit FROM tbl WHERE Credit < 0 ORDER BY Credit\n"
        "\n"
        f"Question: {question}\n\n"
        "Write one SQLite SELECT statement. OUTPUT ONLY THE SQL — no markdown, no explanation.\n"
    )
    sql_messages = [
        {"role": "system", "content": "Output only a valid SQLite SELECT statement. No markdown. No explanation."},
        {"role": "user", "content": sql_prompt},
    ]

    for attempt in range(2):
        try:
            llm_out = _call_llm(sql_messages)
        except Exception as e:
            logger.warning(f"LLM SQL generation failed: {e}")
            return None
        if not llm_out:
            return None
        sql = _strip_sql_fences(llm_out.strip())
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            col_names = [d[0] for d in cursor.description] if cursor.description else []
            return sql, rows, col_names
        except Exception as e:
            logger.warning(f"SQL exec failed (attempt {attempt + 1}): {e}\nSQL: {sql}")
            if attempt == 0:
                sql_messages = sql_messages + [
                    {"role": "assistant", "content": sql},
                    {"role": "user", "content": f"Error: {e}\nReturn only the corrected SQL query."},
                ]
    return None


_TABLE_INTENT_RE = re.compile(
    r"\b(sum|total|average|avg|mean|maximum|minimum|count|how much|how many|"
    r"calculate|tallest|largest|smallest|highest|lowest|most|least|"
    r"(?:max|min)(?!\s+\d)|"
    r"per (month|year|day|week|item|person|category)|"
    r"(sales|revenue|profit|cost|price|amount|balance|credit|debit|spendings?|paid|owe)\b.*\b(of|for|by|in|per)\b|"
    r"\b(of|for|by|in)\b.*\b(sales|revenue|profit|cost|price|amount|balance|credit|debit|spendings?))\b",
    re.IGNORECASE,
)


def _is_table_question(question: str) -> bool:
    """Return True only if the question is asking for quantitative/analytical data from tables."""
    return bool(_TABLE_INTENT_RE.search(question))


def _run_table_query(question: str, source_filter=None) -> tuple[str, str] | None:
    """Answer a question using stored tables via text-to-SQL + LLM synthesis. Returns (answer, sql) or None."""
    import pandas as pd

    if source_filter:
        sources = source_filter
    else:
        # Use all files in DATA_DIR, not just embedded ones
        sources = [f.name for f in Path(DATA_DIR).iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    # On-demand extraction for sources not yet attempted
    for src in sources:
        if not ts.was_attempted(src):
            fp = Path(DATA_DIR) / src
            if fp.exists():
                try:
                    extracted = _extract_tables(str(fp))
                    ts.save(src, extracted)
                except Exception as e:
                    logger.warning(f"On-demand table extraction failed for '{src}': {e}")
                    ts.save(src, [])

    conn, schema_info = ts.load_into_memory(sources)
    if not schema_info:
        conn.close()
        return None

    result = _question_to_sql(question, schema_info, conn)

    if result is None:
        conn.close()
        return None

    sql, rows, col_names = result

    if not rows:
        conn.close()
        return "No matching data found in the tables.", sql

    result_df = pd.DataFrame(rows, columns=col_names) if col_names else pd.DataFrame(rows)
    result_str = result_df.to_string(index=False)

    # When result is a single aggregate value, fetch the underlying detail rows
    detail_str = ""
    if len(rows) == 1 and len(col_names) == 1:
        import re as _re
        m = _re.search(r"(FROM\s+\S+(?:\s+WHERE\s+.+)?)", sql, _re.IGNORECASE | _re.DOTALL)
        if m:
            detail_sql = f"SELECT * {m.group(1).rstrip(';')}"
            try:
                dcursor = conn.execute(detail_sql)
                drows = dcursor.fetchall()
                dcols = [d[0] for d in dcursor.description]
                if drows:
                    ddf = pd.DataFrame(drows, columns=dcols)
                    detail_str = f"\n\nUnderlying rows:\n{ddf.to_string(index=False)}"
            except Exception:
                pass

    conn.close()

    # For a single aggregate value: return it directly — no synthesis LLM to avoid misinterpretation
    if len(rows) == 1 and len(col_names) == 1:
        raw_val = rows[0][0]
        col_label = col_names[0]
        answer = f"**{col_label}:** {raw_val}"
        if detail_str:
            answer += detail_str
        return answer, sql

    answer_messages = [
        {"role": "system", "content": "You are a helpful data analyst. Answer concisely based on the query results. NEVER recalculate or modify the numbers — report them exactly as returned by SQL."},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"SQL: {sql}\n\n"
                f"Results:\n{result_str}"
                f"{detail_str}\n\n"
                "Report the exact values from the results. Do not round, negate, or recalculate any numbers."
            ),
        },
    ]
    answer = _call_llm(answer_messages).strip()
    return (answer if answer else result_str), sql


@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """Query the RAG system."""
    try:
        # Short-circuit chitchat / greetings — don't pollute with RAG results
        chitchat_answer = _chitchat_response(req.question)
        if chitchat_answer is not None:
            return QueryResponse(answer=chitchat_answer, sources=[])
        # Doc-list question but no docs indexed yet
        normalized_q = req.question.strip().lower().rstrip("!?.,")
        if any(p in normalized_q for p in _DOCS_LIST_PATTERNS):
            return QueryResponse(answer="No documents are indexed yet. Please upload some documents first.", sources=[])

        if vs.total_chunks() == 0:
            return QueryResponse(answer="No documents are indexed yet. Please upload some documents first.", sources=[])

        # Run all blocking work (embedding + LLM) in a thread executor
        def _run_query():
            sf = req.source_filter or None

            table_result = _run_table_query(req.question, sf) if _is_table_question(req.question) else None
            if table_result:
                table_answer, table_sql = table_result
                sources = sf if sf else vs.list_sources()
                memory.add("user", req.question)
                memory.add("assistant", f"{table_answer}\n\n[SQL used: {table_sql}]")
                return table_answer, table_sql, "table_query", list(sources), estimate_tokens(req.question), estimate_tokens(table_answer), 0

            # "Summarize each/all doc" — fetch chunks from every source so no doc is missed
            if _is_all_docs_summary(req.question) and not sf:
                results = vs.query_per_source(req.question, n_per_source=2)
            else:
                results = vs.query(req.question, n_results=req.n_results, source_filter=sf)

            relevant = [r for r in results if r.get("distance", 2.0) <= RELEVANCE_THRESHOLD]
            chunks_used = len(relevant)
            if relevant:
                best_dist = min(r.get("distance", 2.0) for r in relevant)
                source_chunks = [r for r in relevant if r.get("distance", 2.0) <= best_dist + 0.3]
            else:
                source_chunks = results if _is_all_docs_summary(req.question) else []
            sources = list({r["metadata"].get("source", "") for r in source_chunks})
            parts = []
            for token in rag.query(req.question, memory, n_results=req.n_results, temperature=req.temperature, stream=False, source_filter=sf, pre_fetched_results=results):
                parts.append(token)
            answer = "".join(parts)
            tokens_user = estimate_tokens(req.question)
            tokens_assistant = estimate_tokens(answer)
            return answer, "", "rag", sources, tokens_user, tokens_assistant, chunks_used

        loop = asyncio.get_running_loop()
        answer, sql_query, answer_method, sources, tokens_user, tokens_assistant, chunks_used = await loop.run_in_executor(None, _run_query)

        return QueryResponse(
            answer=answer,
            sources=sources,
            tokens_user=tokens_user,
            tokens_assistant=tokens_assistant,
            chunks_used=chunks_used,
            sql_query=sql_query,
            answer_method=answer_method,
        )
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
