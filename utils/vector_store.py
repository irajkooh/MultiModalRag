"""
ChromaDB vector store manager with persistent storage.
Handles embedding via sentence-transformers with automatic
CUDA / MPS (Apple Silicon) / CPU device selection.
"""
import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from utils.device import get_device, device_info

logger = logging.getLogger(__name__)

COLLECTION_NAME = "multimodal_rag"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")


class VectorStoreManager:
    def __init__(self, persist_dir: str = "./vectorstore"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        # Detect best available device (CUDA > MPS > CPU)
        self.device = get_device()
        info = device_info()
        logger.info(f"Embedding device: {info['label']}")

        # Persistent Chroma client
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Load embedding model onto selected device
        logger.info(f"Loading embedding model: {EMBED_MODEL} on {self.device}")
        self.embedder = SentenceTransformer(EMBED_MODEL, device=self.device)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Vector store ready — {self.collection.count()} chunks on {self.device}")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # normalize_embeddings=True improves cosine similarity quality
        return self.embedder.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            device=self.device,
        ).tolist()

    def add_documents(self, chunks: List[Dict[str, Any]], source_name: str) -> int:
        """Add document chunks. Returns number of chunks added."""
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        # Build unique IDs: source + chunk index
        ids = [
            f"{source_name}__chunk_{i}"
            for i in range(len(chunks))
        ]

        # Strip non-serializable fields from metadata (e.g. large base64 images)
        clean_metas = []
        for m in metadatas:
            clean = {k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))}
            clean_metas.append(clean)

        embeddings = self._embed(texts)

        # Upsert in batches of 100
        batch = 100
        for i in range(0, len(texts), batch):
            self.collection.upsert(
                ids=ids[i:i+batch],
                embeddings=embeddings[i:i+batch],
                documents=texts[i:i+batch],
                metadatas=clean_metas[i:i+batch],
            )
        
        logger.info(f"Added {len(chunks)} chunks for '{source_name}'")
        return len(chunks)

    def remove_document(self, source_name: str) -> int:
        """Remove all chunks belonging to a source file."""
        results = self.collection.get(where={"source": source_name})
        ids = results.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
            logger.info(f"Removed {len(ids)} chunks for '{source_name}'")
        return len(ids)

    def clear_all(self) -> int:
        """Remove every chunk from the collection. Returns count removed."""
        count = self.collection.count()
        if count == 0:
            return 0
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)
        logger.info(f"Cleared {count} chunks from collection")
        return count

    def query(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Semantic search. Returns list of {text, metadata, distance}."""
        count = self.collection.count()
        if count == 0:
            return []
        
        n = min(n_results, count)
        embedding = self._embed([query_text])[0]
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            output.append({"text": doc, "metadata": meta, "distance": dist})
        return output

    def list_sources(self) -> List[str]:
        """List all unique source document names."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        sources = set()
        for m in results.get("metadatas", []):
            if m and "source" in m:
                sources.add(m["source"])
        return sorted(sources)

    def total_chunks(self) -> int:
        return self.collection.count()
