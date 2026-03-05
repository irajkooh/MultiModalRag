"""
RAG engine: retrieves relevant context from vector store,
builds a strict prompt, and queries Ollama LLM.

OLLAMA_HOST env var controls where Ollama is reached
(default: http://localhost:11434).
This works for both local execution and HF Spaces Docker deployments.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Generator
import ollama

from utils.vector_store import VectorStoreManager
from utils.memory import ConversationMemory

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.2"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_client = ollama.Client(host=OLLAMA_HOST)
SYSTEM_PROMPT = """You are a precise document assistant. Your ONLY job is to answer questions based on the provided document context.

STRICT RULES:
1. ONLY use information explicitly found in the [CONTEXT] section below.
2. If the answer is NOT in the context, respond EXACTLY with: "I DON'T KNOW"
3. Do NOT use any prior knowledge, make assumptions, or speculate.
4. Do NOT be creative. Be factual and concise.
5. Always cite the source document and page number when available.
6. If asked something unrelated to the documents, respond: "I DON'T KNOW"
"""


class RAGEngine:
    def __init__(self, vector_store: VectorStoreManager, model: str = DEFAULT_MODEL):
        self.vs = vector_store
        self.model = model

    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No relevant documents found."
        parts = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            source = meta.get("source", "unknown")
            page = meta.get("page", "")
            doc_type = meta.get("type", "text")
            page_str = f", Page {page}" if page else ""
            type_str = f" [{doc_type}]" if doc_type != "text" else ""
            parts.append(f"[Doc {i} — {source}{page_str}{type_str}]\n{r['text']}")
        return "\n\n---\n\n".join(parts)

    def query(
        self,
        question: str,
        memory: ConversationMemory,
        n_results: int = 5,
        stream: bool = False,
    ) -> Generator[str, None, None]:
        """
        Run RAG: retrieve context → build prompt → call Ollama.
        Yields streamed tokens if stream=True, otherwise yields full response.
        """
        # 1. Retrieve relevant chunks
        results = self.vs.query(question, n_results=n_results)
        context = self._build_context(results)

        # 2. Build messages
        history = memory.get_history_for_prompt()
        
        user_message = f"""[CONTEXT]
{context}

[QUESTION]
{question}

Remember: Answer ONLY from the context above. If not found, say "I DON'T KNOW"."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_message},
        ]

        # 3. Call Ollama
        try:
            if stream:
                response_text = ""
                stream_resp = _client.chat(
                    model=self.model,
                    messages=messages,
                    stream=True,
                )
                for chunk in stream_resp:
                    token = chunk["message"]["content"]
                    response_text += token
                    yield token
                # Store in memory after full response
                memory.add("user", question)
                memory.add("assistant", response_text)
            else:
                response = _client.chat(model=self.model, messages=messages)
                answer = response["message"]["content"]
                memory.add("user", question)
                memory.add("assistant", answer)
                yield answer
        except ollama.ResponseError as e:
            error_msg = f"⚠️ Ollama error: {e}. Make sure Ollama is running and model '{self.model}' is pulled."
            logger.error(error_msg)
            yield error_msg
        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield error_msg

    def list_available_models(self) -> List[str]:
        """List locally available Ollama models."""
        try:
            models = _client.list()
            return [m["name"] for m in models.get("models", [])]
        except Exception:
            return [DEFAULT_MODEL]
