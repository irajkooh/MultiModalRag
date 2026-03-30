"""
RAG engine: retrieves relevant context from vector store,
builds a strict prompt, and queries the LLM.

- If GROQ_API_KEY env var is set → uses Groq API (fast cloud LPU)
- Otherwise → uses Ollama (local)
"""
import os
import logging
from typing import List, Dict, Any, Generator

from utils.vector_store import VectorStoreManager
from utils.memory import ConversationMemory

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_GROQ_MODEL   = "llama-3.3-70b-versatile"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
USE_GROQ     = bool(GROQ_API_KEY)

OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

SYSTEM_PROMPT = """You are a document assistant. Answer questions using ONLY the [CONTEXT] provided.
If the answer is not in the context, respond: "I DON'T KNOW"
Be concise and factual. Cite source and page when available.
"""


def _make_groq_client():
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)


def _make_ollama_client():
    import ollama
    return ollama.Client(host=OLLAMA_HOST)


class RAGEngine:
    def __init__(self, vector_store: VectorStoreManager, model: str = None):
        self.vs = vector_store
        if USE_GROQ:
            self.model  = model or os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
            self._client = _make_groq_client()
            logger.info(f"LLM backend: Groq ({self.model})")
        else:
            self.model  = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
            self._client = _make_ollama_client()
            logger.info(f"LLM backend: Ollama ({self.model})")

    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No relevant documents found."
        parts = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            source   = meta.get("source", "unknown")
            page     = meta.get("page", "")
            doc_type = meta.get("type", "text")
            page_str = f", Page {page}" if page else ""
            type_str = f" [{doc_type}]" if doc_type != "text" else ""
            parts.append(f"[Doc {i} — {source}{page_str}{type_str}]\n{r['text']}")
        return "\n\n---\n\n".join(parts)

    def _build_messages(self, question: str, context: str, memory: ConversationMemory):
        user_message = (
            f"[CONTEXT]\n{context}\n\n"
            f"[QUESTION]\n{question}\n\n"
            "Remember: Answer ONLY from the context above. If not found, say \"I DON'T KNOW\"."
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *memory.get_history_for_prompt(),
            {"role": "user", "content": user_message},
        ]

    def query(
        self,
        question: str,
        memory: ConversationMemory,
        n_results: int = 5,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> Generator[str, None, None]:
        results  = self.vs.query(question, n_results=n_results)
        context  = self._build_context(results)
        messages = self._build_messages(question, context, memory)

        try:
            if USE_GROQ:
                yield from self._query_groq(messages, memory, question, temperature, stream)
            else:
                yield from self._query_ollama(messages, memory, question, temperature, stream)
        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield error_msg

    def _query_groq(self, messages, memory, question, temperature, stream):
        if stream:
            response_text = ""
            with self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            ) as stream_resp:
                for chunk in stream_resp:
                    token = chunk.choices[0].delta.content or ""
                    response_text += token
                    yield token
            memory.add("user", question)
            memory.add("assistant", response_text)
        else:
            resp   = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            answer = resp.choices[0].message.content
            memory.add("user", question)
            memory.add("assistant", answer)
            yield answer

    def _query_ollama(self, messages, memory, question, temperature, stream):
        import ollama as _ollama
        if stream:
            response_text = ""
            stream_resp = self._client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={"temperature": temperature},
            )
            for chunk in stream_resp:
                token = chunk["message"]["content"]
                response_text += token
                yield token
            memory.add("user", question)
            memory.add("assistant", response_text)
        else:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature},
            )
            answer = response["message"]["content"]
            memory.add("user", question)
            memory.add("assistant", answer)
            yield answer

    def list_available_models(self) -> List[str]:
        if USE_GROQ:
            return [self.model]
        try:
            models = self._client.list()
            return [m["name"] for m in models.get("models", [])]
        except Exception:
            return [self.model]
