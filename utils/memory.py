"""
Conversation memory manager with context window overflow handling.
Keeps a sliding window of recent messages and summarizes older ones
when the token budget is exceeded.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Approximate token budget for conversation history (leave room for system + context)
MAX_HISTORY_TOKENS = 2000
AVG_CHARS_PER_TOKEN = 4  # conservative estimate


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // AVG_CHARS_PER_TOKEN)


@dataclass
class Message:
    role: str   # "user" or "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    def token_count(self) -> int:
        return estimate_tokens(self.content)


class ConversationMemory:
    def __init__(self, max_tokens: int = MAX_HISTORY_TOKENS):
        self.max_tokens = max_tokens
        self.messages: List[Message] = []
        self.summary: Optional[str] = None  # compressed older history

    def add(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))
        self._maybe_compress()

    def _total_tokens(self) -> int:
        t = sum(m.token_count() for m in self.messages)
        if self.summary:
            t += estimate_tokens(self.summary)
        return t

    def _maybe_compress(self):
        """If over budget, summarize oldest half of messages."""
        if self._total_tokens() <= self.max_tokens:
            return

        # Keep the last 4 messages always (current exchange)
        keep_n = max(4, len(self.messages) // 2)
        to_compress = self.messages[:-keep_n]
        self.messages = self.messages[-keep_n:]

        if not to_compress:
            return

        # Build a simple bullet-point summary (no LLM call, to avoid circular deps)
        lines = []
        for m in to_compress:
            snippet = m.content[:200].replace("\n", " ")
            lines.append(f"- [{m.role.upper()}]: {snippet}{'...' if len(m.content)>200 else ''}")
        new_summary = "\n".join(lines)

        if self.summary:
            self.summary = f"{self.summary}\n{new_summary}"
        else:
            self.summary = new_summary

        logger.info(f"Compressed {len(to_compress)} messages. History size: {len(self.messages)}")

    def get_history_for_prompt(self) -> List[Dict[str, str]]:
        """Return message list suitable for Ollama chat API."""
        result = []
        if self.summary:
            result.append({
                "role": "user",
                "content": f"[Previous conversation summary]\n{self.summary}",
            })
            result.append({
                "role": "assistant",
                "content": "I understand the previous context.",
            })
        result.extend(m.to_dict() for m in self.messages)
        return result

    def clear(self):
        self.messages = []
        self.summary = None

    def to_gradio_format(self) -> List[List[Optional[str]]]:
        """Convert to Gradio chatbot format [[user, bot], ...]"""
        pairs = []
        i = 0
        msgs = self.messages
        while i < len(msgs):
            if msgs[i].role == "user":
                user_msg = msgs[i].content
                bot_msg = msgs[i+1].content if (i+1 < len(msgs) and msgs[i+1].role == "assistant") else None
                pairs.append([user_msg, bot_msg])
                i += 2 if bot_msg is not None else 1
            else:
                pairs.append([None, msgs[i].content])
                i += 1
        return pairs
