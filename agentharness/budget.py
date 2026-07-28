"""Token-budget management for long-running agent loops — the context
assembler.

Every additional tool call appends more history to the conversation Claude
sees on the next turn. Left unmanaged, a long agent run can grow past the
model's context window, or just burn far more tokens than the task needs.

Instead of trimming oldest-first, this module embeds the task and every
tool result, scores each result by semantic similarity to the task
(plus recency, success/failure, and size), and trims the lowest-value
results first — so something irrelevant gets cut before something
relevant, regardless of which happened more recently.
"""

from __future__ import annotations

import anthropic
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class TokenBudget:
    """Enforces a token ceiling on a conversation using semantic scoring."""

    _embedder: SentenceTransformer | None = None  # loaded once, shared across instances

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        max_tokens: int,
        keep_recent: int = 2,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.embedding_model_name = embedding_model

    def _get_embedder(self) -> SentenceTransformer:
        # Lazy-loaded: only pay the model-load cost if trimming is ever needed.
        if TokenBudget._embedder is None:
            TokenBudget._embedder = SentenceTransformer(self.embedding_model_name)
        return TokenBudget._embedder

    def count(self, messages: list[dict], system: str) -> int:
        """Count tokens for messages + system prompt, with a rough
        character-based fallback if the count_tokens endpoint isn't available."""
        try:
            response = self.client.messages.count_tokens(
                model=self.model, system=system, messages=messages
            )
            return response.input_tokens
        except Exception:
            text = system + "".join(str(m) for m in messages)
            return len(text) // 4  # rough estimate: ~4 chars per token

    def _score(
        self,
        block: dict,
        position: int,
        total: int,
        task_embedding,
        embedder: SentenceTransformer,
    ) -> float:
        """Lower score = trim first. Combines semantic relevance to the
        task, recency, success/failure, and size."""
        content_str = str(block.get("content", ""))[:500]  # cap embedding input length

        result_embedding = embedder.encode(content_str, convert_to_tensor=True)
        relevance = float(cos_sim(task_embedding, result_embedding)[0][0])  # -1 to 1

        recency = position / max(total - 1, 1)  # 0 = oldest, 1 = newest
        is_error = bool(block.get("is_error"))
        size_penalty = min(len(content_str) / 2000, 1.0)

        # Relevance is weighted heaviest — this is the "smart" part.
        score = (relevance * 0.6) + (recency * 0.25)
        score -= 0.15 if is_error else 0.0
        score -= size_penalty * 0.2

        return score

    def _summarize(self, block: dict) -> str:
        content_str = str(block.get("content", ""))
        gist = content_str[:80].replace("\n", " ")
        status = "failed" if block.get("is_error") else "ok"
        return f"[trimmed tool result, {status}: {gist}...]"

    def enforce(self, messages: list[dict], system: str, task: str) -> tuple[list[dict], int]:
        """Trim the lowest-relevance tool results (see _score) until the
        conversation fits under max_tokens, or nothing trimmable remains.

        `task` is the original user task — used as the relevance anchor
        for scoring every tool result.
        """
        messages = [dict(m) for m in messages]
        token_count = self.count(messages, system)
        if token_count <= self.max_tokens:
            return messages, token_count

        embedder = self._get_embedder()
        task_embedding = embedder.encode(task, convert_to_tensor=True)

        trimmable_end = len(messages) - self.keep_recent
        candidates = []  # (score, block)
        for i in range(max(trimmable_end, 0)):
            content = messages[i].get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    if str(block.get("content", "")).startswith("[trimmed"):
                        continue  # already trimmed
                    score = self._score(block, i, len(messages), task_embedding, embedder)
                    candidates.append((score, block))

        candidates.sort(key=lambda c: c[0])  # lowest relevance first

        for _, block in candidates:
            block["content"] = self._summarize(block)
            token_count = self.count(messages, system)
            if token_count <= self.max_tokens:
                break

        return messages, token_count