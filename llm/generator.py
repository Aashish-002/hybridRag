"""
llm/generator.py
================
LLM response generation layer with support for LM Studio, Ollama, and OpenAI.

Purpose:
    - Accept an assembled context + question
    - Format a structured prompt
    - Call the configured LLM backend via the OpenAI-compatible API
    - Return the generated answer (streaming or batch)

Supported Backends (all expose /v1/chat/completions):
    - Ollama      (local, http://localhost:11434/v1)
    - LM Studio   (local, http://localhost:1234/v1)
    - OpenAI      (cloud, https://api.openai.com/v1)
    - Any OpenAI-compatible API (vLLM, LiteLLM, etc.)

Prompt Template:
    System: "You are a helpful assistant. Answer only from the provided context."
    User:   "Context: {context}\n\nQuestion: {question}\n\nAnswer:"

Best Practices:
    - Low temperature (0.0–0.2) for factual document Q&A
    - Include "I could not find this information" fallback in the system prompt
    - Separate context from question clearly in the prompt
    - Streaming mode for better UX (tokens appear as they are generated)

Performance Considerations:
    - First token latency: 200–500ms on local GPU, 1–3s on CPU
    - Streaming hides latency — user sees output immediately
    - max_tokens=1024 is sufficient for most answers; increase for summaries
"""

from __future__ import annotations

import time
from typing import Generator, Iterator, List, Optional

from langchain_core.documents import Document
from openai import OpenAI

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful, precise document assistant.
Your ONLY source of information is the context provided below.
Rules:
- Answer the question using ONLY information from the provided context.
- If the answer is not in the context, respond with:
  "I could not find this information in the provided documents."
- Be concise and factual. Cite the source document when possible.
- Do NOT add information from your training data or general knowledge."""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# ResponseGenerator
# ---------------------------------------------------------------------------

class ResponseGenerator:
    """
    Generates answers from context + question using an LLM.

    Args:
        provider:    'ollama', 'lmstudio', or 'openai'.
        base_url:    API base URL.
        api_key:     API key string.
        model_name:  Model identifier.
        temperature: Sampling temperature.
        max_tokens:  Max tokens in response.
        streaming:   If True, yield tokens as a generator.
    """

    def __init__(
        self,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        streaming: bool | None = None,
    ) -> None:
        cfg = settings.llm
        self.provider = provider or cfg.provider
        self.base_url = base_url or cfg.base_url
        self.api_key = api_key or cfg.api_key
        self.model_name = model_name or cfg.model_name
        self.temperature = temperature if temperature is not None else cfg.temperature
        self.max_tokens = max_tokens or cfg.max_tokens
        self.streaming = streaming if streaming is not None else cfg.streaming
        self._client: Optional[OpenAI] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> OpenAI:
        """Lazy-create OpenAI client (reused across calls)."""
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=settings.llm.timeout,
            )
            logger.debug(
                "LLM client initialised: provider=%s, model=%s, url=%s",
                self.provider, self.model_name, self.base_url,
            )
        return self._client

    def _build_prompt(self, context: str, question: str) -> List[dict]:
        """Build the messages list for the chat completion API."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                context=context, question=question
            )},
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, context: str, question: str) -> str:
        """
        Generate a non-streaming response.

        Args:
            context:  Assembled context string from retrieved chunks.
            question: User's question.

        Returns:
            Full answer string.
        """
        client = self._get_client()
        messages = self._build_prompt(context, question)

        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )
            answer = response.choices[0].message.content.strip()
            elapsed = time.perf_counter() - t0
            logger.info(
                "LLM generated %d tokens in %.3fs (model=%s).",
                response.usage.completion_tokens if response.usage else -1,
                elapsed, self.model_name,
            )
            return answer
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc, exc_info=True)
            raise

    def generate_stream(self, context: str, question: str) -> Iterator[str]:
        """
        Generate a streaming response, yielding token strings as they arrive.

        Use with Streamlit's st.write_stream() for real-time display.

        Args:
            context:  Assembled context string.
            question: User's question.

        Yields:
            Token strings as they are generated by the LLM.
        """
        client = self._get_client()
        messages = self._build_prompt(context, question)

        logger.info("Starting streaming generation (model=%s) …", self.model_name)
        try:
            with client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
        except Exception as exc:
            logger.error("Streaming LLM failed: %s", exc, exc_info=True)
            yield "\n\n⚠️ LLM error: " + str(exc)

    def generate_with_memory(
        self,
        context: str,
        question: str,
        history: List[dict] | None = None,
    ) -> str:
        """
        Generate a response incorporating conversation history.

        Args:
            context:  Retrieved context.
            question: Current question.
            history:  List of {"role": "user"/"assistant", "content": "..."} dicts.
                      Most recent pair is last in the list.

        Returns:
            Full answer string.
        """
        client = self._get_client()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add conversation history (up to memory_k pairs)
        if history:
            k = settings.conversation_memory_k
            messages.extend(history[-(k * 2):])

        # Add current context + question
        messages.append({
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(context=context, question=question)
        })

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("LLM with memory failed: %s", exc, exc_info=True)
            raise

    def test_connection(self) -> bool:
        """
        Quick health check — send a minimal request to verify the LLM is reachable.

        Returns:
            True if connected, False otherwise.
        """
        try:
            client = self._get_client()
            client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                stream=False,
            )
            return True
        except Exception as exc:
            logger.warning("LLM connection test failed: %s", exc)
            return False
