"""LLM client. One OpenAI-compatible interface so Ollama, vLLM and hosted APIs are a config change.

A failed call raises LLMError. It is never turned into a wrong answer: the runner records it as an
error and excludes it from accuracy, so a rate-limited baseline cannot inflate a gain.
"""

from __future__ import annotations

import re
import threading
from typing import Protocol

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class LLMError(RuntimeError):
    """Raised when the model did not return a usable completion after retries."""


THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """Drop reasoning blocks emitted by thinking models (Qwen3, DeepSeek-R1). An unterminated block
    means the completion was cut off inside its reasoning; return "" so the caller treats it as empty."""
    if "<think>" in text and "</think>" not in text:
        return ""
    return THINK_RE.sub("", text).strip()


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenAIChat:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "no-key",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 6,
        no_think: bool = False,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.no_think = no_think
        self._client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, system: str, user: str) -> str:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

        retryable = (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError)

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type(retryable),
        )
        def _call() -> str:
            with self._lock:
                self.calls += 1
            content = user + " /no_think" if self.no_think else user
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = strip_think(resp.choices[0].message.content or "")
            if not text:
                raise APIStatusError("empty completion", response=None, body=None)  # type: ignore[arg-type]
            return text

        try:
            return _call()
        except Exception as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e


class FakeLLM:
    """Deterministic stand-in for tests. `script` maps a substring of the user prompt to a reply;
    `default` is returned when nothing matches. Set `fail_on` to raise LLMError for a substring."""

    def __init__(self, script: dict[str, str] | None = None, default: str = "Final answer: unknown",
                 fail_on: str | None = None) -> None:
        self.script = script or {}
        self.default = default
        self.fail_on = fail_on
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.fail_on and self.fail_on in user:
            raise LLMError("simulated failure")
        for key, reply in self.script.items():
            if key in user:
                return reply
        return self.default
