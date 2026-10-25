"""Transport layer: everything that touches the network lives here.

Two failure classes are defined in this module because the retry policy is the
only thing that distinguishes them:

  TransientError -> the same request may succeed later (429, 5xx, socket reset)
  FatalError     -> retrying is pointless (401, 400, unknown model)

The agent loop never sees these. By the time control returns to it, either a
well-formed Response came back or the exception is unrecoverable.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import httpx

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class TransientError(Exception):
    """Retryable: the same request might succeed on a later attempt."""


class FatalError(Exception):
    """Not retryable: bad credentials, bad request shape, bad model name."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Response:
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


class LLM(Protocol):
    def complete(self, messages: list[dict], tools: list[dict] | None) -> Response: ...


def _parse_body(body: dict[str, Any]) -> Response:
    """Turn a chat-completions payload into a Response, or a TransientError.

    A structurally broken body is treated as transient rather than fatal: the
    provider returned something, so asking again is reasonable. This is the
    difference between "your request is wrong" (fatal) and "their answer is
    wrong" (transient).
    """
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TransientError(f"malformed response body: {str(body)[:200]}") from exc

    usage = body.get("usage") or {}
    calls = []
    for raw in message.get("tool_calls") or []:
        fn = raw.get("function") or {}
        name = fn.get("name")
        if not name:
            raise TransientError(f"tool call without a name: {str(raw)[:200]}")
        calls.append(
            {
                "id": raw.get("id") or f"call_{len(calls)}",
                "name": name,
                # Keep arguments as the raw string. Parsing them is the
                # registry's job, and a malformed string is recoverable there.
                "arguments": fn.get("arguments") or "{}",
            }
        )

    return Response(
        content=message.get("content"),
        tool_calls=calls,
        usage=Usage(
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            completion_tokens=usage.get("completion_tokens", 0) or 0,
        ),
    )


class OpenAIChatLLM:
    """Minimal client for any OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = lambda _msg: None,
    ) -> None:
        self.model = model
        self.max_attempts = max_attempts
        self._sleep = sleep
        self._log = log
        self._client = httpx.Client(
            base_url=(base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/"),
            headers={
                "Authorization": f"Bearer {api_key or os.getenv('OPENAI_API_KEY', '')}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def complete(self, messages: list[dict], tools: list[dict] | None) -> Response:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            payload["tools"] = tools

        last: TransientError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._once(payload)
            except TransientError as exc:
                last = exc
                if attempt == self.max_attempts:
                    break
                # Exponential backoff with jitter. Jitter matters when several
                # workers hit the same 429 at the same instant.
                delay = min(2 ** (attempt - 1), 8) * (1 + random.random() * 0.25)
                self._log(f"transient failure (attempt {attempt}/{self.max_attempts}), "
                          f"retrying in {delay:.1f}s: {exc}")
                self._sleep(delay)

        raise FatalError(f"gave up after {self.max_attempts} attempts: {last}") from last

    def _once(self, payload: dict[str, Any]) -> Response:
        try:
            resp = self._client.post("/chat/completions", json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientError(f"network: {type(exc).__name__}: {exc}") from exc

        if resp.status_code in RETRYABLE_STATUS:
            raise TransientError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise FatalError(f"HTTP {resp.status_code}: {resp.text[:400]}")

        try:
            return _parse_body(resp.json())
        except ValueError as exc:
            raise TransientError(f"response body was not JSON: {resp.text[:200]}") from exc


class ScriptedLLM:
    """Deterministic stand-in used by every test and by the offline demo.

    Each item in `script` is either a Response to return or an Exception to
    raise. That is enough to reproduce retry storms, malformed tool calls and
    model self-correction without a network or an API key.
    """

    def __init__(self, script: list[Response | Exception]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict], tools: list[dict] | None) -> Response:
        self.calls.append({"messages": messages, "tools": tools})
        if not self._script:
            raise AssertionError(f"script exhausted after {len(self.calls)} calls")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def call_tool(name: str, arguments: dict[str, Any] | str, call_id: str = "call_0") -> Response:
    """Shorthand for building a scripted tool-calling response."""
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return Response(tool_calls=[{"id": call_id, "name": name, "arguments": arguments}])


def say(text: str) -> Response:
    """Shorthand for building a scripted plain-text response."""
    return Response(content=text)
