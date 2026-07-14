"""Thin client for any OpenAI-compatible chat completion server.

Works unchanged against llama.cpp's llama-server (default), Ollama, vLLM,
or a hosted endpoint — the serving layer is swappable via LANG2ACT_BASE_URL.

Design notes for small local models:
- `json_schema` (constrained decoding) is used for every structured reply,
  so a 3B model can never produce an unparseable action.
- Timeouts are generous: CPU inference on a laptop is slow by design here.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_BASE_URL = os.environ.get("LANG2ACT_BASE_URL", "http://127.0.0.1:8080/v1")
DEFAULT_MODEL = os.environ.get("LANG2ACT_MODEL", "qwen2.5-vl-3b")
REQUEST_TIMEOUT_S = float(os.environ.get("LANG2ACT_TIMEOUT_S", "600"))


def image_content(rgb_array) -> dict:
    """Encode an HxWx3 uint8 numpy array as an OpenAI image_url content part."""
    from PIL import Image

    img = Image.fromarray(rgb_array)
    # Downscale to bound vision-token count (and CPU encode time).
    img.thumbnail((448, 448))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def text_content(text: str) -> dict:
    return {"type": "text", "text": text}


@dataclass
class ChatResult:
    text: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int

    def json(self) -> Any:
        return json.loads(self.text)


@dataclass
class LLMClient:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    max_tokens: int = 768
    _client: httpx.Client = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self):
        self._client = httpx.Client(timeout=httpx.Timeout(REQUEST_TIMEOUT_S, connect=10))

    def chat(
        self,
        messages: list[dict],
        json_schema: dict | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """One chat completion. If json_schema is given, the server constrains
        decoding so the reply is guaranteed to parse against it."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": json_schema, "strict": True},
            }
        t0 = time.monotonic()
        resp = self._client.post(f"{self.base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") or {}
        return ChatResult(
            text=data["choices"][0]["message"]["content"],
            latency_s=time.monotonic() - t0,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    def health(self) -> bool:
        try:
            r = self._client.get(self.base_url.removesuffix("/v1") + "/health", timeout=5)
            return r.status_code == 200
        except httpx.HTTPError:
            return False
