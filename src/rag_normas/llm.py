"""OpenAI-compatible chat client with a free-tier fallback chain: Groq -> OpenRouter ``:free`` -> Ollama.

A provider without its key is skipped. Fails over on 408/429/5xx, timeouts and connection errors;
any other 4xx is a bug on our side and is raised as is.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


class LLMError(RuntimeError):
    pass


class AllProvidersFailed(LLMError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("todos os provedores falharam: " + "; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    model: str
    api_key: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Completion:
    text: str
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float
    failed: tuple[str, ...] = ()


RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


def providers_from_env(env: dict[str, str] | None = None) -> list[Provider]:
    e = dict(os.environ if env is None else env)
    chain = [p.strip() for p in e.get("RAG_LLM_PROVIDERS", "groq,openrouter").split(",") if p.strip()]
    out: list[Provider] = []
    for name in chain:
        if name == "groq" and e.get("GROQ_API_KEY"):
            out.append(
                Provider(
                    "groq",
                    "https://api.groq.com/openai/v1",
                    e.get("RAG_GROQ_MODEL", "openai/gpt-oss-20b"),
                    e["GROQ_API_KEY"],
                    {"reasoning_effort": "low"},
                )
            )
        elif name == "openrouter" and e.get("OPENROUTER_API_KEY"):
            model = e.get("RAG_OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
            if not model.endswith(":free"):
                raise ValueError(
                    f"RAG_OPENROUTER_MODEL={model!r} não é :free; este projeto só usa o plano grátis"
                )
            out.append(Provider("openrouter", "https://openrouter.ai/api/v1", model, e["OPENROUTER_API_KEY"]))
        elif name == "ollama":
            base = e.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            # Thinking models spend the whole token budget reasoning unless told not to.
            effort = e.get("RAG_OLLAMA_REASONING", "none")
            extra = {"reasoning_effort": effort} if effort else {}
            out.append(Provider("ollama", f"{base}/v1", e.get("RAG_OLLAMA_MODEL", "qwen3.5:9b"), None, extra))
        elif name not in {"groq", "openrouter", "ollama"}:
            raise ValueError(f"provedor desconhecido em RAG_LLM_PROVIDERS: {name}")
    return out


class ChatClient:
    def __init__(
        self, providers: list[Provider], timeout_s: float | None = None, client: httpx.Client | None = None
    ) -> None:
        # No provider is allowed: search still works, only answering fails (503 on the API).
        self.providers = providers
        timeout = timeout_s if timeout_s is not None else float(os.getenv("RAG_LLM_TIMEOUT_S", "60"))
        self._http = client or httpx.Client(timeout=timeout)

    def complete(
        self, messages: list[dict[str, str]], *, max_tokens: int = 700, temperature: float = 0.0
    ) -> Completion:
        if not self.providers:
            raise LLMError("nenhum provedor de LLM configurado (GROQ_API_KEY, OPENROUTER_API_KEY ou ollama)")
        errors: list[str] = []
        for p in self.providers:
            body: dict[str, Any] = {
                "model": p.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **p.extra,
            }
            headers = {"Authorization": f"Bearer {p.api_key}"} if p.api_key else {}
            start = time.perf_counter()
            try:
                resp = self._http.post(f"{p.base_url}/chat/completions", json=body, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                errors.append(f"{p.name}: {type(exc).__name__}")
                continue
            if resp.status_code in RETRYABLE:
                errors.append(f"{p.name}: HTTP {resp.status_code}")
                continue
            if resp.status_code >= 400:
                raise LLMError(f"{p.name}: HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                errors.append(f"{p.name}: resposta sem choices")
                continue
            text = (choices[0].get("message") or {}).get("content") or ""
            usage = data.get("usage") or {}
            return Completion(
                text=text.strip(),
                provider=p.name,
                model=str(data.get("model") or p.model),
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                latency_s=time.perf_counter() - start,
                failed=tuple(errors),
            )
        raise AllProvidersFailed(errors)
