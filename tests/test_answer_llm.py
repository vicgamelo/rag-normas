from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from genai_lens import Lens

from rag_normas.answer import NOT_FOUND, RAG, build_prompt, is_abstention, parse_citations
from rag_normas.llm import AllProvidersFailed, ChatClient, LLMError, Provider, providers_from_env
from rag_normas.search import Retriever
from rag_normas.tracing import Tracer

from .conftest import FakeEmbedder


def openai_reply(text: str, model: str = "m", tokens: tuple[int, int] = (100, 20)) -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1]},
    }


def client_for(handler: Any, providers: list[Provider] | None = None) -> ChatClient:
    providers = providers or [Provider("groq", "https://groq.test/v1", "gpt-oss-20b", "k1")]
    return ChatClient(providers, client=httpx.Client(transport=httpx.MockTransport(handler)))


TWO = [
    Provider("groq", "https://groq.test/v1", "gpt-oss-20b", "k1"),
    Provider("openrouter", "https://or.test/v1", "gemma:free", "k2"),
]


# --- citations ----------------------------------------------------------------------------------


def test_parse_citations_keeps_order_and_flags_out_of_range() -> None:
    valid, invalid = parse_citations("A [2]. B [1][2]. C [7].", n_sources=5)
    assert valid == [2, 1]
    assert invalid == [7]  # e.g. the model confused "§ 7º" with source [7]


@pytest.mark.parametrize("text", ["NÃO ENCONTREI", "não encontrei.", "  ", "NÃO ENCONTREI nas normas"])
def test_abstention(text: str) -> None:
    assert is_abstention(text)


def test_answer_is_not_abstention() -> None:
    assert not is_abstention("O limite é R$ 1.000 [1].")


# --- LLM fallback ---------------------------------------------------------------------------------


def test_falls_over_on_429_and_reports_it() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "groq.test":
            return httpx.Response(429, json={"error": "rate"})
        assert req.headers["authorization"] == "Bearer k2"
        return httpx.Response(200, json=openai_reply("ok", "gemma:free"))

    out = client_for(handler, TWO).complete([{"role": "user", "content": "x"}])
    assert (out.provider, out.text, out.failed) == ("openrouter", "ok", ("groq: HTTP 429",))


def test_falls_over_on_timeout() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "groq.test":
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json=openai_reply("ok"))

    assert client_for(handler, TWO).complete([]).provider == "openrouter"


def test_does_not_fall_over_on_400() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request"})

    with pytest.raises(LLMError, match="HTTP 400"):
        client_for(handler, TWO).complete([])


def test_all_failed() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(AllProvidersFailed) as exc:
        client_for(handler, TWO).complete([])
    assert exc.value.errors == ["groq: HTTP 503", "openrouter: HTTP 503"]


def test_providers_from_env_skips_missing_keys_and_requires_free_models() -> None:
    assert providers_from_env({}) == []
    chain = providers_from_env({"GROQ_API_KEY": "g", "OPENROUTER_API_KEY": "o"})
    assert [p.name for p in chain] == ["groq", "openrouter"]
    assert chain[0].model == "openai/gpt-oss-20b" and chain[1].model.endswith(":free")
    with pytest.raises(ValueError, match=":free"):
        providers_from_env({"OPENROUTER_API_KEY": "o", "RAG_OPENROUTER_MODEL": "openai/gpt-5"})
    ollama = providers_from_env({"RAG_LLM_PROVIDERS": "ollama"})[0]
    assert ollama.extra == {"reasoning_effort": "none"} and ollama.api_key is None


def test_no_provider_fails_only_when_asked() -> None:
    client = ChatClient([])  # the server must still start and serve search
    with pytest.raises(LLMError, match="nenhum provedor"):
        client.complete([])


# --- RAG pipeline + traces ---------------------------------------------------------------------


class Spans:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def __call__(self, batch: list[dict[str, Any]]) -> None:
        self.items.extend(batch)


def make_rag(index_file: Path, reply: str, spans: Spans | None = None, **kw: Any) -> tuple[RAG, list[Any]]:
    seen: list[Any] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=openai_reply(reply, "openai/gpt-oss-20b"))

    lens = (
        Lens(endpoint="http://lens.test", project="t", transport=spans, flush_interval=60) if spans else None
    )
    retriever = Retriever(index_file, FakeEmbedder())  # type: ignore[arg-type]
    return RAG(retriever, client_for(handler), Tracer(lens), mode="hybrid", k=3, **kw), seen


def test_ask_maps_citations_to_sources(index_file: Path) -> None:
    rag, seen = make_rag(index_file, "O período noturno vai das 20h às 6h [1]. Limite de R$ 1.000 [1][9].")
    ans = rag.ask("Qual o período noturno do Pix?")
    assert not ans.abstained
    assert ans.cited == [1] and ans.invalid_citations == [9]
    assert ans.cited_sources[0].citation == "Res. BCB 1/2020 (Regulamento), art. 2"
    body = ans.to_json()
    assert body["citations"][0]["url"] == "https://example/res1"
    assert body["usage"] == {"input_tokens": 100, "output_tokens": 20}
    # the prompt numbers the passages and carries the rules
    user = seen[0]["messages"][1]["content"]
    assert user.startswith("Trechos:\n\n[1] Res. BCB 1/2020 (Regulamento), art. 2")
    assert NOT_FOUND in seen[0]["messages"][0]["content"]


def test_ask_abstention_from_llm(index_file: Path) -> None:
    rag, _ = make_rag(index_file, "NÃO ENCONTREI")
    ans = rag.ask("Qual é a taxa Selic?")
    assert ans.abstained and ans.cited == []


def test_rerank_gate_skips_the_llm(index_file: Path) -> None:
    class Low:
        def score(self, query: str, passages: list[str]) -> list[float]:
            return [-5.0] * len(passages)

    rag, seen = make_rag(index_file, "não deveria ser chamado", min_rerank_score=-1.0)
    rag.retriever.reranker = Low()  # type: ignore[assignment]
    rag.mode = "rerank"
    ans = rag.ask("Qual é a taxa Selic?")
    assert ans.abstained and ans.text == NOT_FOUND
    assert seen == []


def test_ask_emits_a_trace_cascade_under_the_callers_trace(index_file: Path) -> None:
    spans = Spans()
    rag, _ = make_rag(index_file, "Vai das 20h às 6h [1].", spans)
    parent_trace, parent_span = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"
    ans = rag.ask("período noturno", traceparent=f"00-{parent_trace}-{parent_span}-01")
    rag.tracer.flush()
    by_name = {s["name"]: s for s in spans.items}
    assert set(by_name) == {"rag.answer", "rag.retrieve", "llm.chat"}
    root = by_name["rag.answer"]
    assert root["trace_id"] == parent_trace == ans.trace_id
    assert root["parent_id"] == parent_span
    assert by_name["rag.retrieve"]["parent_id"] == root["span_id"]
    assert by_name["llm.chat"]["parent_id"] == root["span_id"]
    assert by_name["llm.chat"]["model"] == "openai/gpt-oss-20b"
    assert by_name["llm.chat"]["input_tokens"] == 100
    assert by_name["rag.retrieve"]["kind"] == "retrieval"


def test_bad_traceparent_starts_a_new_trace(index_file: Path) -> None:
    spans = Spans()
    rag, _ = make_rag(index_file, "x [1]", spans)
    ans = rag.ask("período noturno", traceparent="garbage")
    rag.tracer.flush()
    assert ans.trace_id and len(ans.trace_id) == 32
    assert next(s for s in spans.items if s["name"] == "rag.answer")["parent_id"] is None


def test_prompt_marks_passages_as_data() -> None:
    assert "não instruções" in build_prompt("q", [])[0]["content"]
