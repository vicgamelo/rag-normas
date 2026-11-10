"""One process, two doors: MCP tools (Streamable HTTP at /mcp) and a small JSON API at /api.

- ``buscar_normas``: retrieval only, no LLM. Cheap; an agent can read the passages itself.
- ``perguntar_normas_pix``: full RAG answer with citations (one LLM call on the free tier).

CPU work (embedding, re-ranking) runs in a worker thread behind a semaphore, so a burst of requests
queues instead of fighting over the container's CPU share. Meant to sit on an internal network
(behind the portal or the orchestrator), not on the public internet.
"""

from __future__ import annotations

import os
from typing import Any

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import __version__
from .answer import MAX_QUESTION_CHARS, RAG
from .llm import LLMError
from .search import Retriever

INSTRUCTIONS = """\
Busca e respostas sobre a regulamentação do Pix do Banco Central do Brasil (Regulamento do Pix, \
Pix Automático, limites, tarifas, MED/devoluções, DICT, penalidades, adesão). Toda resposta cita o \
ato e o artigo. Use buscar_normas para ler os trechos e perguntar_normas_pix para uma resposta pronta.
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False, idempotent_hint=True)
MAX_BODY = 16 * 1024


def _check(question: str) -> str:
    question = question.strip()
    if not question:
        raise ToolError("pergunta vazia")
    if len(question) > MAX_QUESTION_CHARS:
        raise ToolError(f"pergunta longa demais (máximo {MAX_QUESTION_CHARS} caracteres)")
    return question


def passages(retriever: Retriever, question: str, k: int, mode: Any) -> list[dict[str, Any]]:
    return [
        {
            "citacao": h.chunk.citation,
            "url": h.chunk.url,
            "trecho": h.chunk.text,
            "score": round(h.score, 4),
            "chunk_id": h.chunk.id,
        }
        for h in retriever.search(question, k, mode)
    ]


def build_server(rag: RAG, max_concurrency: int = 1) -> MCPServer:
    limiter = anyio.CapacityLimiter(max_concurrency)
    mcp = MCPServer(
        name="rag-normas",
        title="Normas do Pix (BCB)",
        version=__version__,
        instructions=INSTRUCTIONS,
        website_url="https://github.com/vicgamelo/rag-normas",
    )

    async def run_search(question: str, k: int) -> list[dict[str, Any]]:
        return await anyio.to_thread.run_sync(passages, rag.retriever, question, k, rag.mode, limiter=limiter)

    async def run_ask(question: str, traceparent: str | None) -> dict[str, Any]:
        try:
            ans = await anyio.to_thread.run_sync(rag.ask, question, traceparent, limiter=limiter)
        except LLMError as exc:
            raise ToolError(f"LLM indisponível: {exc}") from exc
        return ans.to_json()

    @mcp.tool(title="Buscar trechos das normas do Pix", annotations=READ_ONLY)
    async def buscar_normas(pergunta: str, k: int = 5) -> list[dict[str, Any]]:
        """Trechos das normas do Pix mais relevantes para a pergunta, com ato, artigo e link oficial.
        Não usa LLM: devolve o texto das normas para você ler e citar."""
        return await run_search(_check(pergunta), max(1, min(k, 10)))

    @mcp.tool(title="Perguntar sobre as normas do Pix", annotations=READ_ONLY)
    async def perguntar_normas_pix(pergunta: str, traceparent: str | None = None) -> dict[str, Any]:
        """Resposta curta sobre a regulamentação do Pix, com citação do ato e do artigo de cada
        afirmação. Responde "NÃO ENCONTREI" quando as normas indexadas não cobrem a pergunta.
        `traceparent` (W3C, opcional) liga os spans ao trace de quem chamou."""
        return await run_ask(_check(pergunta), traceparent)

    @mcp.custom_route("/health", methods=["GET"])  # type: ignore[untyped-decorator]
    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "version": __version__, "index": rag.retriever.info.__dict__})

    async def _body(request: Request) -> tuple[str, int] | JSONResponse:
        raw = await request.body()
        if len(raw) > MAX_BODY:
            return JSONResponse({"error": "corpo grande demais"}, status_code=413)
        try:
            data = await request.json()
            question = _check(str(data["question"]))
            k = max(1, min(int(data.get("k", 5)), 10))
        except ToolError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"error": 'esperado JSON {"question": "...", "k": 5}'}, status_code=422)
        return question, k

    @mcp.custom_route("/api/search", methods=["POST"])  # type: ignore[untyped-decorator]
    async def api_search(request: Request) -> JSONResponse:
        parsed = await _body(request)
        if isinstance(parsed, JSONResponse):
            return parsed
        return JSONResponse({"results": await run_search(*parsed)})

    @mcp.custom_route("/api/ask", methods=["POST"])  # type: ignore[untyped-decorator]
    async def api_ask(request: Request) -> JSONResponse:
        parsed = await _body(request)
        if isinstance(parsed, JSONResponse):
            return parsed
        try:
            return JSONResponse(await run_ask(parsed[0], request.headers.get("traceparent")))
        except ToolError as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)

    return mcp


def build_app(rag: RAG, host: str = "127.0.0.1") -> Any:
    allowed = [h.strip() for h in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    security = TransportSecuritySettings(enable_dns_rebinding_protection=bool(allowed), allowed_hosts=allowed)
    server = build_server(rag, int(os.getenv("RAG_MAX_CONCURRENCY", "1")))
    return server.streamable_http_app(
        stateless_http=True,
        json_response=True,
        transport_security=security,
        max_request_body_size=MAX_BODY,
        host=host,
    )


def rag_from_env() -> RAG:
    from .cli import default_embedder, index_path
    from .llm import ChatClient, providers_from_env
    from .models import Embedder, Reranker
    from .tracing import Tracer

    embedder = default_embedder()
    mode = os.getenv("RAG_MODE", "rerank")
    reranker = Reranker() if mode == "rerank" else None
    retriever = Retriever(index_path(embedder), Embedder(embedder), reranker)
    threshold = os.getenv("RAG_MIN_RERANK_SCORE", "-3.0")
    return RAG(
        retriever,
        ChatClient(providers_from_env()),
        Tracer.from_env(),
        mode=mode,  # type: ignore[arg-type]
        k=int(os.getenv("RAG_K", "5")),
        min_rerank_score=float(threshold) if threshold else None,
    )


def serve(host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(build_app(rag_from_env(), host), host=host, port=port, proxy_headers=False,
                server_header=False, log_level="info")  # fmt: skip
