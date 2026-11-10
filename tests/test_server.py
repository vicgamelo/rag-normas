from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from starlette.testclient import TestClient

from rag_normas.server import build_app

from .test_answer_llm import make_rag

MCP_HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


def app_client(index_file: Path, reply: str = "Das 20h às 6h [1].") -> tuple[TestClient, list[Any]]:
    rag, seen = make_rag(index_file, reply)
    return TestClient(build_app(rag)), seen


def rpc(client: TestClient, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    resp = client.post("/mcp", headers=MCP_HEADERS, content=json.dumps(body))
    assert resp.status_code == 200, resp.text
    data: dict[str, Any] = resp.json()
    return data


def test_health(index_file: Path) -> None:
    client, _ = app_client(index_file)
    with client:
        body = client.get("/health").json()
    assert body["status"] == "ok" and body["index"]["embedder"] == "fake"


def test_api_search_and_ask(index_file: Path) -> None:
    client, seen = app_client(index_file)
    with client:
        found = client.post("/api/search", json={"question": "período noturno", "k": 2}).json()["results"]
        assert len(found) == 2
        assert found[0]["citacao"] == "Res. BCB 1/2020 (Regulamento), art. 2"
        assert seen == []  # search never calls the LLM
        ans = client.post("/api/ask", json={"question": "período noturno"}).json()
    assert ans["citations"][0]["citation"] == "Res. BCB 1/2020 (Regulamento), art. 2"
    assert len(seen) == 1


def test_api_validates_input(index_file: Path) -> None:
    client, seen = app_client(index_file)
    with client:
        assert client.post("/api/ask", json={}).status_code == 422
        assert client.post("/api/ask", json={"question": "   "}).status_code == 422
        assert client.post("/api/ask", json={"question": "x" * 501}).status_code == 422
        assert client.post("/api/ask", content=b"not json").status_code == 422
        assert client.post("/api/ask", content=b"{" + b" " * 20000 + b"}").status_code == 413
    assert seen == []


def test_api_ask_llm_down_is_503(index_file: Path) -> None:
    rag, _ = make_rag(index_file, "x")
    rag.llm._http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with TestClient(build_app(rag)) as client:
        resp = client.post("/api/ask", json={"question": "período noturno"})
    assert resp.status_code == 503 and "LLM indisponível" in resp.json()["error"]


def test_mcp_tools(index_file: Path) -> None:
    client, _ = app_client(index_file)
    with client:
        tools = {t["name"]: t for t in rpc(client, "tools/list")["result"]["tools"]}
        assert set(tools) == {"buscar_normas", "perguntar_normas_pix"}
        assert tools["buscar_normas"]["annotations"]["readOnlyHint"] is True
        out = rpc(
            client, "tools/call", {"name": "buscar_normas", "arguments": {"pergunta": "fraude", "k": 1}}
        )
        result = out["result"]
        assert not result.get("isError")
        payload = result["structuredContent"]["result"]
        assert payload[0]["citacao"] == "Res. BCB 1/2020 (Regulamento), art. 5"
        err = rpc(client, "tools/call", {"name": "perguntar_normas_pix", "arguments": {"pergunta": ""}})
        assert err["result"]["isError"] is True
