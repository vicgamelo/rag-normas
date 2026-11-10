from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag_normas.index import connect, read_info
from rag_normas.search import Retriever, clean_query, fts_query, rrf

from .conftest import FakeEmbedder


def retriever(index_file: Path) -> Retriever:
    return Retriever(index_file, FakeEmbedder())  # type: ignore[arg-type]


def test_index_metadata(index_file: Path) -> None:
    info = read_info(connect(index_file))
    assert info.embedder == "fake"
    assert info.chunks == 8  # act: title + 2 articles; regulation: heading + 4 articles (art. 3 revoked)
    assert len(info.corpus_sha256) == 64


def test_index_is_opened_read_only(index_file: Path) -> None:
    conn = connect(index_file, read_only=True)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM chunks")


def test_bm25_matches_stemmed_terms(index_file: Path) -> None:
    hits = retriever(index_file).search("limites noturnos de pessoa física", 3, "bm25")
    assert hits[0].chunk.article_key == "res-bcb-1#regulamento#art-2"
    assert hits[0].ranks == {"bm25": 1}


def test_vector_and_hybrid_return_ranks_from_each_retriever(index_file: Path) -> None:
    r = retriever(index_file)
    assert r.search("bloqueio cautelar suspeita de fraude", 1, "vector")[0].chunk.article == "5"
    hit = r.search("bloqueio cautelar suspeita de fraude", 1, "hybrid")[0]
    assert hit.chunk.article == "5"
    assert set(hit.ranks) == {"bm25", "vector"}


def test_empty_or_stopword_query_returns_nothing_from_bm25(index_file: Path) -> None:
    r = retriever(index_file)
    assert r.search("   ", 5, "hybrid") == []
    assert fts_query("de que o a") is None
    assert r.bm25("de que o a") == []


def test_fts_query_quotes_terms() -> None:
    # quoting keeps FTS5 syntax out of user input: no column filters, NEAR, or parse errors
    assert fts_query('limite" OR x:y NEAR(') == '"limit" OR "or" OR "x" OR "y" OR "near"'


def test_rerank_without_reranker_fails_loudly(index_file: Path) -> None:
    with pytest.raises(ValueError, match="rerank"):
        retriever(index_file).search("limite", 3, "rerank")


def test_rerank_reorders_pool(index_file: Path) -> None:
    class ByLength:
        def score(self, query: str, passages: list[str]) -> list[float]:
            return [float(len(p)) for p in passages]

    r = Retriever(index_file, FakeEmbedder(), ByLength())  # type: ignore[arg-type]
    hits = r.search("pix", 3, "rerank")
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all("hybrid" in h.ranks for h in hits)


def test_index_built_with_other_embedder_is_refused(index_file: Path) -> None:
    other = FakeEmbedder()
    other.name = "e5-small-int8"
    with pytest.raises(ValueError, match="índice feito com fake"):
        Retriever(index_file, other)  # type: ignore[arg-type]


def test_rrf_sums_reciprocal_ranks() -> None:
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    assert [rowid for rowid, _ in fused] == [1, 3, 2]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_clean_query_drops_invisible_characters() -> None:
    assert clean_query("  limite\u200b  noturno\ufeff \n") == "limite noturno"
