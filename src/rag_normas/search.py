"""Retrieval: BM25 (FTS5), vector (sqlite-vec), hybrid with reciprocal rank fusion, optional re-rank."""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .chunking import Chunk
from .index import IndexInfo, connect, read_info
from .models import Embedder, Reranker
from .text import analyze

Mode = Literal["bm25", "vector", "hybrid", "rerank"]
MODES: tuple[Mode, ...] = ("bm25", "vector", "hybrid", "rerank")

RRF_K = 60
CANDIDATES = 50
RERANK_POOL = int(os.getenv("RAG_RERANK_POOL", "10"))


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # rank in each retriever, 1-based


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: sum of 1/(k + rank) over the lists an item appears in."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, rowid in enumerate(ranking, start=1):
            scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


INVISIBLE = re.compile(r"[\u200b-\u200d\u2060\ufeff]")


def clean_query(text: str) -> str:
    """Zero-width characters out, whitespace collapsed (the corpus went through the same cleaning)."""
    return " ".join(INVISIBLE.sub("", text).split())


def fts_query(text: str) -> str | None:
    terms = dict.fromkeys(analyze(text))  # dedupe, keep order
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms)


class Retriever:
    def __init__(self, index_path: Path, embedder: Embedder, reranker: Reranker | None = None) -> None:
        self.conn: sqlite3.Connection = connect(index_path, read_only=True)
        self.info: IndexInfo = read_info(self.conn)
        if self.info.embedder != embedder.name:
            raise ValueError(f"índice feito com {self.info.embedder}, mas o embedder é {embedder.name}")
        self.embedder = embedder
        self.reranker = reranker
        self._lock = threading.Lock()  # one sqlite connection, shared by the server threads

    def _chunks(self, rowids: list[int]) -> dict[int, Chunk]:
        if not rowids:
            return {}
        marks = ",".join("?" * len(rowids))
        with self._lock:
            rows = self.conn.execute(
                "SELECT rowid, id, doc_id, label, part, article, heading, text, url FROM chunks"
                f" WHERE rowid IN ({marks})",
                rowids,
            ).fetchall()
        return {r[0]: Chunk(*r[1:]) for r in rows}

    def bm25(self, query: str, limit: int = CANDIDATES) -> list[int]:
        q = fts_query(query)
        if q is None:
            return []
        with self._lock:
            rows = self.conn.execute(
                "SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?", (q, limit)
            ).fetchall()
        return [r[0] for r in rows]

    def vector(self, query: str, limit: int = CANDIDATES) -> list[int]:
        vec = self.embedder.embed_query(query).astype(np.float32).tobytes()
        with self._lock:
            rows = self.conn.execute(
                "SELECT rowid FROM vec WHERE embedding MATCH ? AND k = ? ORDER BY distance", (vec, limit)
            ).fetchall()
        return [r[0] for r in rows]

    def search(self, query: str, k: int = 5, mode: Mode = "hybrid") -> list[Hit]:
        query = clean_query(query)
        if not query:
            return []
        ranks: dict[str, list[int]] = {}
        if mode in ("bm25", "hybrid", "rerank"):
            ranks["bm25"] = self.bm25(query)
        if mode in ("vector", "hybrid", "rerank"):
            ranks["vector"] = self.vector(query)
        if mode in ("bm25", "vector"):
            ranked = [(rowid, 1.0 / (RRF_K + i)) for i, rowid in enumerate(ranks[mode], start=1)]
        else:
            ranked = rrf(list(ranks.values()))
        pool = ranked[: RERANK_POOL if mode == "rerank" else k]
        chunks = self._chunks([rowid for rowid, _ in pool])
        hits = [
            Hit(
                chunks[rowid],
                score,
                {name: lst.index(rowid) + 1 for name, lst in ranks.items() if rowid in lst},
            )
            for rowid, score in pool
            if rowid in chunks
        ]
        if mode == "rerank":
            if self.reranker is None:
                raise ValueError("modo rerank sem re-ranker carregado")
            scores = self.reranker.score(query, [h.chunk.passage() for h in hits])
            for pos, (hit, s) in enumerate(zip(hits, scores, strict=True), start=1):
                hit.ranks["hybrid"] = pos
                hit.score = s
            hits.sort(key=lambda h: -h.score)
        return hits[:k]
