"""The index: one SQLite file with the chunks, an FTS5 table (BM25) and a sqlite-vec table.

It is built offline (``rag-normas build``) and opened read-only by the server, so the query path
needs no database service and the container can run with a read-only filesystem.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sqlite_vec

from . import __version__
from .chunking import Chunk, chunk_corpus
from .corpus import load_corpus
from .models import Embedder
from .text import analyze

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE chunks (
    rowid INTEGER PRIMARY KEY,
    id TEXT UNIQUE NOT NULL,
    doc_id TEXT NOT NULL,
    label TEXT NOT NULL,
    part TEXT NOT NULL,
    article TEXT,
    heading TEXT NOT NULL,
    text TEXT NOT NULL,
    url TEXT NOT NULL
);
CREATE VIRTUAL TABLE fts USING fts5(terms, tokenize = 'unicode61 remove_diacritics 2');
"""


@dataclass(frozen=True)
class IndexInfo:
    embedder: str
    chunks: int
    corpus_sha256: str
    built_at: str
    version: str


def connect(path: Path, read_only: bool = True) -> sqlite3.Connection:
    # immutable=1: no locks, no journal files, so it works on a read-only filesystem
    uri = f"file:{path}?mode=ro&immutable=1" if read_only else f"file:{path}"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def corpus_hash(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.id.encode())
        h.update(c.text.encode())
    return h.hexdigest()


def build_index(corpus_dir: Path, out: Path, embedder: Embedder) -> IndexInfo:
    chunks = chunk_corpus(load_corpus(corpus_dir))
    vectors = embedder.embed_passages([c.passage() for c in chunks])
    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    conn = connect(tmp, read_only=False)
    with conn:
        conn.executescript(SCHEMA)
        conn.execute(
            f"CREATE VIRTUAL TABLE vec USING vec0(embedding float[{embedder.dim}] distance_metric=cosine)"
        )
        for rowid, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True), start=1):
            conn.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (rowid, chunk.id, chunk.doc_id, chunk.label, chunk.part, chunk.article, chunk.heading,
                 chunk.text, chunk.url),
            )  # fmt: skip
            conn.execute(
                "INSERT INTO fts(rowid, terms) VALUES (?, ?)", (rowid, " ".join(analyze(chunk.passage())))
            )
            conn.execute(
                "INSERT INTO vec(rowid, embedding) VALUES (?, ?)",
                (rowid, vector.astype(np.float32).tobytes()),
            )
        info = IndexInfo(
            embedder=embedder.name,
            chunks=len(chunks),
            corpus_sha256=corpus_hash(chunks),
            built_at=datetime.now(UTC).isoformat(timespec="seconds"),
            version=__version__,
        )
        conn.executemany("INSERT INTO meta VALUES (?, ?)", [(k, str(v)) for k, v in info.__dict__.items()])
        conn.execute("INSERT INTO fts(fts) VALUES ('optimize')")
    conn.execute("VACUUM")
    conn.close()
    tmp.replace(out)
    return info


def read_info(conn: sqlite3.Connection) -> IndexInfo:
    meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    meta["chunks"] = int(meta["chunks"])
    return IndexInfo(**meta)


def info_json(info: IndexInfo) -> str:
    return json.dumps(info.__dict__, indent=2)
