"""ONNX models on CPU: the e5 embedder and the optional cross-encoder re-ranker.

Files are pinned by Hugging Face revision and SHA-256, downloaded once to ``models/`` and loaded with
onnxruntime. No torch, so the image stays small and the query path stays light.

Both models use the XLM-R vocabulary. Instead of the Hugging Face ``tokenizers`` JSON (about 370 MB of
RSS per instance for this 250k-piece vocabulary) we run the original 5 MB SentencePiece model and apply
XLM-R's fairseq id offset; tests check the ids match ``tokenizers`` on the whole corpus.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import sentencepiece as spm

Vector = npt.NDArray[np.float32]


@dataclass(frozen=True)
class RemoteFile:
    repo: str
    revision: str
    path: str
    sha256: str
    size: int
    runtime: bool = True  # False: only for comparisons and tests (`rag-normas models --all`)

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.path}"


E5_REPO, E5_REV = "intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3"
CE_REPO, CE_REV = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "1427fd652930e4ba29e8149678df786c240d8825"

FILES: dict[str, RemoteFile] = {
    "e5-small-int8": RemoteFile(
        E5_REPO,
        E5_REV,
        "onnx/model_qint8_avx512_vnni.onnx",
        "dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88",
        118346824,
    ),
    "xlmr-spm": RemoteFile(
        E5_REPO,
        E5_REV,
        "onnx/sentencepiece.bpe.model",
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
        5069051,
    ),
    "reranker-int8": RemoteFile(
        CE_REPO,
        CE_REV,
        "onnx/model_qint8_arm64.onnx",  # byte-identical to the avx512 file: portable dynamic int8
        "1825907d6c1a9001ff78124780bbde20a614a8c3df3b63409cf3c72c6fe5c8b4",
        118620017,
    ),
    "e5-small-fp32": RemoteFile(
        E5_REPO,
        E5_REV,
        "onnx/model.onnx",
        "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
        470268510,
        runtime=False,
    ),
    "e5-tokenizer": RemoteFile(
        E5_REPO,
        E5_REV,
        "onnx/tokenizer.json",
        "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
        17082730,
        runtime=False,
    ),
    "reranker-tokenizer": RemoteFile(
        CE_REPO,
        CE_REV,
        "tokenizer.json",
        "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626",
        17082660,
        runtime=False,
    ),
}

EMBEDDERS = {"e5-small-int8", "e5-small-fp32"}


def models_dir() -> Path:
    return Path(os.getenv("RAG_MODELS_DIR", "models"))


def local_path(name: str, root: Path | None = None) -> Path:
    f = FILES[name]
    return (root or models_dir()) / f.repo.replace("/", "__") / f.path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(name: str, root: Path | None = None) -> Path:
    """Download a pinned file (idempotent) and refuse it if the checksum does not match."""
    f = FILES[name]
    dest = local_path(name, root)
    if dest.exists() and dest.stat().st_size == f.size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(f.url, timeout=60) as resp, tmp.open("wb") as out:
        while block := resp.read(1 << 20):
            out.write(block)
    digest = _sha256(tmp)
    if digest != f.sha256:
        tmp.unlink()
        raise RuntimeError(f"{f.url}: sha256 {digest} != {f.sha256}")
    tmp.replace(dest)
    return dest


class XLMRTokenizer:
    """SentencePiece + fairseq offsets, the way XLM-R builds its ids.

    fairseq reserves <s>=0, <pad>=1, </s>=2, <unk>=3 and shifts every SentencePiece id by one.
    Single text: ``<s> A </s>``. Pair: ``<s> A </s></s> B </s>``, truncating the longer side first.
    """

    BOS, PAD, EOS, UNK = 0, 1, 2, 3

    def __init__(self, model_file: Path) -> None:
        self.sp = spm.SentencePieceProcessor(model_file=str(model_file))
        self._sp_unk = self.sp.unk_id()

    def pieces(self, text: str) -> list[int]:
        return [self.UNK if i == self._sp_unk else i + 1 for i in self.sp.encode(text)]

    def encode(self, text: str, max_length: int) -> list[int]:
        return [self.BOS, *self.pieces(text)[: max_length - 2], self.EOS]

    def encode_pair(self, a: str, b: str, max_length: int) -> list[int]:
        x, y = self.pieces(a), self.pieces(b)
        excess = len(x) + len(y) + 4 - max_length
        while excess > 0:  # longest_first, one token at a time
            (x if len(x) > len(y) else y).pop()
            excess -= 1
        return [self.BOS, *x, self.EOS, self.EOS, *y, self.EOS]

    def batch(self, ids: list[list[int]]) -> dict[str, npt.NDArray[np.int64]]:
        width = max(len(i) for i in ids)
        input_ids = np.full((len(ids), width), self.PAD, dtype=np.int64)
        mask = np.zeros((len(ids), width), dtype=np.int64)
        for row, seq in enumerate(ids):
            input_ids[row, : len(seq)] = seq
            mask[row, : len(seq)] = 1
        return {"input_ids": input_ids, "attention_mask": mask, "token_type_ids": np.zeros_like(input_ids)}


def _session(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # The arena keeps the peak allocation forever; without it RSS drops back between requests.
    opts.enable_cpu_mem_arena = False
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _threads() -> int:
    return int(os.getenv("RAG_ORT_THREADS", "1"))


def _feeds(session: ort.InferenceSession, batch: dict[str, Any]) -> dict[str, Any]:
    names = {i.name for i in session.get_inputs()}
    return {k: v for k, v in batch.items() if k in names}


_tokenizers: dict[Path, XLMRTokenizer] = {}


def shared_tokenizer(root: Path | None = None) -> XLMRTokenizer:
    """Embedder and re-ranker share the XLM-R vocabulary, so they share one tokenizer."""
    path = local_path("xlmr-spm", root)
    if path not in _tokenizers:
        _tokenizers[path] = XLMRTokenizer(path)
    return _tokenizers[path]


class Embedder:
    """multilingual-e5-small: "query: " / "passage: " prefixes, mean pooling, L2 normalised."""

    dim = 384

    def __init__(self, name: str = "e5-small-int8", root: Path | None = None, max_length: int = 512) -> None:
        if name not in EMBEDDERS:
            raise ValueError(f"embedder desconhecido: {name}")
        self.name = name
        self.max_length = max_length
        self.tokenizer = shared_tokenizer(root)
        self.session = _session(local_path(name, root), _threads())

    def _encode(self, texts: list[str], batch_size: int) -> Vector:
        out: list[Vector] = []
        for start in range(0, len(texts), batch_size):
            ids = [self.tokenizer.encode(t, self.max_length) for t in texts[start : start + batch_size]]
            batch = self.tokenizer.batch(ids)
            hidden = self.session.run(None, _feeds(self.session, batch))[0]  # (batch, seq, dim)
            mask = batch["attention_mask"][..., None].astype(np.float32)
            pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)
            out.append(pooled.astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def embed_passages(self, texts: list[str], batch_size: int = 16) -> Vector:
        return self._encode([f"passage: {t}" for t in texts], batch_size)

    def embed_query(self, text: str) -> Vector:
        vec: Vector = self._encode([f"query: {text}"], 1)[0]
        return vec


class Reranker:
    """mMiniLM cross-encoder trained on mMARCO (multilingual, includes Portuguese)."""

    # mMARCO passages are short: 256 tokens ranked better than 512 on our eval set, at half the cost.
    def __init__(self, root: Path | None = None, max_length: int = 256) -> None:
        self.max_length = max_length
        self.tokenizer = shared_tokenizer(root)
        self.session = _session(local_path("reranker-int8", root), _threads())

    def score(self, query: str, passages: list[str], batch_size: int | None = None) -> list[float]:
        # Small batches keep the activation peak (and the container's memory limit) low; the CPU cost is
        # the same either way.
        batch_size = batch_size or int(os.getenv("RAG_RERANK_BATCH", "1"))
        scores: list[float] = []
        for start in range(0, len(passages), batch_size):
            chunk = passages[start : start + batch_size]
            batch = self.tokenizer.batch(
                [self.tokenizer.encode_pair(query, p, self.max_length) for p in chunk]
            )
            logits = self.session.run(None, _feeds(self.session, batch))[0]
            scores.extend(float(x) for x in np.asarray(logits).reshape(len(chunk), -1)[:, 0])
        return scores
