"""The SentencePiece tokenizer must give exactly the ids of the Hugging Face tokenizers it replaces."""

from __future__ import annotations

import json

import pytest

from rag_normas.chunking import chunk_corpus
from rag_normas.corpus import load_corpus
from rag_normas.models import XLMRTokenizer, local_path
from rag_normas.search import clean_query

from .conftest import ROOT

MODELS = ROOT / "models"
pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(
        not all(local_path(n, MODELS).exists() for n in ("xlmr-spm", "e5-tokenizer", "reranker-tokenizer")),
        reason="run `rag-normas models --all` first",
    ),
]


def texts() -> list[str]:
    out = [f"passage: {c.passage()}" for c in chunk_corpus(load_corpus(ROOT / "data" / "corpus"))]
    out += [f"query: {json.loads(line)['question']}" for line in (ROOT / "eval" / "questions.jsonl").open()]
    # queries go through clean_query first: e5's tokenizer.json alone emits "▁" for a trailing U+200B
    edge = ["", "  espaços   duplos\te tab", "R$ 1.000,00 — “aspas” ½ ﬁ", "emoji 🚀", "fim\u200b"]
    return [*out, *(clean_query(t) for t in edge)]


@pytest.mark.parametrize("hf_name", ["e5-tokenizer", "reranker-tokenizer"])
def test_single_texts_match_hf(hf_name: str) -> None:
    from tokenizers import Tokenizer

    hf = Tokenizer.from_file(str(local_path(hf_name, MODELS)))
    hf.enable_truncation(max_length=512)
    ours = XLMRTokenizer(local_path("xlmr-spm", MODELS))
    mismatches = [t[:60] for t in texts() if hf.encode(t).ids != ours.encode(t, 512)]
    assert mismatches == []


def test_pairs_with_truncation_match_hf() -> None:
    from tokenizers import Tokenizer

    hf = Tokenizer.from_file(str(local_path("reranker-tokenizer", MODELS)))
    hf.enable_truncation(max_length=256)
    ours = XLMRTokenizer(local_path("xlmr-spm", MODELS))
    passages = texts()[:200]
    for q in ("qual o limite do pix à noite?", "prazo " * 300):
        for p in passages:
            assert hf.encode(q, p).ids == ours.encode_pair(q, p, 256), (q[:30], p[:60])
