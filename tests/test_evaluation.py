from __future__ import annotations

import json
from collections import Counter

import pytest

from rag_normas.chunking import chunk_corpus
from rag_normas.corpus import SOURCES, doc_id, load_corpus
from rag_normas.evaluation import (
    compare_markdown,
    evaluate_retrieval,
    hit_at,
    load_questions,
    ndcg_at,
    recall_at,
    reciprocal_rank,
)

from .conftest import ROOT, models_available

QUESTIONS = ROOT / "eval" / "questions.jsonl"
BASELINE = ROOT / "eval" / "baseline.json"


def test_metrics_on_a_hand_case() -> None:
    ranked = ["a", "b", "c", "d"]
    rel = {"c", "z"}
    assert recall_at(ranked, rel, 2) == 0.0
    assert recall_at(ranked, rel, 3) == 0.5
    assert hit_at(ranked, rel, 3) == 1.0
    assert reciprocal_rank(ranked, rel) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, rel, k=2) == 0.0
    # one hit at rank 3 out of an ideal of two hits at ranks 1-2
    assert ndcg_at(ranked, rel) == pytest.approx((1 / 2) / (1 + 1 / 1.5849625))


def test_committed_corpus_matches_sources() -> None:
    docs = load_corpus(ROOT / "data" / "corpus")
    assert sorted(d.id for d in docs) == sorted(doc_id(t, n) for t, n, _ in SOURCES)
    for d in docs:
        assert d.url.startswith("https://www.bcb.gov.br/")
        assert "Redação dada" not in d.text  # superseded text and notes stay out


def test_questions_point_at_real_articles() -> None:
    questions = load_questions(QUESTIONS)
    keys = {c.article_key for c in chunk_corpus(load_corpus(ROOT / "data" / "corpus"))}
    assert len({q.id for q in questions}) == len(questions)
    assert len({q.question for q in questions}) == len(questions)
    for q in questions:
        assert set(q.relevant) <= keys, q.id
        assert (q.answer is None) == (not q.relevant), q.id
    counts = Counter(q.answerable for q in questions)
    assert counts[True] >= 60 and counts[False] >= 10


def test_review_sample_is_part_of_the_set() -> None:
    ids = {q.id for q in load_questions(QUESTIONS)}
    sample = [line.split("|")[1].strip() for line in (ROOT / "eval" / "REVIEW.md").read_text().splitlines()
              if line.startswith("| q")]  # fmt: skip
    assert len(sample) == 15 and set(sample) <= ids


@pytest.mark.models
@pytest.mark.skipif(not models_available(), reason="run `rag-normas models` first")
def test_retrieval_does_not_regress_against_baseline() -> None:
    """The CI gate: the default configuration must stay within 0.03 of the committed baseline."""
    from rag_normas.models import Embedder, Reranker
    from rag_normas.search import Retriever

    index = ROOT / "data" / "index-e5-small-int8.sqlite"
    if not index.exists():
        pytest.skip("run `rag-normas build` first")
    baseline = json.loads(BASELINE.read_text())
    retriever = Retriever(index, Embedder("e5-small-int8", ROOT / "models"), Reranker(ROOT / "models"))
    questions = load_questions(QUESTIONS)
    for mode, expected in baseline["modes"].items():
        got = evaluate_retrieval(retriever, questions, mode)["metrics"]
        for metric in ("recall@5", "mrr@10"):
            assert got[metric] >= expected[metric] - 0.03, (mode, metric, got[metric], expected[metric])


def test_compare_markdown_shows_deltas() -> None:
    run = {"embedder": "e", "mode": "hybrid", "metrics": {"recall@5": 0.8, "mrr@10": 0.7, "ndcg@10": 0.75}}
    better = {**run, "metrics": {"recall@5": 0.85, "mrr@10": 0.7, "ndcg@10": 0.7}}
    extra = {**run, "mode": "rerank"}
    table = compare_markdown({"runs": [run]}, {"runs": [better, extra]})
    assert "| e | hybrid | 0.850 (+0.050) | 0.700 (=) | 0.700 (-0.050) |" in table
    assert "| e | rerank | 0.800 (new)" in table
