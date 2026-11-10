"""Evaluation: retrieval metrics (deterministic, runs in CI) and answer metrics (LLM judge, run on demand).

Relevance is at article level: a question lists the article(s) that answer it, and a retrieved chunk
counts if it belongs to one of them. Retrieved chunks are deduplicated into a ranked list of articles.
"""

from __future__ import annotations

import json
import math
import platform
import statistics
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .answer import RAG, Answer
from .llm import ChatClient, LLMError
from .search import Hit, Mode, Retriever

KS = (1, 3, 5, 10)
POOL = 20


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    relevant: list[str]
    answer: str | None

    @property
    def answerable(self) -> bool:
        return bool(self.relevant)


def load_questions(path: Path) -> list[Question]:
    return [
        Question(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def ranked_articles(hits: Iterable[Hit]) -> list[str]:
    return list(dict.fromkeys(h.chunk.article_key for h in hits))


def recall_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(relevant & set(ranked[:k])) / len(relevant)


def hit_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if relevant & set(ranked[:k]) else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    for i, key in enumerate(ranked[:k], start=1):
        if key in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, key in enumerate(ranked[:k], start=1) if key in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def evaluate_retrieval(retriever: Retriever, questions: list[Question], mode: Mode) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for q in questions:
        if not q.answerable:
            continue
        t0 = time.perf_counter()
        hits = retriever.search(q.question, POOL, mode)
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = ranked_articles(hits)
        rel = set(q.relevant)
        rows.append(
            {
                "id": q.id,
                "rank": next((i for i, key in enumerate(ranked, start=1) if key in rel), None),
                **{f"recall@{k}": recall_at(ranked, rel, k) for k in KS},
                "mrr@10": reciprocal_rank(ranked, rel),
                "ndcg@10": ndcg_at(ranked, rel),
            }
        )
    metrics = {
        key: round(statistics.fmean(float(r[key]) for r in rows), 4)
        for key in [*(f"recall@{k}" for k in KS), "mrr@10", "ndcg@10"]
    }
    metrics["latency_p50_ms"] = round(_pct(latencies, 0.5), 1)
    metrics["latency_p95_ms"] = round(_pct(latencies, 0.95), 1)
    return {
        "mode": mode,
        "embedder": retriever.info.embedder,
        "questions": len(rows),
        "metrics": metrics,
        "per_question": rows,
    }


# --- answers ---------------------------------------------------------------------------------------

JUDGE_SYSTEM = (
    """Você avalia respostas de um sistema de perguntas sobre normas do Pix. Responda só com JSON."""
)

JUDGE_USER = """Trechos que o sistema recebeu:
{sources}

Pergunta: {question}
Resposta de referência (pode estar incompleta): {reference}
Resposta do sistema: {answer}

Avalie:
- "claims": divida a resposta do sistema em afirmações; para cada uma diga se ela é sustentada pelos \
trechos citados nela ("supported": true/false).
- "correct": "yes" se a resposta do sistema concorda com a referência no essencial, "partial" se \
concorda em parte, "no" se contradiz ou não responde.

Formato: {{"claims": [{{"text": "...", "supported": true}}], "correct": "yes"}}"""


def _judge(judge: ChatClient, q: Question, ans: Answer) -> dict[str, Any]:
    sources = "\n\n".join(f"[{s.n}] {s.citation}\n{s.text}" for s in ans.sources)
    msg = JUDGE_USER.format(sources=sources, question=q.question, reference=q.answer, answer=ans.text)
    out = judge.complete(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": msg}], max_tokens=1200
    )
    text = out.text.strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        data: dict[str, Any] = json.loads(text[start : end + 1])
    except ValueError:
        return {"claims": [], "correct": None, "judge_error": text[:200]}
    return data


def evaluate_answers(rag: RAG, questions: list[Question], judge: ChatClient | None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for q in questions:
        try:
            ans = rag.ask(q.question)
        except LLMError as exc:
            rows.append({"id": q.id, "answerable": q.answerable, "error": str(exc)[:200]})
            continue
        cited_keys = {s.article_key for s in ans.cited_sources}
        row: dict[str, Any] = {
            "id": q.id,
            "answerable": q.answerable,
            "abstained": ans.abstained,
            "answer": ans.text,
            "cited": [s.citation for s in ans.cited_sources],
            "invalid_citations": ans.invalid_citations,
            "provider": ans.provider,
            "model": ans.model,
            "llm_ms": round(ans.llm_ms),
        }
        if q.answerable and not ans.abstained:
            row["has_citation"] = bool(ans.cited)
            row["cites_relevant"] = bool(cited_keys & set(q.relevant))
            if judge is not None:
                try:
                    verdict = _judge(judge, q, ans)
                except LLMError as exc:
                    verdict = {"claims": [], "correct": None, "judge_error": str(exc)[:200]}
                claims = verdict.get("claims") or []
                row["claims"] = len(claims)
                row["supported"] = sum(1 for c in claims if c.get("supported") is True)
                row["correct"] = verdict.get("correct")
        rows.append(row)

    errors = [r for r in rows if "error" in r]
    done = [r for r in rows if "error" not in r]
    answerable = [r for r in done if r["answerable"]]
    unanswerable = [r for r in done if not r["answerable"]]
    answered = [r for r in answerable if not r["abstained"]]
    judged = [r for r in answered if r.get("claims")]

    def frac(items: list[dict[str, Any]], pred: Any) -> float:
        return round(sum(1 for r in items if pred(r)) / len(items), 4) if items else 0.0

    metrics = {
        "answered_rate": frac(answerable, lambda r: not r["abstained"]),
        "abstention_on_unanswerable": frac(unanswerable, lambda r: r["abstained"]),
        "citation_rate": frac(answered, lambda r: r["has_citation"]),
        "citation_accuracy": frac(answered, lambda r: r["cites_relevant"]),
        "invalid_citation_rate": frac(done, lambda r: bool(r["invalid_citations"])),
        "errors": len(errors),
    }
    if judged:
        metrics["faithfulness"] = round(statistics.fmean(r["supported"] / r["claims"] for r in judged), 4)
        metrics["correct"] = frac(judged, lambda r: r["correct"] == "yes")
        metrics["correct_or_partial"] = frac(judged, lambda r: r["correct"] in ("yes", "partial"))
    latencies = [float(r["llm_ms"]) for r in done if r["llm_ms"]]
    metrics["llm_latency_p50_ms"] = round(_pct(latencies, 0.5))
    return {"mode": rag.mode, "questions": len(rows), "metrics": metrics, "per_question": rows}


def environment() -> dict[str, str]:
    return {"python": platform.python_version(), "machine": platform.machine(), "system": platform.system()}


def compare_markdown(
    base: dict[str, Any], new: dict[str, Any], metrics: tuple[str, ...] = ("recall@5", "mrr@10", "ndcg@10")
) -> str:
    """Markdown table of retrieval metrics, new vs base, one row per (embedder, mode)."""

    def index(doc: dict[str, Any]) -> dict[tuple[str, str], dict[str, float]]:
        return {(r["embedder"], r["mode"]): r["metrics"] for r in doc["runs"]}

    b, n = index(base), index(new)
    head = "| embedder | mode | " + " | ".join(metrics) + " |"
    lines = [head, "|" + " --- |" * (len(metrics) + 2)]
    for key in n:
        cells = []
        for m in metrics:
            new_v = n[key][m]
            if key in b:
                delta = new_v - b[key][m]
                cells.append(f"{new_v:.3f} ({delta:+.3f})" if abs(delta) >= 0.0005 else f"{new_v:.3f} (=)")
            else:
                cells.append(f"{new_v:.3f} (new)")
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
