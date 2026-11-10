"""rag-normas command line.

rag-normas fetch                 # download the acts from the BCB API into data/corpus (cleaned JSON)
rag-normas models                # download the pinned ONNX models into models/
rag-normas build                 # chunk + embed + write the SQLite index
rag-normas search "pergunta"     # retrieval only
rag-normas ask "pergunta"        # answer with citations (needs an LLM provider)
rag-normas eval retrieval        # recall@k / MRR / nDCG for every mode -> eval/results/retrieval.json
rag-normas eval answers          # answer metrics with an LLM judge -> eval/results/answers-<provider>.json
rag-normas serve                 # MCP (Streamable HTTP at /mcp) + JSON API at /api
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .search import MODES

DATA = Path(os.getenv("RAG_DATA_DIR", "data"))
EVAL = Path(os.getenv("RAG_EVAL_DIR", "eval"))


def index_path(embedder: str) -> Path:
    return Path(os.getenv("RAG_INDEX", str(DATA / f"index-{embedder}.sqlite")))


def default_embedder() -> str:
    return os.getenv("RAG_EMBEDDER", "e5-small-int8")


def _retriever(embedder: str, rerank: bool = True):  # type: ignore[no-untyped-def]
    from .models import Embedder, Reranker
    from .search import Retriever

    return Retriever(index_path(embedder), Embedder(embedder), Reranker() if rerank else None)


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"-> {path}", file=sys.stderr)


def cmd_fetch(_: argparse.Namespace) -> None:
    from .corpus import fetch_corpus

    for doc in fetch_corpus(DATA / "corpus", DATA / "raw"):
        print(
            f"{doc.id:12} v{doc.versao:<3} {doc.data} {len(doc.text.split()):>6} palavras  {doc.assunto[:70]}"
        )


def cmd_models(args: argparse.Namespace) -> None:
    from .models import FILES, download

    names = [n for n, f in FILES.items() if args.all or f.runtime]
    for name in names:
        print(name, download(name))


def cmd_build(args: argparse.Namespace) -> None:
    from .index import build_index, info_json
    from .models import Embedder

    os.environ.setdefault("RAG_ORT_THREADS", str(os.cpu_count() or 1))
    for name in args.embedder:
        info = build_index(DATA / "corpus", index_path(name), Embedder(name))
        print(info_json(info))


def cmd_search(args: argparse.Namespace) -> None:
    retriever = _retriever(args.embedder, rerank=args.mode == "rerank")
    for i, hit in enumerate(retriever.search(args.query, args.k, args.mode), start=1):
        print(f"{i}. [{hit.score:.3f}] {hit.chunk.citation}  {hit.ranks}")
        print("   " + hit.chunk.text[:240].replace("\n", " ") + "\n")


def _rag(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    from .answer import RAG
    from .llm import ChatClient, providers_from_env
    from .tracing import Tracer

    threshold = os.getenv("RAG_MIN_RERANK_SCORE", "-3.0")
    return RAG(
        _retriever(
            args.embedder if isinstance(args.embedder, str) else args.embedder[0],
            rerank=args.mode == "rerank",
        ),
        ChatClient(providers_from_env()),
        Tracer.from_env(),
        mode=args.mode,
        k=args.k,
        min_rerank_score=float(threshold) if threshold else None,
    )


def cmd_ask(args: argparse.Namespace) -> None:
    rag = _rag(args)
    answer = rag.ask(args.question)
    rag.tracer.flush()
    print(json.dumps(answer.to_json(), ensure_ascii=False, indent=2))


def cmd_eval(args: argparse.Namespace) -> None:
    from .evaluation import environment, evaluate_answers, evaluate_retrieval, load_questions

    if args.what == "compare":
        from .evaluation import compare_markdown

        base, new = (json.loads(Path(f).read_text(encoding="utf-8")) for f in (args.base, args.new))
        print(compare_markdown(base, new))
        return

    questions = load_questions(EVAL / "questions.jsonl")
    if args.what == "retrieval":
        runs = []
        for embedder in args.embedder:
            retriever = _retriever(embedder, rerank=True)
            for mode in args.modes:
                run = evaluate_retrieval(retriever, questions, mode)
                runs.append(run)
                print(f"{embedder:14} {mode:7} " + "  ".join(f"{k}={v}" for k, v in run["metrics"].items()))
        meta = {"version": __version__, "index": retriever.info.__dict__, "environment": environment()}
        _write(Path(args.out or EVAL / "results" / "retrieval.json"), {**meta, "runs": runs})
        return

    from .llm import ChatClient, providers_from_env

    rag = _rag(args)
    judge = None
    if args.judge:
        env = {**os.environ, "RAG_LLM_PROVIDERS": args.judge}
        if args.judge_model:
            env.update(RAG_OLLAMA_MODEL=args.judge_model, RAG_GROQ_MODEL=args.judge_model)
        judge = ChatClient(providers_from_env(env))
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q.id in wanted]
    run = evaluate_answers(rag, questions, judge)
    run["judge"] = judge.providers[0].model if judge else None
    print(json.dumps(run["metrics"], indent=2))
    provider = rag.llm.providers[0]
    out = Path(args.out or EVAL / "results" / f"answers-{provider.name}.json")
    _write(out, {"version": __version__, "llm": provider.model, "environment": environment(), **run})


def cmd_serve(args: argparse.Namespace) -> None:
    from .server import serve

    serve(args.host, args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rag-normas")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch").set_defaults(fn=cmd_fetch)
    p = sub.add_parser("models")
    p.add_argument(
        "--all",
        action="store_true",
        help="also the fp32 embedder (470 MB) and the HF tokenizers, for comparisons and tests",
    )
    p.set_defaults(fn=cmd_models)
    p = sub.add_parser("build")
    p.add_argument("--embedder", nargs="+", default=[default_embedder()])
    p.set_defaults(fn=cmd_build)

    for name, fn in (("search", cmd_search), ("ask", cmd_ask)):
        p = sub.add_parser(name)
        p.add_argument("query" if name == "search" else "question")
        p.add_argument("-k", type=int, default=5)
        p.add_argument("--mode", choices=MODES, default=os.getenv("RAG_MODE", "rerank"))
        p.add_argument("--embedder", default=default_embedder())
        p.set_defaults(fn=fn)

    p = sub.add_parser("eval")
    p.add_argument("what", choices=["retrieval", "answers", "compare"])
    p.add_argument("--base", help="compare: the older retrieval.json")
    p.add_argument("--new", help="compare: the newer retrieval.json")
    p.add_argument("--embedder", nargs="+", default=[default_embedder()])
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--mode", choices=MODES, default=os.getenv("RAG_MODE", "rerank"), help="answers only")
    p.add_argument("-k", type=int, default=5, help="answers only: passages sent to the LLM")
    p.add_argument("--judge", help="provider chain for the judge, e.g. ollama")
    p.add_argument("--judge-model", help="judge model (use a different family from the answering model)")
    p.add_argument("--ids", help="answers only: comma separated question ids")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("serve")
    p.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    p.set_defaults(fn=cmd_serve)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
