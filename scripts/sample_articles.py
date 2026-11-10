"""Pick the articles the evaluation questions are written from: random, stratified by act, seed fixed."""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from rag_normas.chunking import chunk_corpus
from rag_normas.corpus import load_corpus

QUOTA = {
    ("res-bcb-1", "regulamento"): 28, ("res-bcb-1", "ato"): 2, ("in-bcb-511", None): 8,
    ("res-bcb-507", None): 5, ("in-bcb-513", None): 5, ("in-bcb-512", None): 5, ("res-bcb-338", None): 3,
    ("res-bcb-19", None): 3, ("in-bcb-667", None): 3, ("in-bcb-199", None): 2, ("in-bcb-491", None): 2,
    ("in-bcb-32", None): 1, ("in-bcb-198", None): 1, ("in-bcb-313", None): 1, ("in-bcb-412", None): 1,
}  # fmt: skip
SKIP_WORDS = ("entra em vigor", "Fica revogad", "Ficam revogad")

articles: dict[str, list] = defaultdict(list)
for c in chunk_corpus(load_corpus(Path("data/corpus"))):
    if c.article:
        articles[c.article_key].append(c)

rng = random.Random(7)
picked = []
for (doc, part), n in QUOTA.items():
    keys = sorted(
        k
        for k, cs in articles.items()
        if k.startswith(doc + "#")
        and (part is None or k.split("#")[1] == part)
        and len(" ".join(c.text for c in cs).split()) >= 25
        and not any(w in cs[0].text[:200] for w in SKIP_WORDS)
    )
    picked += rng.sample(keys, min(n, len(keys)))
for k in picked:
    text = "\n".join(c.text for c in articles[k])
    print(
        json.dumps(
            {"key": k, "cit": articles[k][0].citation, "text": text[: int(sys.argv[1])]}, ensure_ascii=False
        )
    )
