"""Portuguese text normalisation for BM25: lowercase, stopwords out, Snowball stem, accents off."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import snowballstemmer

_stemmer = snowballstemmer.stemmer("portuguese")
TOKEN = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)

STOPWORDS = frozenset(
    {
        "a", "à", "ao", "aos", "as", "às", "até", "com", "como", "da", "das", "de", "dela", "dele", "deles",
        "demais", "do", "dos", "e", "é", "ela", "elas", "ele", "eles", "em", "entre", "era", "essa",
        "essas", "esse", "esses", "esta", "está", "estão", "estas", "este", "estes", "eu", "foi", "for",
        "há", "isso", "isto", "já", "la", "lhe", "mais", "mas", "me", "mesmo", "meu", "minha", "muito",
        "na", "nas", "nem", "no", "nos", "nós", "num", "numa", "o", "os", "ou", "para", "pela", "pelas",
        "pelo", "pelos", "por", "qual", "quais", "quando", "que", "quem", "se", "sem", "seja", "ser", "seu",
        "seus", "sua", "suas", "são", "só", "também", "te", "tem", "têm", "tu", "um", "uma", "umas", "uns",
        "você", "vocês", "vos"
    }
)  # fmt: skip


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


@lru_cache(maxsize=65536)
def _stem(word: str) -> str:
    return strip_accents(_stemmer.stemWord(word))


def analyze(text: str) -> list[str]:
    tokens = []
    for raw in TOKEN.findall(text.lower()):
        if raw in STOPWORDS:
            continue
        if raw[0].isdigit():
            tokens.append(raw.split(",")[0].replace(".", ""))  # "15.000,00" and "15000" -> "15000"
        else:
            tokens.extend(_stem(part) for part in raw.split("-") if part not in STOPWORDS)
    return [t for t in tokens if t]
