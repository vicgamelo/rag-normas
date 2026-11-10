"""Download and clean the corpus: Pix regulations from the Banco Central do Brasil.

The texts come from the public normativos API used by bcb.gov.br, which serves the *consolidated*
version of each act (amendments already applied). Acts are official texts (Lei 9.610/98, art. 8º, IV)
and the BCB allows reproduction with attribution, so the cleaned corpus is committed to the repo and
the index can be rebuilt offline.
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

API = "https://www.bcb.gov.br/api/conteudo/app/normativos/exibenormativo"
PAGE = "https://www.bcb.gov.br/estabilidadefinanceira/exibenormativo"
USER_AGENT = "rag-normas/0.1 (+https://github.com/vicgamelo/rag-normas)"

# (tipo, número, rótulo curto usado nas citações)
SOURCES: list[tuple[str, int, str]] = [
    ("Resolução BCB", 1, "Res. BCB 1/2020"),
    ("Resolução BCB", 19, "Res. BCB 19/2020"),
    ("Resolução BCB", 338, "Res. BCB 338/2023"),
    ("Resolução BCB", 507, "Res. BCB 507/2025"),
    ("Instrução Normativa BCB", 32, "IN BCB 32/2020"),
    ("Instrução Normativa BCB", 198, "IN BCB 198/2021"),
    ("Instrução Normativa BCB", 199, "IN BCB 199/2021"),
    ("Instrução Normativa BCB", 313, "IN BCB 313/2022"),
    ("Instrução Normativa BCB", 412, "IN BCB 412/2023"),
    ("Instrução Normativa BCB", 491, "IN BCB 491/2024"),
    ("Instrução Normativa BCB", 511, "IN BCB 511/2024"),
    ("Instrução Normativa BCB", 512, "IN BCB 512/2024"),
    ("Instrução Normativa BCB", 513, "IN BCB 513/2024"),
    ("Instrução Normativa BCB", 667, "IN BCB 667/2025"),
]


@dataclass(frozen=True)
class Document:
    id: str  # "res-bcb-1"
    label: str  # "Res. BCB 1/2020"
    tipo: str
    numero: int
    data: str  # ISO date of the act
    versao: int  # consolidated version number at the BCB
    assunto: str
    url: str
    text: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def doc_id(tipo: str, numero: int) -> str:
    prefix = {"Resolução BCB": "res-bcb", "Instrução Normativa BCB": "in-bcb"}[tipo]
    return f"{prefix}-{numero}"


# "(Redação dada pela Resolução BCB nº 429, de 11/11/2024.)", "(Incluído, a partir de ..., pela ...)"
AMENDMENT_NOTE = re.compile(
    r"\s*\((?:Redação dada|Incluíd[oa]s?|Revogad[oa]s?|Renumerad[oa]s?|Alterad[oa]s?|Vide|Vigência)"
    r"[^()]*(?:\([^()]*\)[^()]*)*\)",
    re.IGNORECASE,
)


class _TextExtractor(HTMLParser):
    BLOCK = frozenset({"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"})
    # <s> is how the BCB marks superseded wording in consolidated texts: drop it, keep only text in force.
    SKIP = frozenset({"style", "script", "title", "s", "strike", "del"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")
        elif tag == "td":
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            # Source HTML wraps lines inside paragraphs; only block tags are real line breaks.
            self.parts.append(re.sub(r"\s+", " ", data))


def html_to_text(raw: str) -> str:
    parser = _TextExtractor()
    parser.feed(raw)
    text = html.unescape("".join(parser.parts))
    text = text.replace("​", "").replace("\xa0", " ")
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip(" |") for line in text.split("\n")]
    out: list[str] = []
    for line in lines:
        line = line.strip()
        if line:
            out.append(line)
    text = "\n".join(out)
    text = AMENDMENT_NOTE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", re.sub(r"[ \t]{2,}", " ", text))
    # "Art. 3 º" / "Art. 1 o" -> "Art. 3º"
    text = re.sub(r"\bArt\. (\d+) ([º°o])\b", r"Art. \1º", text)
    return text


def parse_api_payload(payload: dict[str, Any], tipo: str, numero: int, label: str) -> Document:
    items = payload.get("conteudo") or []
    if not items:
        raise ValueError(f"{tipo} {numero}: resposta sem conteúdo")
    item = items[0]
    if item.get("Revogado") or item.get("Cancelado"):
        raise ValueError(f"{tipo} {numero} está revogada ou cancelada")
    query = urllib.parse.urlencode({"tipo": tipo, "numero": numero})
    return Document(
        id=doc_id(tipo, numero),
        label=label,
        tipo=tipo,
        numero=numero,
        data=str(item["Data"])[:10],
        versao=int(float(item.get("VersaoNormativo") or 1)),
        assunto=re.sub(r"\s+", " ", str(item.get("Assunto") or "")).strip(),
        url=f"{PAGE}?{query}",
        text=html_to_text(str(item.get("Texto") or "")),
    )


def fetch_payload(tipo: str, numero: int, timeout: float = 30.0) -> dict[str, Any]:
    url = f"{API}?{urllib.parse.urlencode({'p1': tipo, 'p2': numero})}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload: dict[str, Any] = json.loads(resp.read())
    return payload


def fetch_corpus(out_dir: Path, raw_dir: Path | None = None, pause_s: float = 0.5) -> list[Document]:
    """Download every act in SOURCES and write the cleaned JSON to ``out_dir``.

    ``raw_dir`` caches the API payloads, so re-cleaning does not hit the BCB again.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = []
    for tipo, numero, label in SOURCES:
        raw_file = raw_dir / f"{doc_id(tipo, numero)}.json" if raw_dir else None
        if raw_file and raw_file.exists():
            payload = json.loads(raw_file.read_text(encoding="utf-8"))
        else:
            payload = fetch_payload(tipo, numero)
            if raw_file:
                raw_file.parent.mkdir(parents=True, exist_ok=True)
                raw_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            time.sleep(pause_s)
        doc = parse_api_payload(payload, tipo, numero, label)
        (out_dir / f"{doc.id}.json").write_text(
            json.dumps(doc.to_json(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        docs.append(doc)
    return docs


def load_corpus(corpus_dir: Path) -> list[Document]:
    docs = [Document(**json.loads(p.read_text(encoding="utf-8"))) for p in sorted(corpus_dir.glob("*.json"))]
    if not docs:
        raise FileNotFoundError(f"nenhum documento em {corpus_dir}; rode `rag-normas fetch`")
    return docs
