"""Structure-aware chunking of Brazilian legal acts.

A chunk is one article (caput, paragraphs, items), because that is what people cite. Long articles are
split at line boundaries into windows that repeat the article's caput, so every piece still says what
it is about. Annex text without articles (tables, forms) is chunked by windows under the annex name.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .corpus import Document

MAX_WORDS = 280
CAPUT_CONTEXT_WORDS = 40

ARTICLE = re.compile(r"^Art\.\s*(\d+)\s*[º°o]?(?:\s*-\s*([A-Z]{1,2}))?\s*[.\-–]?(?:\s+(.*))?$")
CHAPTER = re.compile(r"^CAP[IÍ]TULO\s+([IVXLC]+(?:-[A-Z])?)\b", re.IGNORECASE)
SECTION = re.compile(r"^(Sub)?[Ss]e[çc][ãa]o\s+([IVXLC]+(?:-[A-Z])?)\b")
ANNEX = re.compile(r"^(?:REGULAMENTO\s+)?ANEXO\b(?:\s+([IVX]+)\b)?", re.IGNORECASE)
STRUCTURE_NOTE = re.compile(r"^\((?:Cap[ií]tulo|Se[çc][ãa]o|Subse[çc][ãa]o|T[ií]tulo)[^)]*\)$", re.IGNORECASE)
EMPTY_ITEM = re.compile(r"^(?:[IVXLC]+\s*[-–]|[a-z]\)|§\s*\d+\s*[º°]?|Parágrafo único\.?)\s*$")
STOP = frozenset({"NOTA"})


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    label: str
    part: str  # "" for the act itself, "Regulamento", "Anexo I"...
    article: str | None  # "39-C"
    heading: str  # chapter/section titles, for context
    text: str
    url: str

    @property
    def article_key(self) -> str:
        """What the evaluation set refers to: one article of one part of one act."""
        art = f"art-{self.article}" if self.article else "texto"
        return f"{self.doc_id}#{_slug(self.part)}#{art}"

    @property
    def citation(self) -> str:
        part = f" ({self.part})" if self.part else ""
        art = f", art. {self.article}" if self.article else ""
        return f"{self.label}{part}{art}"

    def passage(self) -> str:
        """Text that gets embedded and indexed: citation and headings give the model context."""
        head = f"{self.citation}. {self.heading}".strip(" .")
        return f"{head}\n{self.text}"


def _slug(part: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", part.lower()).strip("-") or "ato"


def article_key(doc_id: str, part: str, article: str | None) -> str:
    art = f"art-{article}" if article else "texto"
    return f"{doc_id}#{_slug(part)}#{art}"


def _annex_name(line: str) -> str | None:
    m = ANNEX.match(line)
    if not m:
        return None
    if line.upper().startswith("REGULAMENTO"):
        return "Regulamento"
    return f"Anexo {m.group(1).upper()}" if m.group(1) else "Anexo"


def _windows(lines: list[str], max_words: int) -> Iterable[list[str]]:
    buf: list[str] = []
    words = 0
    for line in lines:
        n = len(line.split())
        if buf and words + n > max_words:
            yield buf
            buf, words = [], 0
        buf.append(line)
        words += n
    if buf:
        yield buf


@dataclass
class _Block:
    part: str
    article: str | None
    heading: str
    lines: list[str]


def _blocks(doc: Document) -> list[_Block]:
    blocks: list[_Block] = []
    part = ""
    chapter = section = ""
    pending_title: str | None = None  # the line after "CAPÍTULO II" is its title
    current = _Block(part, None, "", [])

    def flush() -> None:
        nonlocal current
        body = [ln for ln in current.lines if not EMPTY_ITEM.match(ln)]
        if current.article is not None:
            # "Art. 120." alone means the article was revoked.
            first = ARTICLE.match(body[0]) if body else None
            if (first is not None and first.group(3)) or len(body) > 1:
                blocks.append(_Block(current.part, current.article, current.heading, body))
        elif body:
            blocks.append(_Block(current.part, None, current.heading, body))
        current = _Block(part, None, _heading(chapter, section), [])

    def _heading(ch: str, sec: str) -> str:
        return " · ".join(x for x in (ch, sec) if x)

    for raw in doc.text.split("\n"):
        line = raw.strip()
        if not line or STRUCTURE_NOTE.match(line):
            continue
        if line.strip(" :").upper() in STOP:
            break
        if pending_title is not None:
            if not ARTICLE.match(line) and not SECTION.match(line) and not CHAPTER.match(line):
                if pending_title == "chapter":
                    chapter = f"{chapter} – {line}"
                else:
                    section = f"{section} – {line}"
                pending_title = None
                current.heading = _heading(chapter, section)
                continue
            pending_title = None
        annex = _annex_name(line)
        if annex and len(line.split()) <= 16:
            flush()
            part = annex
            chapter = section = ""
            current = _Block(part, None, "", [line])
            continue
        if m := CHAPTER.match(line):
            flush()
            chapter, section, pending_title = f"Capítulo {m.group(1).upper()}", "", "chapter"
            current.heading = chapter
            continue
        if m := SECTION.match(line):
            flush()
            section = f"{'Subseção' if m.group(1) else 'Seção'} {m.group(2)}"
            pending_title = "section"
            current.heading = _heading(chapter, section)
            continue
        if m := ARTICLE.match(line):
            flush()
            current.article = m.group(1) + (f"-{m.group(2)}" if m.group(2) else "")
        current.lines.append(line)
    flush()
    return blocks


def chunk_document(doc: Document, max_words: int = MAX_WORDS) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for block in _blocks(doc):
        caput = " ".join(block.lines[0].split()[:CAPUT_CONTEXT_WORDS])
        windows = list(_windows(block.lines, max_words))
        for i, window in enumerate(windows):
            if i > 0 and block.article:
                window = [f"{caput} [...]", *window]
            base = article_key(doc.id, block.part, block.article)
            n = seen.get(base, 0)
            seen[base] = n + 1
            chunks.append(
                Chunk(
                    id=f"{base}#{n}",
                    doc_id=doc.id,
                    label=doc.label,
                    part=block.part,
                    article=block.article,
                    heading=block.heading,
                    text="\n".join(window),
                    url=doc.url,
                )
            )
    return chunks


def chunk_corpus(docs: Iterable[Document], max_words: int = MAX_WORDS) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, max_words)]
