from __future__ import annotations

import pytest

from rag_normas.chunking import chunk_document
from rag_normas.corpus import html_to_text, parse_api_payload
from rag_normas.text import analyze

from .conftest import make_doc

CONSOLIDATED_HTML = """
<p>§ 3º Além das instituições mencionadas no <b>caput</b>, fica facultada a adesão ao Pix&#58;</p>
<p><s><span style="color:red">I - das demais instituições financeiras que ofereçam contas; </span></s></p>
<p><s><span>I - das demais instituições financeiras;
e </span></s><a href="x"><s><span>(Redação dada pela
Resolução BCB nº 403, de 22/7/2024.)</span></s></a></p>
<p><span>I - das
demais instituições autorizadas pelo Banco Central do Brasil; e </span><a href="y"><span>(Redação dada pela
Resolução BCB nº 429, de 11/11/2024.)</span></a></p>
<p>Art.
120. (Revogado pela Resolução BCB nº 506, de 26/9/2025.)</p>
<p>Art. 3 º O limite é de R$&#160;1.000,00.</p>
"""


def test_html_keeps_only_text_in_force() -> None:
    text = html_to_text(CONSOLIDATED_HTML)
    lines = text.split("\n")
    # superseded wording (<s>) is gone, the current one stays
    assert "que ofereçam contas" not in text
    assert "I - das demais instituições financeiras;" not in text
    assert "I - das demais instituições autorizadas pelo Banco Central do Brasil; e" in lines
    # amendment notes are stripped, source line wraps are joined
    assert "Redação dada" not in text and "Revogado pela" not in text
    assert "Art. 120." in lines
    assert "Art. 3º O limite é de R$ 1.000,00." in lines


def test_parse_payload_refuses_revoked_acts() -> None:
    payload = {"conteudo": [{"Data": "2020-01-01T00:00:00Z", "Texto": "<p>Art. 1º X.</p>", "Revogado": True}]}
    with pytest.raises(ValueError, match="revogada"):
        parse_api_payload(payload, "Resolução BCB", 9, "Res. BCB 9")


def test_parse_payload_builds_document() -> None:
    payload = {
        "conteudo": [
            {
                "Data": "2024-08-30T12:00:00Z",
                "VersaoNormativo": 5.0,
                "Assunto": " Dispõe sobre   limites ",
                "Texto": "<p>Art. 1º Texto.</p>",
                "Revogado": False,
                "Cancelado": False,
            }
        ]
    }
    doc = parse_api_payload(payload, "Instrução Normativa BCB", 512, "IN BCB 512/2024")
    assert (doc.id, doc.versao, doc.data, doc.assunto) == (
        "in-bcb-512",
        5,
        "2024-08-30",
        "Dispõe sobre limites",
    )
    assert "numero=512" in doc.url and doc.text == "Art. 1º Texto."


def test_chunks_follow_articles_parts_and_headings() -> None:
    chunks = chunk_document(make_doc())
    keys = [c.article_key for c in chunks]
    # the act and its annexed regulation both have an "art. 1": different parts, different keys
    assert "res-bcb-1#ato#art-1" in keys and "res-bcb-1#regulamento#art-1" in keys
    art2 = next(c for c in chunks if c.article_key == "res-bcb-1#regulamento#art-2")
    assert "Parágrafo único" in art2.text
    assert art2.heading == "Capítulo II – DOS LIMITES · Seção I – Do período noturno"
    assert art2.citation == "Res. BCB 1/2020 (Regulamento), art. 2"
    assert art2.passage().startswith("Res. BCB 1/2020 (Regulamento), art. 2. Capítulo II")


def test_revoked_article_is_dropped_and_not_merged() -> None:
    chunks = chunk_document(make_doc())
    keys = [c.article_key for c in chunks]
    assert "res-bcb-1#regulamento#art-3" not in keys
    art2 = next(c for c in chunks if c.article_key == "res-bcb-1#regulamento#art-2")
    assert "Art. 3º" not in art2.text  # an empty "Art. 3º" line must not stick to art. 2


def test_structure_notes_and_nota_tail_are_not_indexed() -> None:
    text = "\n".join(c.text for c in chunk_document(make_doc()))
    assert "Seção X revogada" not in text
    assert "impacto regulatório" not in text


def test_long_article_is_split_with_its_caput() -> None:
    body = "\n".join(
        f"{n} - item de número {n} com bastante texto para encher a janela." for n in range(1, 60)
    )
    doc = make_doc("Art. 7º O participante deve cumprir o seguinte:\n" + body)
    chunks = chunk_document(doc, max_words=120)
    assert len(chunks) > 1
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.article == "7" for c in chunks)
    assert all(c.text.startswith("Art. 7º O participante") for c in chunks)
    assert all(len(c.text.split()) <= 120 + 40 for c in chunks)


def test_analyzer_stems_and_normalises_numbers() -> None:
    assert analyze("limites noturnos") == analyze("limite noturno")
    assert analyze("Período") == analyze("periodo")
    assert analyze("15.000,00") == analyze("15000") == ["15000"]
    assert "de" not in analyze("limite de valor")
