from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rag_normas.corpus import Document
from rag_normas.index import build_index

ROOT = Path(__file__).resolve().parents[1]

REG = """RESOLUÇÃO BCB Nº 1, DE 12 DE AGOSTO DE 2020
Institui o arranjo de pagamentos Pix e aprova o seu Regulamento.
Art. 1º Fica instituído o arranjo de pagamentos Pix.
Art. 2º Fica aprovado o Regulamento anexo.
REGULAMENTO ANEXO À RESOLUÇÃO BCB Nº 1, DE 12 DE AGOSTO DE 2020
CAPÍTULO I
DAS DEFINIÇÕES
Art. 1º Este Regulamento disciplina o funcionamento do Pix.
CAPÍTULO II
DOS LIMITES
Seção I
Do período noturno
Art. 2º O período noturno compreende, em geral, o período entre as 20 horas e as 6 horas.
Parágrafo único. O limite noturno para pessoa física é de R$1.000,00 (mil reais).
Art. 3º
Art. 4º As tarifas de transferência podem ser cobradas apenas dos usuários pagadores pessoa jurídica.
Seção X
(Seção X revogada pela Resolução BCB nº 506, de 26/9/2025.)
Art. 5º O bloqueio cautelar de recursos ocorre quando houver suspeita de fraude.
NOTA
O Decreto nº 10.411 prevê a análise de impacto regulatório."""


def make_doc(text: str = REG, id: str = "res-bcb-1", label: str = "Res. BCB 1/2020") -> Document:
    return Document(id, label, "Resolução BCB", 1, "2020-08-12", 43, "Pix", "https://example/res1", text)


class FakeEmbedder:
    """Bag of hashed words: deterministic, no model files, enough to exercise the vector path."""

    name = "fake"
    dim = 384

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "res-bcb-1.json").write_text(json.dumps(make_doc().to_json(), ensure_ascii=False), encoding="utf-8")
    return d


@pytest.fixture
def index_file(tmp_path: Path, corpus_dir: Path) -> Path:
    out = tmp_path / "index.sqlite"
    build_index(corpus_dir, out, FakeEmbedder())  # type: ignore[arg-type]
    return out


def models_available() -> bool:
    from rag_normas.models import FILES, local_path

    return all(local_path(n, ROOT / "models").exists() for n, f in FILES.items() if f.runtime)
