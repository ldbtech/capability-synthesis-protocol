"""
backend/rag_store.py
~~~~~~~~~~~~~~~~~~~~~
In-memory RAG store over an uploaded CSV.

Each row becomes a text chunk ("col: value | col: value | ..."), embedded with
BGE. Retrieval is cosine similarity (dot product on normalized vectors). Small
and dependency-light — fine for the demo's CSV sizes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from embeddings import embed, embed_one

log = logging.getLogger("app.rag_store")


@dataclass
class RagStore:
    """Holds one CSV's rows + their embeddings for retrieval."""
    filename: str                       = ""
    columns:  list[str]                 = field(default_factory=list)
    rows:     list[dict[str, Any]]      = field(default_factory=list)
    _chunks:  list[str]                 = field(default_factory=list)
    _vecs:    Optional[np.ndarray]      = None

    def index_csv(self, df: pd.DataFrame, filename: str) -> None:
        """Embed every row of the dataframe and store it for retrieval."""
        self.filename = filename
        self.columns  = [str(c) for c in df.columns]
        self.rows     = df.to_dict(orient="records")
        self._chunks  = [_row_to_text(r) for r in self.rows]
        self._vecs    = embed(self._chunks) if self._chunks else None
        log.info("indexed %d rows from %s", len(self.rows), filename)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k most similar rows to the query."""
        if self._vecs is None or len(self._chunks) == 0:
            return []
        q = embed_one(query)
        scores = self._vecs @ q                       # cosine sim (normalized)
        top = np.argsort(-scores)[:k]
        return [
            {"row": self.rows[i], "text": self._chunks[i], "score": float(scores[i])}
            for i in top
        ]

    @property
    def ready(self) -> bool:
        return self._vecs is not None and len(self._chunks) > 0

    def summary(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "columns":  self.columns,
            "row_count": len(self.rows),
            "ready":    self.ready,
        }


def _row_to_text(row: dict[str, Any]) -> str:
    return " | ".join(f"{k}: {v}" for k, v in row.items())
