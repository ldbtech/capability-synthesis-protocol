"""
backend/embeddings.py
~~~~~~~~~~~~~~~~~~~~~~
Local BGE embeddings via sentence-transformers. No API key, runs offline.

The model (BAAI/bge-small-en-v1.5, 384-dim) is downloaded once on first use
and cached by huggingface under ~/.cache. Embeddings are L2-normalized so a
dot product equals cosine similarity.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

log = logging.getLogger("app.embeddings")

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM   = 384


@lru_cache(maxsize=1)
def _model():
    """Load the BGE model once (lazy — first call pays the download/load cost)."""
    from sentence_transformers import SentenceTransformer
    log.info("loading embedding model %s ...", _MODEL_NAME)
    m = SentenceTransformer(_MODEL_NAME)
    log.info("embedding model ready")
    return m


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts → (n, 384) float32 array, L2-normalized."""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    vecs = _model().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32)


def embed_one(text: str) -> np.ndarray:
    """Embed a single text → (384,) float32 vector."""
    return embed([text])[0]
