"""
csp.orchestrator.selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Capability *selection* — how CSP picks which existing capabilities are worth
showing the planner for a given goal.

Why this exists
---------------
The planner used to receive EVERY capability in the registry on every goal.
That is fine for a handful of capabilities, but it has two costs that grow with
the registry:

  1. token cost  — N capabilities = N lines of prompt, every single call.
  2. attention   — past a few hundred entries the model can't reliably scan
                   them, so both selection AND the reuse-vs-synthesize decision
                   degrade (the classic "tool bloat" / context-rot problem).

A SelectionStrategy fixes this by shortlisting the top-k capabilities relevant
to the goal, so the planner reasons over a small candidate set instead of the
whole registry. Selection cost becomes ~O(k) instead of O(N).

Two strategies ship, chosen by the developer at ``Orchestrator(...)`` time:

  * TagLexicalStrategy (DEFAULT) — pure Python. No dependencies, no model, no
    vector store, no per-request latency. The semantic work is front-loaded to
    *synthesis time* (the synthesizer tags each capability — free, the LLM call
    already happens) and query time is a cheap BM25 / tag match. This keeps the
    library frictionless for a startup: install csp and ship.

  * EmbeddingStrategy (OPT-IN) — semantic vector retrieval for higher recall at
    large scale. The developer supplies an ``embed_fn`` (their own model or a
    provider's embedding endpoint); CSP stays dependency-pure. Pays for an
    embedding on write and an ANN-ish cosine scan on read.

Both implement the same tiny contract — ``shortlist(goal, caps, k)`` — so they
are fully interchangeable and the rest of the pipeline never changes.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Callable, Sequence

from .capability import AnyCapability

# Tiny English stopword set — dropped before lexical scoring so common words
# ("the", "of", "by") don't dominate the match. Kept deliberately small.
_STOPWORDS = frozenset(
    "a an and are as at be by for from in into is it of on or over the to with "
    "per each into via".split()
)

# Split on non-alphanumerics AND on camelCase / snake boundaries so that a
# capability named "plotChart" or "plot_chart" tokenizes to {plot, chart}.
_SPLIT = re.compile(r"[^0-9a-zA-Z]+|(?<=[a-z0-9])(?=[A-Z])")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords removed. Pure, dependency-free."""
    tokens = (t.lower() for t in _SPLIT.split(text) if t)
    return [t for t in tokens if t and t not in _STOPWORDS]


def capability_text(cap: AnyCapability) -> str:
    """
    The searchable text for a capability: its name, description and tags.
    Used by every strategy so they all index the same surface.
    """
    tags = getattr(cap, "tags", None) or []
    return f"{cap.name} {getattr(cap, 'description', '') or ''} {' '.join(tags)}"


class SelectionStrategy(ABC):
    """
    Decide which capabilities the planner should see for a goal.

    Implementations must be cheap to call repeatedly and must NEVER mutate the
    capabilities they're given. ``shortlist`` returns at most ``k`` capabilities,
    most relevant first; when there are fewer than ``k`` it returns them all.
    """

    @abstractmethod
    def shortlist(
        self,
        goal: str,
        caps: Sequence[AnyCapability],
        k: int,
    ) -> list[AnyCapability]:
        ...


class TagLexicalStrategy(SelectionStrategy):
    """
    Default strategy — pure-Python lexical (BM25) ranking over each capability's
    name + description + tags. Zero dependencies, zero infra, no model call.

    BM25 is a well-worn information-retrieval ranking: it rewards goal terms that
    appear in a capability's text, down-weights terms that appear in *every*
    capability (low information), and saturates term frequency so one repeated
    word can't dominate. It is fast (microseconds for thousands of short docs)
    and works well here because capability descriptions are short and tend to
    share vocabulary with the goal — especially once the synthesizer tags them.

    Recall ceiling: a goal worded with entirely different vocabulary than the
    capability can miss (no synonym understanding). Synthesis-time tags and the
    creation-time dedup sweep are what close most of that gap; swap in
    EmbeddingStrategy if you need true semantic recall.
    """

    # Standard BM25 knobs. k1 controls term-frequency saturation; b controls how
    # much document length normalizes the score. These defaults are the usual
    # general-purpose values and rarely need tuning for short capability docs.
    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b

    def shortlist(
        self,
        goal: str,
        caps: Sequence[AnyCapability],
        k: int,
    ) -> list[AnyCapability]:
        if len(caps) <= k:
            return list(caps)

        # Tokenize every capability once for this call. The registry is small
        # enough that recomputing per call is cheaper than maintaining an index;
        # if that ever changes, cache by capability id.
        docs = [_tokenize(capability_text(c)) for c in caps]
        doc_lens = [len(d) for d in docs]
        avg_len = (sum(doc_lens) / len(doc_lens)) or 1.0
        n_docs = len(docs)

        # Document frequency: how many capabilities contain each term.
        df: Counter[str] = Counter()
        for d in docs:
            for term in set(d):
                df[term] += 1

        query_terms = set(_tokenize(goal))

        def score(doc: list[str], doc_len: int) -> float:
            if not doc:
                return 0.0
            tf = Counter(doc)
            s = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                # idf: rarer terms across the registry carry more signal.
                idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
                freq = tf[term]
                norm = freq * (self._k1 + 1) / (
                    freq + self._k1 * (1 - self._b + self._b * doc_len / avg_len)
                )
                s += idf * norm
            return s

        ranked = sorted(
            range(n_docs),
            key=lambda i: score(docs[i], doc_lens[i]),
            reverse=True,
        )
        return [caps[i] for i in ranked[:k]]


# An embedding function maps a batch of texts to a batch of float vectors. The
# developer supplies one (local model, provider endpoint, anything) so the core
# library never depends on a specific embedding stack.
EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


class EmbeddingStrategy(SelectionStrategy):
    """
    Opt-in strategy — semantic vector retrieval for high recall at scale.

    You provide ``embed_fn`` (e.g. a BGE model, OpenAI/Voyage embeddings, etc.).
    CSP embeds each capability's text once and caches the vector by capability
    id, then ranks candidates by cosine similarity to the embedded goal. Cosine
    is computed in pure Python so there's still no hard numpy dependency, though
    a vectorized ``embed_fn`` is recommended for large registries.

    Trade-off vs. TagLexical: higher recall (understands synonyms / paraphrase)
    at the cost of an embedding call per new capability and a similarity scan per
    goal. Use it when you've outgrown lexical matching, not before.
    """

    def __init__(self, embed_fn: EmbedFn, *, cache: bool = True) -> None:
        if not callable(embed_fn):
            raise TypeError("EmbeddingStrategy requires a callable embed_fn")
        self._embed = embed_fn
        self._cache_enabled = cache
        # capability id -> embedding vector. Cheap to keep; capabilities are
        # immutable once synthesized, so a cached vector never goes stale.
        self._vec_cache: dict[str, Sequence[float]] = {}

    def shortlist(
        self,
        goal: str,
        caps: Sequence[AnyCapability],
        k: int,
    ) -> list[AnyCapability]:
        if len(caps) <= k:
            return list(caps)

        # Embed any capabilities we haven't seen yet (one batched call), then
        # serve the rest from cache.
        missing = [c for c in caps if self._cache_key(c) not in self._vec_cache]
        if missing or not self._cache_enabled:
            to_embed = missing if self._cache_enabled else list(caps)
            vectors = self._embed([capability_text(c) for c in to_embed])
            for c, v in zip(to_embed, vectors):
                self._vec_cache[self._cache_key(c)] = v

        goal_vec = self._embed([goal])[0]
        scored = sorted(
            caps,
            key=lambda c: _cosine(goal_vec, self._vec_cache[self._cache_key(c)]),
            reverse=True,
        )
        return list(scored[:k])

    def _cache_key(self, cap: AnyCapability) -> str:
        # Synthesized capabilities have a stable uuid; registered ones use their
        # name. Either way the key is stable for the capability's lifetime.
        return getattr(cap, "id", None) or f"reg:{cap.name}"


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors. Returns 0.0 for a zero vector."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
