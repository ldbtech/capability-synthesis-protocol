"""
tests/test_selection.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the capability SelectionStrategy layer — the lexical (default) and
embedding (opt-in) strategies, plus the registry shortlisting that keeps the
planner prompt bounded as the registry grows. No network/LLM calls.
"""

from __future__ import annotations

import pytest

from csp.orchestrator.capability import SynthesizedCapability
from csp.orchestrator.registry import CapabilityRegistry
from csp.orchestrator.selection import (
    TagLexicalStrategy,
    EmbeddingStrategy,
    capability_text,
)


def _synth(name: str, description: str = "", tags=None) -> SynthesizedCapability:
    spec = {"params": {"tags": tags or []}}
    return SynthesizedCapability(name=name, spec=spec, description=description)


def _caps():
    return [
        _synth("aggregate_table", "group rows and compute averages, sums, counts",
               tags=["aggregation", "group-by", "statistics"]),
        _synth("plot_chart", "draw a bar, line, scatter or histogram figure",
               tags=["plot", "chart", "matplotlib", "visualization"]),
        _synth("send_email", "send an email message to a recipient",
               tags=["email", "notify"]),
        _synth("detect_anomalies", "flag outlier rows in a numeric series",
               tags=["anomaly", "outlier", "statistics"]),
        _synth("translate_text", "translate a string between languages",
               tags=["translation", "language"]),
    ]


# ── Lexical strategy ──────────────────────────────────────────────────────────

def test_lexical_ranks_relevant_capability_first():
    strat = TagLexicalStrategy()
    out = strat.shortlist("average salary grouped by department", _caps(), k=2)
    assert out[0].name == "aggregate_table"
    assert len(out) == 2


def test_lexical_matches_on_tags_not_just_name():
    # The goal shares no words with the NAME "plot_chart", only with its tags
    # and description ("histogram"/"chart").
    strat = TagLexicalStrategy()
    out = strat.shortlist("draw a histogram", _caps(), k=1)
    assert out[0].name == "plot_chart"


def test_lexical_returns_all_when_fewer_than_k():
    caps = _caps()
    out = TagLexicalStrategy().shortlist("anything", caps, k=99)
    assert len(out) == len(caps)


def test_capability_text_includes_name_desc_tags():
    cap = _synth("x", "does a thing", tags=["alpha", "beta"])
    text = capability_text(cap)
    assert "x" in text and "does a thing" in text and "alpha" in text and "beta" in text


# ── Embedding strategy ────────────────────────────────────────────────────────

def _fake_embed(texts):
    # Deterministic toy embedding: a 2-d vector counting two themes so cosine
    # similarity is meaningful without a real model.
    vecs = []
    for t in texts:
        tl = t.lower()
        stats = sum(w in tl for w in ("group", "average", "aggregation", "salary", "department"))
        viz = sum(w in tl for w in ("plot", "chart", "histogram", "draw", "figure"))
        vecs.append([float(stats), float(viz)])
    return vecs


def test_embedding_ranks_by_cosine():
    strat = EmbeddingStrategy(_fake_embed)
    out = strat.shortlist("average salary by department", _caps(), k=1)
    assert out[0].name == "aggregate_table"


def test_embedding_caches_capability_vectors():
    calls = {"n": 0}

    def counting_embed(texts):
        calls["n"] += 1
        return _fake_embed(texts)

    strat = EmbeddingStrategy(counting_embed)
    caps = _caps()
    strat.shortlist("draw a chart", caps, k=2)
    after_first = calls["n"]
    strat.shortlist("plot a figure", caps, k=2)
    # Second call should only embed the new goal, not re-embed the capabilities.
    assert calls["n"] == after_first + 1


def test_embedding_rejects_non_callable():
    with pytest.raises(TypeError):
        EmbeddingStrategy(embed_fn=None)  # type: ignore[arg-type]


# ── Registry integration: shortlist only past the threshold ───────────────────

@pytest.mark.asyncio
async def test_registry_shows_all_below_threshold():
    reg = CapabilityRegistry(shortlist_threshold=10)
    for c in _caps():
        reg._synthesized[c.name] = c
    summary = await reg.summary_for_planner("draw a histogram")
    # 5 caps < threshold 10 → no shortlisting, every capability present.
    assert "showing" not in summary
    for c in _caps():
        assert c.name in summary


@pytest.mark.asyncio
async def test_registry_shortlists_above_threshold():
    reg = CapabilityRegistry(shortlist_threshold=3, shortlist_k=2)
    for c in _caps():
        reg._synthesized[c.name] = c
    summary = await reg.summary_for_planner("average salary by department")
    assert "showing 2 of 5" in summary
    assert "aggregate_table" in summary
    # An unrelated capability should have been filtered out of the prompt.
    assert "translate_text" not in summary


@pytest.mark.asyncio
async def test_registry_no_goal_shows_all():
    reg = CapabilityRegistry(shortlist_threshold=3, shortlist_k=2)
    for c in _caps():
        reg._synthesized[c.name] = c
    summary = await reg.summary_for_planner(None)
    # Without a goal there's nothing to rank against → show everything.
    assert "showing" not in summary
    assert "translate_text" in summary
