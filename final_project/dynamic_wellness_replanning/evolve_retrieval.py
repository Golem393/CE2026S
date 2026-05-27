"""evolve_retrieval — wellness memory retrieval helper.

Glue layer over final_project/retrieval.py:RetrievalCorpus that loads Nik's
memory_corpus.json once, applies wellness-shaped filters post-scoring, and
returns ranked chunks ready for the Memory & Context Manager's memory_packet.

Wayne calls `from evolve_retrieval import retrieve` from evolve_memory.py.

Design:
- One module-level RetrievalCorpus over the FULL corpus (no per-filter map):
  embeddings cached once on disk under .cache/retrieval/wellness_corpus/,
  reused across calls.
- Filters apply AFTER scoring, then trim to k. Pre-filtering small corpora
  (32 records) would routinely empty the candidate pool before ranking.
- "hybrid" / "embedding" both route to RetrievalCorpus.search(strategy=
  "embedding"), which returns hybrid_score = 0.25*lexical + 0.75*embedding
  per retrieval.py:140. "lexical" is the no-API-key fallback.
- When `runner` is None and OPENAI_API_KEY is unset, embedding strategy
  silently downgrades to lexical so local smoke tests keep moving.

Per TA Lee Joohyun's 2026-05-23 feedback: all embedding usage MUST be metered
through the official runtime.runner passed in from solve_episode. This module
no longer constructs its own LLMRunner — callers are responsible for handing
in the runtime's runner. `retrieve()` returns (results, usage) so its caller
can combine usage into the episode-level rollup.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).resolve().parent
_FINAL_PROJECT = _DATA_DIR.parent
_REPO_ROOT = _FINAL_PROJECT.parent
_CACHE_DIR = _REPO_ROOT / ".cache" / "retrieval" / "wellness_corpus"

if str(_FINAL_PROJECT) not in sys.path:
    sys.path.insert(0, str(_FINAL_PROJECT))
from retrieval import RetrievalCorpus  # noqa: E402

_corpus: RetrievalCorpus | None = None


class _StubRunner:
    """Minimal LLMRunner shim — only empty_usage() is touched in lexical mode."""

    def empty_usage(self) -> dict:
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                "embedding_tokens": 0, "total_tokens": 0,
                "estimated_cost_usd": 0.0, "by_model": {}}


def _load_corpus() -> RetrievalCorpus:
    global _corpus
    if _corpus is None:
        docs = json.loads((_DATA_DIR / "memory_corpus.json").read_text())
        _corpus = RetrievalCorpus(docs, cache_dir=_CACHE_DIR)
    return _corpus


def _resolve_runner(strategy: str, runner: Any) -> tuple[Any, str]:
    """Return (runner, effective_strategy). Auto-fallback to lexical if no runner.

    Per TA Q (2026-05-23): no module-level LLMRunner — callers must hand in the
    official runtime.runner so embedding usage flows through the metered path.
    When `runner` is None, we can only run lexical (no API key path constructed
    here).
    """
    if strategy == "lexical":
        return runner or _StubRunner(), "lexical"
    if runner is not None:
        return runner, strategy
    # No runner supplied + non-lexical requested. Surface the gap once and
    # downgrade gracefully so smoke tests / sandbox invocations don't crash.
    if not os.environ.get("OPENAI_API_KEY"):
        warnings.warn(
            f"evolve_retrieval: strategy={strategy!r} requested but no runner "
            "was supplied and OPENAI_API_KEY is unset — falling back to lexical.",
            stacklevel=3,
        )
    else:
        warnings.warn(
            f"evolve_retrieval: strategy={strategy!r} requested but no runner "
            "was supplied — falling back to lexical. Pass runtime.runner from "
            "solve_episode to enable metered embedding retrieval.",
            stacklevel=3,
        )
    return _StubRunner(), "lexical"


def _matches_filters(doc: dict, filters: dict) -> bool:
    user_id = filters.get("user_id")
    if user_id and doc.get("user_id") not in {None, user_id}:
        return False
    memory_type = filters.get("memory_type")
    if memory_type and doc.get("memory_type") != memory_type:
        return False
    signals = filters.get("signals")
    if signals:
        tags = set(doc.get("tags") or [])
        if not (set(signals) & tags):
            return False
    date_range = filters.get("date_range")
    if date_range:
        d = doc.get("date")
        if d is None or not (date_range[0] <= d <= date_range[1]):
            return False
    return True


def _lost_in_middle_reorder(ranked: list[dict]) -> list[dict]:
    """Place the highest-relevance result at the front and the second-highest
    at the back, with mid-relevance results in the middle. Mitigates the LLM
    primacy/recency bias on long contexts identified by Liu et al. 2024
    ("Lost in the Middle: How Language Models Use Long Contexts",
    arXiv:2307.03172) — models reliably attend to the start and end of an
    input window and lose information from the middle. Returning the two
    highest-relevance docs at those two positions is a deterministic, audit-
    able mitigation that costs ~0 inference and pairs cleanly with the
    hybrid 0.25*lexical + 0.75*embedding scoring upstream.

    No-op when len(ranked) <= 2 (interleaving adds nothing).
    """
    if len(ranked) <= 2:
        return ranked
    front, back, middle = [], [], []
    for i, r in enumerate(ranked):
        if i == 0:
            front.append(r)
        elif i == 1:
            back.append(r)
        else:
            middle.append(r)
    return front + middle + back


_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def retrieve(
    query: str,
    k: int = 5,
    filters: dict | None = None,
    strategy: str = "hybrid",
    runner: Any = None,
    embedding_model: str | None = None,
) -> tuple[list[dict], dict]:
    """Retrieve top-k relevant chunks from the wellness memory corpus.

    Returns: (results, usage). `results` is a list of dicts with doc_id, text,
    score, source_type, metadata. `usage` is the metered usage payload from
    the embedding calls (empty in lexical mode), shaped per LLMRunner usage
    contract so the caller can pass it straight into runtime.combine_usages.

    Filters keys: user_id, memory_type, signals (any-match against tags),
                  date_range (tuple of ISO date strings, inclusive).

    Result ordering is *not* monotonically decreasing by score. Results are
    reordered post-trim via _lost_in_middle_reorder to place the two highest-
    scoring docs at positions 0 and -1, mitigating the Liu et al. 2024
    long-context recall failure mode.

    Per TA Lee Joohyun's 2026-05-23 feedback: pass runtime.runner as `runner`
    so embedding usage is metered through the official path. When `runner` is
    None, retrieval downgrades to lexical and `usage` is empty.
    """
    corpus = _load_corpus()
    runner, effective = _resolve_runner(strategy, runner)
    underlying = "embedding" if effective in {"hybrid", "embedding"} else "lexical"

    pool_k = max(k * 4, len(corpus.docs))
    raw = corpus.search(
        runner=runner,
        query=query,
        strategy=underlying,
        top_k=pool_k,
        embedding_model=embedding_model or _DEFAULT_EMBEDDING_MODEL,
    )
    usage = raw.get("usage") or runner.empty_usage()
    candidates = raw["results"]
    if filters:
        candidates = [c for c in candidates if _matches_filters(c, filters)]
    trimmed = _lost_in_middle_reorder(candidates[:k])

    results = [
        {
            "doc_id": r["doc_id"],
            "text": r.get("text", ""),
            "score": r.get("hybrid_score", r.get("lexical_score", 0.0)),
            "source_type": r.get("memory_type", "unknown"),
            "metadata": {
                "user_id": r.get("user_id"),
                "context": r.get("context"),
                "family": r.get("family"),
                "tags": r.get("tags") or [],
                "date": r.get("date"),
                "status": r.get("status"),
                "rank": r.get("rank"),
                "lexical_score": r.get("lexical_score"),
                "embedding_score": r.get("embedding_score"),
            },
        }
        for r in trimmed
    ]
    return results, usage
