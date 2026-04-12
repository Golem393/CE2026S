from __future__ import annotations

"""Student-owned helper module.

This file is intentionally lightweight. Teams can add their own wrappers around the
primitive tool API, such as semantic rerankers, bundle search helpers, fallback
search, or verifier helpers.
"""

from typing import Any, Dict, List


def rerank_hotels(candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Placeholder for student custom hotel reranking."""
    return candidates


def rerank_restaurants(candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Placeholder for student custom restaurant reranking."""
    return candidates


def choose_bundle(bundle_candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any] | None:
    """Placeholder for student bundle search / scoring."""
    return bundle_candidates[0] if bundle_candidates else None
