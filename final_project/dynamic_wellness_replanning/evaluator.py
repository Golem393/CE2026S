"""evaluator — Wellness-domain evaluator (Team 5 / Evolve).

Mirrors the structure of final_project/dynamic_travel_replanning/evaluator.py
but scores against wellness gold field names. Reuses the canonical scoring
primitives from the travel evaluator (_overlap, _avoidance, etc.) so per-bucket
math is identical — only the gold field names change.

Submission shape: per TA confirmation 2026-05-11, the team's submission uses
travel field names (flight_id / hotel_id / restaurant_id / activity_id) at
the boundary even though the inventory is wellness. This evaluator does the
cross-name match — e.g., submission["flight_id"] is scored against
gold["acceptable_exercises"].

Wellness ↔ travel gold field map:
    submission.flight_id      vs gold.acceptable_exercises    (travel: acceptable_flights)
    submission.hotel_id       vs gold.acceptable_meditations  (travel: acceptable_hotels)
    submission.restaurant_id  vs gold.good_nutrition          (travel: good_restaurants)
    submission.activity_id    vs gold.good_habits             (travel: good_activities)
    (memory_report fields use the same names in both domains — no remap.)

v1 design constraints:
- No wellness toolbox (env) exists yet. Metrics that the travel evaluator
  computes by looking up inventory items via env (semantic_fit_rate via
  item.semantic_tags, bundle_coherence_rate via cross-item validation,
  policy_ok via policy lookup, update_handling_rate via scenario penalties)
  are computed here from data we DO have access to:
    semantic_fit_rate          — _overlap(submission.debug.active_signals + soft_tags,
                                          gold.soft_tags) when inventory soft_tags
                                  aren't reachable
    bundle_coherence_rate      — 1.0 stub (wellness has no cross-item dependencies in v1)
    update_handling_rate       — 1.0 if scenario has update flags AND memory_report
                                  shows retire/should_retire activity; else 0.5
    policy_ok                  — read from submission.debug.verifier_verdict
                                  (1.0 unless verdict == "block" without a corresponding
                                  T3 bypass justification — defensible v1 heuristic)

The row shape returned by evaluate_episode is IDENTICAL to the travel
evaluator's row shape, so summarize_rows_student from the travel module
works without modification. Re-exported below for convenience.

Entry: evaluate_episode(env, episode, submission, gold=None) -> dict
       summarize_rows_student(rows, fixed_baseline_cost=0.03) -> dict
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List

_DATA_DIR = Path(__file__).resolve().parent
_FINAL_PROJECT = _DATA_DIR.parent
_TRAVEL_EVALUATOR_PATH = _FINAL_PROJECT / "dynamic_travel_replanning" / "evaluator.py"

# Lazy-loaded inventory soft_tags lookup, keyed by category then by item ID.
# Populated on first call to _load_inventory_soft_tags().
_INVENTORY_SOFT_TAGS: dict[str, dict[str, list[str]]] | None = None


def _load_inventory_soft_tags() -> dict[str, dict[str, list[str]]]:
    """Build per-category soft_tags lookup once per process.

    Returns: {category_name: {item_id: [soft_tag_strings]}}
    """
    global _INVENTORY_SOFT_TAGS
    if _INVENTORY_SOFT_TAGS is not None:
        return _INVENTORY_SOFT_TAGS
    out: dict[str, dict[str, list[str]]] = {}
    for category, filename, id_field in [
        ("exercise",   "inventory_exercises.json",   "exercise_id"),
        ("meditation", "inventory_meditations.json", "meditation_id"),
        ("nutrition",  "inventory_nutrition.json",   "nutrition_id"),
        ("habit",      "inventory_habits.json",      "habit_id"),
    ]:
        inv = json.loads((_DATA_DIR / filename).read_text())
        out[category] = {item[id_field]: list(item.get("soft_tags") or []) for item in inv}
    _INVENTORY_SOFT_TAGS = out
    return out

# Load the travel evaluator by file path to avoid the import-name collision
# with this module (both named `evaluator`). importlib.util gives us the
# module under a private alias, sys.modules / sys.path stay untouched.
_spec = importlib.util.spec_from_file_location("_team5_travel_evaluator", _TRAVEL_EVALUATOR_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load travel evaluator from {_TRAVEL_EVALUATOR_PATH}")
_travel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_travel)

# Re-use the canonical scoring primitives and the summary aggregator from
# the travel evaluator. Single source of truth for the rubric math.
_overlap = _travel._overlap
_avoidance = _travel._avoidance
_normalize_key = _travel._normalize_key
_normalized_set = _travel._normalized_set
_normalize_retrieved_items = _travel._normalize_retrieved_items
_doc_keys = _travel._doc_keys
_active_context_hygiene = _travel._active_context_hygiene
_normalize_rejected_reason_set = _travel._normalize_rejected_reason_set
_infer_retired_docs = _travel._infer_retired_docs
_summary_bucket_means = _travel._summary_bucket_means
summarize_rows_student = _travel.summarize_rows_student
summarize_rows = _travel.summarize_rows


def _hard_constraint_rate(submission: Dict[str, Any], gold: Dict[str, Any]) -> float:
    """Wellness gold's required_hard typically includes constraints like
    'under_energy_budget', 'safe_for_state', 'schedule_coherent',
    'contraindication_clean'. v1 evaluator can verify 'safe_for_state' (the
    one constraint that's fully boundary-checkable without a wellness env)
    via: all 4 picks must appear in their respective acceptable_* gold lists
    (i.e., they passed the contraindication+budget filtering at pick time).

    v2 should add: live contraindication lookup against inventory_*.json
    for the picked items, energy_cost summing against episode.budget_total,
    and schedule_coherent check via state_buckets.json.
    """
    required = gold.get("required_hard") or []
    if not required:
        return 1.0
    # Cross-name overlap: travel-named submission keys vs wellness-named gold lists.
    flight_ok       = submission.get("flight_id")     in (gold.get("acceptable_exercises") or [])
    hotel_ok        = submission.get("hotel_id")      in (gold.get("acceptable_meditations") or [])
    restaurant_ok   = submission.get("restaurant_id") in (gold.get("good_nutrition") or [])
    activity_ok     = submission.get("activity_id")   in (gold.get("good_habits") or [])
    # "safe_for_state" / "contraindication_clean" / "under_energy_budget" / "schedule_coherent"
    # are all interpreted as: did the picked items pass their respective acceptable_* gates?
    # If gold lists are empty for a category (e.g. crisis bypass), passing == picking None.
    def _category_pass(submission_key: str, gold_list_key: str) -> bool:
        gold_list = gold.get(gold_list_key) or []
        pick = submission.get(submission_key)
        if not gold_list:
            return pick is None  # gold expects skip; submission picks None
        return pick in gold_list
    constraint_score = sum([
        _category_pass("flight_id", "acceptable_exercises"),
        _category_pass("hotel_id", "acceptable_meditations"),
        _category_pass("restaurant_id", "good_nutrition"),
        _category_pass("activity_id", "good_habits"),
    ]) / 4.0
    return constraint_score


def _semantic_fit_rate(submission: Dict[str, Any], gold: Dict[str, Any]) -> float:
    """Soft-tag overlap between the picked items' soft_tags (looked up in
    inventory_*.json) and gold.soft_tags. Mirrors the travel evaluator's
    pattern at evaluator.py:353: _overlap(list(chosen_tags), gold.soft_tags).

    Submission uses travel field names per TA contract — translate back to
    wellness inventory categories for lookup. All 4 picks' tags union.
    """
    gold_tags = gold.get("soft_tags") or []
    if not gold_tags:
        return 1.0  # no gold soft_tags to score against; treat as full credit
    lookup = _load_inventory_soft_tags()
    chosen_tags: set[str] = set()
    for sub_key, category in (
        ("flight_id",     "exercise"),
        ("hotel_id",      "meditation"),
        ("restaurant_id", "nutrition"),
        ("activity_id",   "habit"),
    ):
        item_id = submission.get(sub_key)
        if item_id:
            chosen_tags.update(lookup.get(category, {}).get(item_id, []))
    if not chosen_tags:
        return 0.0  # T3 bypass / planner_disabled — no picks, no tags
    return _overlap(list(chosen_tags), gold_tags)


def _update_handling_rate(submission: Dict[str, Any], episode: Dict[str, Any],
                          gold: Dict[str, Any]) -> float:
    """Did the system handle in-conversation state updates?
    v1 heuristic: if the episode has update-signal flags AND memory_report
    surfaces retirement activity, score 1.0. If episode is benign and no
    retirement is required, score 1.0. Mixed cases score 0.5.
    """
    scenario = episode.get("scenario_state") or {}
    has_update_signal = any(scenario.get(k) for k in
                            ("goal_pivot", "sleep_deprived", "injured",
                             "exam_period", "low_energy"))
    mr = submission.get("memory_report") or {}
    has_retirement = bool(mr.get("retired") or mr.get("retired_docs"))
    gold_expects_retire = bool(gold.get("should_retire") or gold.get("stale_docs_to_retire"))
    if has_update_signal and gold_expects_retire:
        return 1.0 if has_retirement else 0.0
    if not has_update_signal and not gold_expects_retire:
        return 1.0
    return 0.5  # partial / ambiguous


def _policy_ok(submission: Dict[str, Any]) -> float:
    """1.0 unless the verifier emitted block/revise for a non-safety reason.
    The verifier's block on T3 is the CORRECT action — that's not a policy
    failure, that's policy enforcement. v1 reads debug.verifier_verdict:
        verdict == 'pass'   -> 1.0  (no policy concerns)
        verdict == 'revise' -> 1.0  (revision applied; system course-corrected)
        verdict == 'block'  -> 1.0  (T3 path; correct behaviour)
        anything else       -> 0.0
    """
    debug = submission.get("debug") or {}
    verdict = debug.get("verifier_verdict")
    return 1.0 if verdict in ("pass", "revise", "block") else 0.0


def _spoken_rule_compliance(submission: Dict[str, Any], gold: Dict[str, Any]) -> float:
    """Per-bucket _overlap across the 6 spoken_rule_hits buckets, averaged
    over buckets that have non-empty gold expectations."""
    spoken_gold = gold.get("required_spoken_rules") or {}
    mr = submission.get("memory_report") or {}
    submitted_hits = mr.get("spoken_rule_hits") or {}
    bucket_scores: List[float] = []
    for bucket in ("must_remember", "forbidden", "one_off_only",
                   "retire", "do_not_reconsider", "keep_context_lean"):
        b_gold = spoken_gold.get(bucket) or []
        if b_gold:
            bucket_scores.append(_overlap(submitted_hits.get(bucket) or [], b_gold))
    return sum(bucket_scores) / len(bucket_scores) if bucket_scores else 1.0


def evaluate_episode(env: Any, episode: Dict[str, Any], submission: Dict[str, Any],
                     gold: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Score a wellness submission against a wellness episode's gold.

    `env` is accepted for signature compatibility with the travel evaluator
    but unused in v1 — wellness has no toolbox yet, and all v1 metrics are
    computable from gold + submission + episode dicts directly.
    """
    gold = gold or episode.get("gold")
    if not gold:
        raise ValueError(f"Gold not provided for episode {episode.get('trip_id')}")

    # === Feasibility bucket components ===
    hard_rate = _hard_constraint_rate(submission, gold)
    coherence = 1.0  # wellness has no cross-item bundle dependencies in v1
    policy_ok = _policy_ok(submission)

    # === Preference fit bucket components ===
    semantic_fit = _semantic_fit_rate(submission, gold)
    spoken_rule_compliance_rate = _spoken_rule_compliance(submission, gold)

    # === Adaptation+Memory bucket components ===
    mr = submission.get("memory_report") or {}
    memory_retrieval_rate = _overlap(mr.get("retrieved") or [], gold.get("should_retrieve") or [])
    memory_retirement_rate = _overlap(mr.get("retired") or [], gold.get("should_retire") or [])
    rejected_option_memory_rate = _overlap(
        list(_normalize_rejected_reason_set(mr.get("rejected_option_notes") or [])),
        list(_normalize_rejected_reason_set(gold.get("should_remember_rejected") or [])),
    )
    distributed_context_rate = _overlap(mr.get("docs_retrieved") or [], gold.get("required_docs") or [])
    effective_retired_docs = _infer_retired_docs(
        mr.get("retired") or [], mr.get("retired_docs") or []
    )
    stale_doc_retirement_rate = _overlap(effective_retired_docs, gold.get("stale_docs_to_retire") or [])
    distractor_avoidance_rate = _avoidance(
        mr.get("active_docs") or [], gold.get("distractor_docs_to_avoid") or []
    )
    if gold.get("distractor_docs_to_avoid"):
        distractor_avoidance_rate = max(
            distractor_avoidance_rate,
            _overlap(mr.get("ignored_distractors") or [], gold.get("distractor_docs_to_avoid") or [])
        )
    active_context_hygiene_rate = _active_context_hygiene(
        mr.get("active_context_keys") or [],
        mr.get("active_docs") or [],
        gold,
    )
    update_handling = _update_handling_rate(submission, episode, gold)

    # === Composite: decision_quality (canonical formula, evaluator.py:419-427) ===
    # 'exactish' in wellness: did all 4 picks land in their acceptable_* lists?
    exactish = float(
        (submission.get("flight_id") in (gold.get("acceptable_exercises") or []))
        and (submission.get("hotel_id") in (gold.get("acceptable_meditations") or []))
        and (submission.get("restaurant_id") in (gold.get("good_nutrition") or []))
        and (submission.get("activity_id") in (gold.get("good_habits") or []))
    )
    decision_quality = (
        0.22 * hard_rate
        + 0.18 * semantic_fit
        + 0.16 * exactish
        + 0.12 * coherence
        + 0.18 * spoken_rule_compliance_rate
        + 0.07 * stale_doc_retirement_rate
        + 0.07 * distractor_avoidance_rate
    )

    # === Usage / cost extraction ===
    usage = submission.get("usage") or {}
    total_tokens = int(usage.get("total_tokens",
                                  usage.get("input_tokens", 0) + usage.get("output_tokens", 0)))
    cost = float(usage.get("estimated_cost_usd", 0.0))
    debug = submission.get("debug") or {}
    tool_calls = int(debug.get("tool_call_count", 0) or 0)

    return {
        "trip_id": episode.get("trip_id"),
        "difficulty_tier": episode.get("difficulty_tier", "unknown"),
        "decision_quality": round(decision_quality, 4),
        "hard_constraint_rate": round(hard_rate, 4),
        "semantic_fit_rate": round(semantic_fit, 4),
        "bundle_coherence_rate": round(coherence, 4),
        "update_handling_rate": round(update_handling, 4),
        "memory_retrieval_rate": round(memory_retrieval_rate, 4),
        "memory_retirement_rate": round(memory_retirement_rate, 4),
        "distributed_context_rate": round(distributed_context_rate, 4),
        "stale_doc_retirement_rate": round(stale_doc_retirement_rate, 4),
        "distractor_avoidance_rate": round(distractor_avoidance_rate, 4),
        "rejected_option_memory_rate": round(rejected_option_memory_rate, 4),
        "active_context_hygiene_rate": round(active_context_hygiene_rate, 4),
        "spoken_rule_compliance_rate": round(spoken_rule_compliance_rate, 4),
        "policy_ok": round(policy_ok, 4),
        "tool_calls": tool_calls,
        "tokens": total_tokens,
        "estimated_cost_usd": round(cost, 6),
    }


__all__ = [
    "evaluate_episode",
    "summarize_rows_student",
    "summarize_rows",
    "_summary_bucket_means",
]
