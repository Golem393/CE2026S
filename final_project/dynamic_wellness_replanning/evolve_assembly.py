"""evolve_assembly — TravelDecision submission builder (Team 5 / Evolve).

The course's grading harness uses travel-domain field names
(flight_id, hotel_id, restaurant_id, activity_id) per TA confirmation
2026-05-11. Internal team code uses wellness-domain names
(exercise_id, meditation_id, nutrition_id, habit_id) per the validated
domain mapping. This module applies the field-name translation at the
submission boundary so internal code stays domain-coherent while the
output matches the grading harness contract.

The mapping is defined by WELLNESS_TO_TRAVEL below. The shim is
controlled by cfg["use_travel_field_names"], which defaults to True per
the same TA confirmation. Toggling False is supported for debugging /
local inspection (keys stay in wellness form, e.g. for diffing against
the validators in scripts/).

Builds memory_report by merging the three agents' contributions:
  - Memory & Context Manager: retrieved + retired + retired_docs +
    rejected_option_notes + active_context_keys + docs_retrieved +
    active_docs + ignored_distractors (from evolve_memory's packet)
  - Wellness Planner: clinical_rule_docs (guideline:* IDs) appended
    into docs_retrieved + active_docs; rejected_options appended into
    rejected_option_notes
  - Safety Verifier: spoken_rule_hits (6-bucket, verifier wins per the
    canonical-source contract) + safety:* doc IDs appended into
    docs_retrieved
"""

from __future__ import annotations

from typing import Any

# Domain mapping — wellness names internally, travel names at submission boundary.
# Surfaced as a module-level dict so graders can read the translation directly.
WELLNESS_TO_TRAVEL: dict[str, str] = {
    "exercise_id":   "flight_id",
    "meditation_id": "hotel_id",
    "nutrition_id":  "restaurant_id",
    "habit_id":      "activity_id",
}

_SPOKEN_RULE_BUCKETS = ("must_remember", "forbidden", "one_off_only",
                        "retire", "do_not_reconsider", "keep_context_lean")


def _dedupe(seq):
    """Order-preserving dedupe."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _empty_spoken_rule_hits() -> dict:
    return {b: [] for b in _SPOKEN_RULE_BUCKETS}


def _build_memory_report(memory_packet: dict, planner_output: dict,
                         verifier_output: dict) -> dict:
    """Merge memory + planner + verifier into the 9-field MemoryReport.

    Field-by-field ownership per architecture spec p.5:
      retrieved              ← memory.retrieved (context keys)
      retired                ← memory.retired (context keys)
      retired_docs           ← memory.retired_docs (doc_ids)
      rejected_option_notes  ← planner.rejected_options (planner-derived)
      active_context_keys    ← memory.active_context_keys
      docs_retrieved         ← UNION(memory + verifier.safety + planner.guidelines)
      active_docs            ← memory.active_docs ∪ verifier.active_docs
      ignored_distractors    ← memory.ignored_distractors
      spoken_rule_hits       ← VERIFIER WINS (policy-derived, the canonical
                                source for these 6 buckets)
    """
    memory_packet = memory_packet or {}
    planner_output = planner_output or {}
    verifier_output = verifier_output or {}

    verifier_mp = verifier_output.get("memory_packet_contributions") or {}
    verifier_docs = verifier_mp.get("docs_retrieved") or []
    verifier_active = verifier_mp.get("active_docs") or []
    verifier_hits = verifier_mp.get("spoken_rule_hits") or {}

    planner_rule_docs = planner_output.get("clinical_rule_docs") or []
    # rejected_options is a list of (category, reason) tuples — flatten to strings
    planner_rejected = planner_output.get("rejected_options") or []
    rejected_notes = [f"rejected_{cat}:{reason}" for cat, reason in planner_rejected]

    # Spoken rule hits: union of verifier (safety/policy-derived, canonical)
    # and planner (content/preference, LLM-extracted from user turns per the
    # Step 2 iteration on 2026-05-11). Verifier values are kept first; planner
    # values are appended and deduped. Per-bucket. This is the planner_owns_
    # content / verifier_owns_safety boundary made explicit.
    planner_hits = planner_output.get("spoken_rule_hits") or {}
    spoken_rule_hits = _empty_spoken_rule_hits()
    for bucket in _SPOKEN_RULE_BUCKETS:
        spoken_rule_hits[bucket] = _dedupe(
            list(verifier_hits.get(bucket) or []) + list(planner_hits.get(bucket) or [])
        )

    # Step 3 cross-bucket lift: must_remember tokens belong in retrieved +
    # active_context_keys per gold convention (e.g. crisis_signal_t3_logged
    # appears in both should_retrieve and required_spoken_rules.must_remember).
    must_remember_lift = list(spoken_rule_hits.get("must_remember") or [])

    return {
        "retrieved":             _dedupe(list(memory_packet.get("retrieved") or []) + must_remember_lift),
        "retired":               list(memory_packet.get("retired") or []),
        "retired_docs":          list(memory_packet.get("retired_docs") or []),
        "rejected_option_notes": _dedupe(list(memory_packet.get("rejected_option_notes") or []) + rejected_notes),
        "active_context_keys":   _dedupe(list(memory_packet.get("active_context_keys") or []) + must_remember_lift),
        "docs_retrieved": _dedupe(
            list(memory_packet.get("docs_retrieved") or [])
            + list(verifier_docs)
            + list(planner_rule_docs)
        ),
        "active_docs": _dedupe(
            list(memory_packet.get("active_docs") or [])
            + list(verifier_active)
        ),
        "ignored_distractors":   list(memory_packet.get("ignored_distractors") or []),
        "spoken_rule_hits":      spoken_rule_hits,
    }


def maybe_remap_field_names(submission: dict, cfg: dict) -> dict:
    """Apply wellness → travel field-name shim if cfg flag is set.

    Default per TA confirmation 2026-05-11 is True (travel field names).
    Bracket access — KeyError surfaces missing-config bugs early.
    """
    if cfg["use_travel_field_names"]:
        return {WELLNESS_TO_TRAVEL.get(k, k): v for k, v in submission.items()}
    return submission


def build_submission(memory_packet: dict, planner_output: dict,
                     verifier_output: dict, cfg: dict) -> dict:
    """Assemble the TravelDecision-shaped submission payload.

    Internal code passes wellness-named fields all the way through; the
    final remap step at the submission boundary applies the field-name
    translation per the TA's grading contract (2026-05-11).
    """
    # Notes: response_text (T3 bypass template) wins if planner was disabled;
    # otherwise empty (planner output isn't human-readable in v1).
    notes = ""
    if planner_output.get("planner_disabled"):
        notes = planner_output.get("response_text") or ""

    submission = {
        "exercise_id":   planner_output.get("exercise_id"),
        "meditation_id": planner_output.get("meditation_id"),
        "nutrition_id":  planner_output.get("nutrition_id"),
        "habit_id":      planner_output.get("habit_id"),
        "memory_report": _build_memory_report(memory_packet, planner_output, verifier_output),
        "notes":         notes,
        "debug":         {
            "active_signals":          planner_output.get("active_signals", []),
            "verifier_tier":           verifier_output.get("tier"),
            "verifier_verdict":        verifier_output.get("verdict"),
            "planner_disabled":        bool(planner_output.get("planner_disabled")),
            "microsteps":              planner_output.get("microsteps", {}),
            "planner_rationale_keys":  list((planner_output.get("rationale") or {}).keys()),
        },
    }
    return maybe_remap_field_names(submission, cfg)
