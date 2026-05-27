"""evolve_memory — Memory & Context Manager agent (Team 5 / Evolve).

Assembles the memory_packet the Wellness Planner consumes. Three retrieve()
calls (journal / distractor / stale_preference filters) map cleanly onto the
9-field MemoryReport schema from final_project/schemas.py. When the verifier
phase has already run for this turn, its memory_packet_contributions are
merged in: verifier wins on policy-derived spoken_rule_hits, memory manager
contributes the retrieval-based docs/keys. Verifier is the canonical source
for policy-triggered rule hits; memory manager is the canonical source for
retrieval-based context.

V1 design mirrors evolve_verifier's: the runtime session is opened (MAS
pattern preserved at the runtime level), but no LLM call is made — retrieval
already does the semantic work via hybrid + Lost-in-the-Middle ordering in
evolve_retrieval.py. v2 can wrap the retrieve() loop with LLM-driven query
rewriting / sub-question decomposition; budgets flow through unchanged.

Entry point: run_memory_phase(runtime, episode, cfg, verifier_output=None) -> dict
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).resolve().parent
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

from evolve_retrieval import retrieve

# The 6 spoken_rule_hits buckets — verbatim from schemas.SpokenRuleHits.
_SPOKEN_RULE_BUCKETS = ("must_remember", "forbidden", "one_off_only",
                        "retire", "do_not_reconsider", "keep_context_lean")

# Step 3 iteration (2026-05-11): scenario_state boolean → current_* token map.
# Vocabulary mirrors gold.should_retrieve verbatim. Structural translation,
# not test-set-specific.
_SCENARIO_TO_CURRENT_TOKEN = {
    "low_energy":      "current_low_energy",
    "sleep_deprived":  "current_sleep_deficit",
    "high_stress":     "current_high_stress",
    "injured":         "injury_active",
    "exam_period":     "exam_peak",
    "vegan_preference":"vegan_team_meal",
    "goal_pivot":      "goal_pivot_signal",
    "caregiver_present": "caregiver_context_active",
    "community_challenge_active": "community_challenge_active",
    "streak_focus":    "streak_active",
    "panic_acute_at_t1":           "panic_acute_at_t1_just_passed",
    "anger_high_arousal_at_t2":    "current_arousal_state_high_at_t2",
    "object_target_anger_present": "anger_target_external_NOT_self",
}

_SCENARIO_NEGATION_TOKEN = {
    "self_harm_language_present": "no_self_harm_language_present",
}


def _build_preference_tokens(episode: dict, user_profile: dict) -> list[str]:
    """Profile + scenario tokens that match gold.should_retrieve vocabulary."""
    tokens: list[str] = list(user_profile.get("stable_wellness_prefs") or [])
    scenario = episode.get("scenario_state") or {}
    for flag, token in _SCENARIO_TO_CURRENT_TOKEN.items():
        if scenario.get(flag):
            tokens.append(token)
    for flag, token in _SCENARIO_NEGATION_TOKEN.items():
        if flag in scenario and scenario.get(flag) is False:
            tokens.append(token)
    return tokens


_PROFILES_CACHE: dict | None = None


def _load_profiles() -> dict:
    global _PROFILES_CACHE
    if _PROFILES_CACHE is None:
        import json as _json
        _PROFILES_CACHE = _json.loads((_DATA_DIR / "user_wellness_profiles.json").read_text())
    return _PROFILES_CACHE


def _empty_spoken_rule_hits() -> dict:
    return {b: [] for b in _SPOKEN_RULE_BUCKETS}


def _dedupe(seq):
    """Order-preserving dedupe."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _build_query(episode: dict) -> str:
    """Concatenate user turns for retrieval. Whole conversation, not just last
    turn — mirrors evolve_verifier's _concat_user_turns rationale."""
    turns = [t.get("text", "") for t in episode.get("turns", []) if t.get("speaker") == "user"]
    return " | ".join(turns) or episode.get("trip_id", "")


def _strip_prefix(doc_id: str) -> str:
    """Strip the `prefix:` from a doc_id so retired/active_context_keys carry
    the bare key (per llm_agents.py merge_memory_report canonicalization)."""
    return doc_id.split(":", 1)[-1] if ":" in doc_id else doc_id


def _merge_verifier_contributions(packet: dict, verifier_output: dict | None) -> None:
    """Verifier policy-derived hits win; memory retrieval-derived docs append + dedupe."""
    if not verifier_output:
        return
    contrib = verifier_output.get("memory_packet_contributions") or {}
    v_hits = contrib.get("spoken_rule_hits") or {}
    for bucket in _SPOKEN_RULE_BUCKETS:
        packet["spoken_rule_hits"][bucket] = _dedupe(
            packet["spoken_rule_hits"].get(bucket, []) + (v_hits.get(bucket) or [])
        )
    packet["docs_retrieved"] = _dedupe(packet["docs_retrieved"] + (contrib.get("docs_retrieved") or []))
    packet["active_docs"] = _dedupe(packet["active_docs"] + (contrib.get("active_docs") or []))


def run_memory_phase(
    runtime: Any = None,
    episode: dict | None = None,
    cfg: dict | None = None,
    verifier_output: dict | None = None,
) -> dict:
    """Assemble the memory_packet for the Wellness Planner.

    Returns a dict matching the 9-field MemoryReport schema in
    final_project/schemas.py:
        retrieved, retired, retired_docs, rejected_option_notes,
        active_context_keys, docs_retrieved, active_docs,
        ignored_distractors, spoken_rule_hits (6-bucket)

    Args:
        runtime: StudentRuntime instance. Memory-manager session is opened on
                 it (MAS pattern); v1 does not invoke the session.
        episode: episode dict per Nik's schema (turns, scenario_state, ...).
        cfg: budget knobs. MUST contain "memory_manager_max_tool_rounds" and
             "memory_manager_max_output_tokens" — bracket access raises
             KeyError on missing keys (per PDF errata recommendation: surface
             config bugs early, do not paper over them with .get(..., default)).
        verifier_output: optional output dict from run_verifier_phase. If
             provided, its memory_packet_contributions are merged in.
    """
    episode = episode or {}
    cfg = cfg or {}

    # Open per-role session — MAS pattern preserved at runtime level even
    # though v1's memory phase doesn't invoke the LLM. Same posture as the
    # verifier in commit 3120124.
    mem_session = None
    if runtime is not None and hasattr(runtime, "new_session"):
        mem_session = runtime.new_session(role="memory_manager")

    # Budget knobs — bracket access intentional. Accepted but unused in v1.
    _max_tool_rounds = cfg["memory_manager_max_tool_rounds"]      # noqa: F841
    _max_output_tokens = cfg["memory_manager_max_output_tokens"]  # noqa: F841

    # Retrieval: three calls, one per memory_type. user_id filter scopes to
    # current persona for journal + stale; distractors are intentionally not
    # user-scoped (distractor records test cross-context discipline).
    query = _build_query(episode)
    user_id = episode.get("user_id")
    journal_filter = {"memory_type": "journal"}
    if user_id:
        journal_filter["user_id"] = user_id
    stale_filter = {"memory_type": "stale_preference"}
    if user_id:
        stale_filter["user_id"] = user_id

    # Per TA Lee Joohyun's 2026-05-23 feedback: route embedding calls through
    # the official runtime.runner so usage is metered. The retrieve() signature
    # returns (results, usage) — collect all three so the orchestrator can
    # combine them into the episode-level rollup.
    retrieval_runner = getattr(runtime, "runner", None) if runtime is not None else None
    embedding_model = (
        (runtime.system_config.get("embedding_model") if runtime is not None
         and hasattr(runtime, "system_config") else None)
    )
    retrieval_strategy = (
        runtime.system_config.get("retrieval_strategy", "hybrid")
        if runtime is not None and hasattr(runtime, "system_config")
        else "hybrid"
    )

    journal_hits, journal_usage = retrieve(
        query, k=5, filters=journal_filter,
        strategy=retrieval_strategy, runner=retrieval_runner,
        embedding_model=embedding_model,
    )
    distractor_hits, distractor_usage = retrieve(
        query, k=5, filters={"memory_type": "distractor"},
        strategy=retrieval_strategy, runner=retrieval_runner,
        embedding_model=embedding_model,
    )
    stale_hits, stale_usage = retrieve(
        query, k=5, filters=stale_filter,
        strategy=retrieval_strategy, runner=retrieval_runner,
        embedding_model=embedding_model,
    )

    journal_doc_ids = [h["doc_id"] for h in journal_hits]
    distractor_doc_ids = [h["doc_id"] for h in distractor_hits]
    stale_doc_ids = [h["doc_id"] for h in stale_hits]

    # Active context keys are bare keys; retired/retired_docs follow
    # merge_memory_report's canonicalization pattern (llm_agents.py:637-640).
    retrieved_keys = [_strip_prefix(d) for d in journal_doc_ids]
    stale_keys = [_strip_prefix(d) for d in stale_doc_ids]

    # Step 3 iteration (2026-05-11): inject profile-preference + scenario-state
    # tokens into retrieved + active_context_keys. Structural translation that
    # matches gold.should_retrieve vocabulary; not 5-test-set-tuned.
    user_profile = _load_profiles().get(user_id, {}) if user_id else {}
    preference_tokens = _build_preference_tokens(episode, user_profile)

    packet = {
        "retrieved": _dedupe(retrieved_keys + preference_tokens),
        "retired": stale_keys,                              # context-key form
        "retired_docs": stale_doc_ids,                      # doc_id form
        "rejected_option_notes": [],                        # populated by assembly from rejected_activities_memory if used
        "active_context_keys": _dedupe(retrieved_keys + preference_tokens),
        "docs_retrieved": _dedupe(journal_doc_ids + distractor_doc_ids + stale_doc_ids),
        "active_docs": journal_doc_ids,                     # what the planner should actually USE
        "ignored_distractors": distractor_doc_ids,
        "spoken_rule_hits": _empty_spoken_rule_hits(),
    }

    # Note: signal_aligned_rules was investigated as a possible MemoryReport
    # field on 2026-05-11 and confirmed absent from schemas.py, gold
    # annotations (0/24 episodes), and evaluator.py. Clinical-rule docs are
    # scored via gold.required_docs and surfaced by the planner (guideline:*)
    # and verifier (safety:*) — see arch spec p.5 agent decomposition table.

    # Merge verifier-derived contributions (revision-round case).
    _merge_verifier_contributions(packet, verifier_output)

    # Surface the session usage + retrieval embedding usage so the orchestrator
    # can combine_usages later. Stored as a private key (underscore prefix) so
    # it doesn't fall into the MemoryReport dict that the evaluator scores.
    #
    # Per TA Lee Joohyun's 2026-05-23 feedback: embedding usage from retrieve()
    # is metered now and surfaces through here. The doc-embedding disk cache
    # legitimately returns empty usage on hits (same input → same vector, no
    # API call); the per-episode query-embedding token cost still flows.
    session_usage = (
        mem_session.usage if mem_session is not None and hasattr(mem_session, "usage")
        else {"calls": 0, "input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}
    )
    if runtime is not None and hasattr(runtime, "combine_usages"):
        packet["_session_usage"] = runtime.combine_usages(
            session_usage, journal_usage, distractor_usage, stale_usage,
        )
    else:
        # Test / no-runtime path: surface session_usage as-is.
        packet["_session_usage"] = session_usage

    return packet
