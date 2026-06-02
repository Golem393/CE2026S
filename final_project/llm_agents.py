from __future__ import annotations

import json
from typing import Any, Dict, List, Set

from llm_runner import LLMRunner
from llm_tools import TravelToolbox, TravelToolSession


SYSTEM_BASELINE = "llm_single_baseline"
SYSTEM_MEMORY = "llm_memory_single"
SYSTEM_MAS = "llm_anchor_mas"

MEMORY_REPORT_GUIDANCE = """
Memory report contract:
- Every array item must be a SHORT benchmark key or ID. Never put prose, explanations, copied tool text, spaces, or punctuation-heavy strings inside arrays.
- `retrieved` / `active_context_keys` / `critical_constraints`: use compact snake_case keys only, such as `avoid_red_eye`, `prefer_quiet_hotel`, `low_friction_transit`, `prefer_airport_access`, `client_dinner_polished`, `weather_safe_backup`, `bundle_discount_value`, `conference_badge_access`, `team_dietary_flex`, `late_checkin_risk`, `shuttle_bundle`.
- `retired`: use short retired keys only, such as `old_budget_cap`, `old_local_character_priority`, `old_weather_assumption`, `old_chain_absolute_rule`, `old_departure_rule`, `old_social_bundle_default`, `old_bundle_discount_absolute`, `late_checkin_irrelevant`.
- `retired_docs` / `docs_retrieved` / `active_docs` / `ignored_distractors`: use doc IDs only, such as `stale:budget_cap_archive`.
- `rejected_option_notes`: use compact `reason_key:OPTION_ID` format, such as `rejected_hotel_for_noise:HT205`.
- `spoken_rule_hits` must use benchmark keys only:
  must_remember -> `quiet_matters`, `client_ready_dinner`
  forbidden -> `red_eye`, `loud_after_10pm`
  one_off_only -> `airport_access_more_important_now`, `chain_ok_this_trip`
  retire -> `old_budget_cap`, `old_local_character_priority`, `old_weather_assumption`, `old_chain_absolute_rule`, `old_departure_rule`, `old_social_bundle_default`, `old_bundle_discount_absolute`, `late_checkin_irrelevant`
  do_not_reconsider -> `noise_rejected_hotel`, `wrong_vibe_restaurant`
  keep_context_lean -> `relevant_only`
If unsure, omit the item instead of inventing a long phrase.
""".strip()


REJECTED_NOTE_ALIASES = {
    "wrong_vibe_restaurant": "rejected_restaurant_for_vibe",
    "noise_rejected_hotel": "rejected_hotel_for_noise",
    "red_eye_rejected_flight": "rejected_flight_for_red_eye",
}

RETIRED_DOC_BY_KEY = {
    "old_budget_cap": "stale:budget_cap_archive",
    "old_local_character_priority": "stale:local_character_default",
    "avoid_chain_hotels_stable": "stale:avoid_chain_hotels_absolute",
    "old_weather_assumption": "stale:dry_weather_ops_assumption",
    "old_social_bundle_default": "stale:partner_social_default",
    "old_bundle_discount_absolute": "stale:bundle_discount_always_wins",
    "late_checkin_irrelevant": "stale:late_checkin_irrelevant",
}


CONTEXT_KEY_ALIASES = {
    "quiet_matters": "prefer_quiet_hotel",
    "quiet_room_matters": "prefer_quiet_hotel",
    "prefer_quiet_hotel_room": "prefer_quiet_hotel",
    "prefer_quiet_hotel": "prefer_quiet_hotel",
    "red_eye": "avoid_red_eye",
    "avoid_red_eye": "avoid_red_eye",
    "loud_after_10pm": "loud_after_10pm",
    "filter_noise_after_10pm": "loud_after_10pm",
    "avoid_loud_nightlife": "loud_after_10pm",
    "client_ready_dinner": "client_dinner_polished",
    "client_dinner_polished": "client_dinner_polished",
    "airport_access_more_important_now": "prefer_airport_access",
    "airport_access_one_off": "prefer_airport_access",
    "prefer_airport_access": "prefer_airport_access",
    "chain_ok_this_trip": "chain_ok_this_trip",
    "chain_exception_this_trip": "chain_ok_this_trip",
    "relevant_only": "relevant_only",
    "keep_context_lean": "relevant_only",
    "meeting_zone_namba": "meeting_zone",
    "work_functional_prep_priority": "low_friction_transit",
    "low_friction": "low_friction_transit",
    "low_friction_transit": "low_friction_transit",
    "weather_safe": "weather_safe_backup",
    "weather_safe_backup": "weather_safe_backup",
    "team_dietary_flex": "team_dietary_flex",
    "teammate_vegan": "team_dietary_flex",
    "refundable_priority": "refundable_priority",
    "conference_badge_access": "conference_badge_access",
    "badge_unlock": "conference_badge_access",
    "loyalty_bundle_value": "loyalty_bundle_value",
    "private_room_bonus": "private_room_bonus",
    "bundle_discount_value": "bundle_discount_value",
    "late_checkin_risk": "late_checkin_risk",
    "shuttle_bundle": "shuttle_bundle",
    "transfer_friction_risk": "transfer_friction_risk",
    "old_social_bundle_default": "old_social_bundle_default",
    "old_bundle_discount_absolute": "old_bundle_discount_absolute",
    "late_checkin_irrelevant": "late_checkin_irrelevant",
}

ACTIVE_CONTEXT_PRIORITY = [
    "prefer_quiet_hotel",
    "avoid_red_eye",
    "loud_after_10pm",
    "low_friction_transit",
    "client_dinner_polished",
    "team_dietary_flex",
    "prefer_airport_access",
    "weather_safe_backup",
    "conference_badge_access",
    "refundable_priority",
    "loyalty_bundle_value",
    "bundle_discount_value",
    "private_room_bonus",
    "shuttle_bundle",
    "late_checkin_risk",
    "chain_ok_this_trip",
    "local_character_if_safe",
    "transfer_friction_risk",
    "meeting_zone",
    "relevant_only",
]
ACTIVE_CONTEXT_ALLOWED = set(ACTIVE_CONTEXT_PRIORITY)


def canonicalize_context_key(value: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith(("profile:", "venue:", "city_ops:", "heuristic:", "stale:", "distractor:")):
        return cleaned
    return CONTEXT_KEY_ALIASES.get(cleaned, cleaned)


def canonicalize_context_keys(items: List[str], *, keep_doc_ids: bool = True) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        normalized = canonicalize_context_key(item)
        if not normalized:
            continue
        if not keep_doc_ids and normalized.startswith(("profile:", "venue:", "city_ops:", "heuristic:", "stale:", "distractor:")):
            continue
        if normalized not in seen:
            out.append(normalized)
            seen.add(normalized)
    return out


def inferred_retrieved_keys_from_report(report: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    candidates.extend(report.get("retrieved", []))
    candidates.extend(report.get("active_context_keys", []))
    spoken = report.get("spoken_rule_hits", {}) or {}
    for bucket in ["must_remember", "forbidden", "one_off_only", "keep_context_lean"]:
        candidates.extend(spoken.get(bucket, []))
    return canonicalize_context_keys(candidates, keep_doc_ids=False)





def synthesize_critical_constraints(context_pack: Dict[str, Any], episode: Dict[str, Any], limit: int = 6) -> List[str]:
    keys = set(canonicalize_context_keys(context_pack.get("active_context_keys", []) + context_pack.get("retrieved", []), keep_doc_ids=False))
    scenario = episode.get("scenario_state", {}) or {}
    ordered: List[str] = []

    def add(key: str) -> None:
        if key and key in ACTIVE_CONTEXT_ALLOWED and key not in ordered:
            ordered.append(key)

    for key in [
        "prefer_quiet_hotel",
        "avoid_red_eye",
        "loud_after_10pm",
        "low_friction_transit",
        "prefer_airport_access",
        "client_dinner_polished",
        "team_dietary_flex",
        "weather_safe_backup",
        "conference_badge_access",
        "refundable_priority",
        "bundle_discount_value",
        "loyalty_bundle_value",
        "late_checkin_risk",
        "shuttle_bundle",
        "transfer_friction_risk",
        "chain_ok_this_trip",
        "relevant_only",
    ]:
        if key in keys:
            add(key)

    if scenario.get("airport_priority"):
        add("prefer_airport_access")
    if scenario.get("client_dinner"):
        add("client_dinner_polished")
    if scenario.get("teammate_vegan"):
        add("team_dietary_flex")
    if scenario.get("rainy") or scenario.get("event_disruption"):
        add("weather_safe_backup")
        add("transfer_friction_risk")
    if scenario.get("badge_available"):
        add("conference_badge_access")
    if scenario.get("refund_risk"):
        add("refundable_priority")
    if scenario.get("partner_bundle"):
        add("bundle_discount_value")
    if scenario.get("loyalty_focus"):
        add("loyalty_bundle_value")
    if scenario.get("late_arrival_risk"):
        add("late_checkin_risk")
        add("shuttle_bundle")

    if not ordered:
        ordered = [
            "prefer_quiet_hotel",
            "avoid_red_eye",
            "low_friction_transit",
            "relevant_only",
        ]
    return ordered[:limit]

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def canonicalize_rejected_note(note: str) -> str:
    if not note:
        return note
    if ":" not in note:
        return REJECTED_NOTE_ALIASES.get(note, note)
    prefix, suffix = note.split(":", 1)
    return f"{REJECTED_NOTE_ALIASES.get(prefix, prefix)}:{suffix}"


def scenario_active_context_hints(episode: Dict[str, Any]) -> List[str]:
    hooks = episode.get("scenario_hooks", {}) or {}
    state = episode.get("scenario_state", {}) or {}
    stakeholder_ids = set(state.get("stakeholder_ids", []) or hooks.get("stakeholders", []) or [])
    hints: List[str] = []
    if state.get("airport_priority"):
        hints.append("prefer_airport_access")
    if state.get("rainy"):
        hints.append("weather_safe_backup")
    if state.get("client_dinner") or "stakeholder:client_polished" in stakeholder_ids:
        hints.append("client_dinner_polished")
    if state.get("chain_exception"):
        hints.append("chain_ok_this_trip")
    if state.get("partner_bundle"):
        hints.append("bundle_discount_value")
    if state.get("badge_available") or hooks.get("badge_status") == "active":
        hints.append("conference_badge_access")
    if state.get("refund_risk") or hooks.get("schedule_volatility") == "high":
        hints.append("refundable_priority")
    if state.get("loyalty_focus"):
        hints.append("loyalty_bundle_value")
    if state.get("late_arrival_risk") or hooks.get("late_arrival_risk"):
        hints.append("late_checkin_risk")
    if state.get("teammate_vegan") or "stakeholder:teammate_vegan" in stakeholder_ids:
        hints.append("team_dietary_flex")
    if state.get("event_disruption") or hooks.get("event_sensitive"):
        hints.append("low_friction_transit")
    if state.get("partner_bundle") and hooks.get("bundle_watch"):
        hints.extend(["private_room_bonus", "shuttle_bundle"])
    return canonicalize_context_keys(hints, keep_doc_ids=False)


def synthesize_active_context_keys(report: Dict[str, Any], summary: Dict[str, Any], episode: Dict[str, Any], cap: int) -> List[str]:
    spoken = report.get("spoken_rule_hits", {}) or {}
    candidate_buckets = [
        spoken.get("must_remember", []),
        spoken.get("forbidden", []),
        spoken.get("one_off_only", []),
        spoken.get("keep_context_lean", []),
        report.get("active_context_keys", []),
        report.get("retrieved", []),
        summary.get("retrieved_keys_seen", []),
        scenario_active_context_hints(episode),
    ]
    retired_now = set(canonicalize_context_keys(report.get("retired", []), keep_doc_ids=False))
    ordered: List[str] = []
    seen: Set[str] = set()
    for bucket in candidate_buckets:
        for item in bucket:
            normalized = canonicalize_context_key(item)
            if not normalized or normalized.startswith(("profile:", "venue:", "city_ops:", "heuristic:", "stale:", "distractor:")):
                continue
            if normalized not in ACTIVE_CONTEXT_ALLOWED:
                continue
            if normalized in retired_now or normalized.startswith("old_"):
                continue
            if normalized not in seen:
                ordered.append(normalized)
                seen.add(normalized)
    ordered.sort(key=lambda key: ACTIVE_CONTEXT_PRIORITY.index(key) if key in ACTIVE_CONTEXT_PRIORITY else len(ACTIVE_CONTEXT_PRIORITY))
    return ordered[:cap]


def observable_episode(episode: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in episode.items() if key != "gold"}


def short_key_schema(max_length: int = 40) -> Dict[str, Any]:
    return {
        "type": "string",
        "maxLength": max_length,
        "pattern": rf"^[a-z][a-z0-9_:-]{{0,{max_length - 1}}}$",
    }


def doc_id_schema(max_length: int = 72) -> Dict[str, Any]:
    return {
        "type": "string",
        "maxLength": max_length,
        "pattern": rf"^[A-Za-z][A-Za-z0-9_:-]{{0,{max_length - 1}}}$",
    }


def rejected_note_schema() -> Dict[str, Any]:
    return {
        "type": "string",
        "maxLength": 64,
        "pattern": r"^[a-z][a-z0-9_]*:(FL|HT|RS|AC)[0-9]{3}$",
    }


def spoken_rule_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "must_remember": {"type": "array", "items": short_key_schema(32), "maxItems": 4},
            "forbidden": {"type": "array", "items": short_key_schema(32), "maxItems": 4},
            "one_off_only": {"type": "array", "items": short_key_schema(40), "maxItems": 3},
            "retire": {"type": "array", "items": short_key_schema(40), "maxItems": 5},
            "do_not_reconsider": {"type": "array", "items": short_key_schema(40), "maxItems": 4},
            "keep_context_lean": {"type": "array", "items": short_key_schema(24), "maxItems": 3},
        },
        "required": ["must_remember", "forbidden", "one_off_only", "retire", "do_not_reconsider", "keep_context_lean"],
        "additionalProperties": False,
    }


try: 
    from ta_only.grounding_utils import ensure_grounded_submission as _ta_ensure_grounded_submission
except Exception:  # pragma: no cover - normal in the student release
    _ta_ensure_grounded_submission = None


def _empty_memory_report() -> Dict[str, Any]:
    return {
        "retrieved": [],
        "retired": [],
        "retired_docs": [],
        "rejected_option_notes": [],
        "active_context_keys": [],
        "docs_retrieved": [],
        "active_docs": [],
        "ignored_distractors": [],
        "spoken_rule_hits": {
            "must_remember": [],
            "forbidden": [],
            "one_off_only": [],
            "retire": [],
            "do_not_reconsider": [],
            "keep_context_lean": [],
        },
        "critical_constraints": [],
    }


def _merge_public_memory_report(payload_report: Dict[str, Any] | None, initial_report: Dict[str, Any] | None) -> Dict[str, Any]:
    report = _empty_memory_report()
    for source in (initial_report or {}, payload_report or {}):
        for key, value in source.items():
            if key == "spoken_rule_hits" and isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if subkey in report["spoken_rule_hits"] and isinstance(subvalue, list):
                        for item in subvalue:
                            if item not in report["spoken_rule_hits"][subkey]:
                                report["spoken_rule_hits"][subkey].append(item)
            elif key in report and isinstance(report[key], list) and isinstance(value, list):
                for item in value:
                    if item not in report[key]:
                        report[key].append(item)
    return report


def _id_index(rows: List[Dict[str, Any]], id_key: str) -> Dict[str, Dict[str, Any]]:
    return {str(row.get(id_key)): row for row in rows if row.get(id_key)}


def _first_id(rows: List[Dict[str, Any]], id_key: str) -> str | None:
    return str(rows[0][id_key]) if rows and rows[0].get(id_key) else None


def _public_ensure_grounded_submission(
    session: TravelToolSession,
    episode: Dict[str, Any],
    payload: Dict[str, Any] | None,
    *,
    initial_report: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Student-safe grounding fallback for the shipped baseline example.

    It validates that final option IDs exist in the local simulator inventory and
    fills missing/invalid IDs with a simple deterministic candidate.  This is not
    intended to be a strong solver; students are expected to implement better
    search, verification, and memory discipline in student_solver.py.
    """
    payload = dict(payload or {})
    env = session.toolbox.env
    city = episode.get("city")
    origin = episode.get("origin")

    flights = env.search_flights(origin, city) if origin and city else []
    hotels = env.search_hotels(city) if city else []
    restaurants = env.search_restaurants(city) if city else []
    activities = env.search_activities(city) if city else []

    # Conservative defaults: avoid red-eyes, prefer quieter/central options, and
    # keep baseline choices grounded rather than optimal.
    flight_candidates = sorted(
        flights,
        key=lambda row: (bool(row.get("red_eye")), int(row.get("fare_total", 10**9)), int(row.get("stops", 99))),
    )
    hotel_candidates = sorted(
        hotels,
        key=lambda row: (-float(row.get("quiet_score", 0)), -float(row.get("airport_access_score", 0)), int(row.get("nightly_price", 10**9))),
    )
    restaurant_candidates = sorted(
        restaurants,
        key=lambda row: (-float(row.get("quiet_score", 0)), -float(row.get("client_ready_score", 0)), int(row.get("price_level", 9))),
    )
    activity_candidates = sorted(
        activities,
        key=lambda row: (not bool(row.get("indoor")), int(row.get("price", 10**9))),
    )

    indexes = {
        "flight_id": _id_index(flights, "flight_id"),
        "hotel_id": _id_index(hotels, "hotel_id"),
        "restaurant_id": _id_index(restaurants, "restaurant_id"),
        "activity_id": _id_index(activities, "activity_id"),
    }
    defaults = {
        "flight_id": _first_id(flight_candidates, "flight_id"),
        "hotel_id": _first_id(hotel_candidates, "hotel_id"),
        "restaurant_id": _first_id(restaurant_candidates, "restaurant_id"),
        "activity_id": _first_id(activity_candidates, "activity_id"),
    }

    repaired: Dict[str, Any] = dict(payload)
    for key in ("flight_id", "hotel_id", "restaurant_id", "activity_id"):
        value = repaired.get(key)
        if not value or str(value) not in indexes[key]:
            repaired[key] = defaults[key]

    repaired["memory_report"] = _merge_public_memory_report(repaired.get("memory_report"), initial_report)
    repaired.setdefault("notes", payload.get("notes") or "Grounded fallback checked final IDs against local inventory.")
    return repaired


def ensure_grounded_submission(
    session: TravelToolSession,
    episode: Dict[str, Any],
    payload: Dict[str, Any] | None,
    *,
    initial_report: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if _ta_ensure_grounded_submission is not None:
        return _ta_ensure_grounded_submission(session, episode, payload, initial_report=initial_report)
    return _public_ensure_grounded_submission(session, episode, payload, initial_report=initial_report)


def memory_report_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "retrieved": {"type": "array", "items": short_key_schema(40), "maxItems": 8},
            "retired": {"type": "array", "items": short_key_schema(40), "maxItems": 8},
            "retired_docs": {"type": "array", "items": doc_id_schema(72), "maxItems": 6},
            "rejected_option_notes": {"type": "array", "items": rejected_note_schema(), "maxItems": 6},
            "active_context_keys": {"type": "array", "items": short_key_schema(40), "maxItems": 6},
            "docs_retrieved": {"type": "array", "items": doc_id_schema(72), "maxItems": 8},
            "active_docs": {"type": "array", "items": doc_id_schema(72), "maxItems": 4},
            "ignored_distractors": {"type": "array", "items": doc_id_schema(72), "maxItems": 6},
            "spoken_rule_hits": spoken_rule_schema(),
        },
        "required": [
            "retrieved",
            "retired",
            "retired_docs",
            "rejected_option_notes",
            "active_context_keys",
            "docs_retrieved",
            "active_docs",
            "ignored_distractors",
            "spoken_rule_hits",
        ],
        "additionalProperties": False,
    }


def final_decision_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "flight_id": {"type": ["string", "null"]},
            "hotel_id": {"type": ["string", "null"]},
            "restaurant_id": {"type": ["string", "null"]},
            "activity_id": {"type": ["string", "null"]},
            "memory_report": memory_report_schema(),
            "notes": {"type": "string", "maxLength": 320},
        },
        "required": ["flight_id", "hotel_id", "restaurant_id", "activity_id", "memory_report", "notes"],
        "additionalProperties": False,
    }


def planner_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "flight_id": {"type": ["string", "null"]},
            "hotel_id": {"type": ["string", "null"]},
            "restaurant_id": {"type": ["string", "null"]},
            "activity_id": {"type": ["string", "null"]},
            "notes": {"type": "string", "maxLength": 320},
        },
        "required": ["flight_id", "hotel_id", "restaurant_id", "activity_id", "notes"],
        "additionalProperties": False,
    }


def context_pack_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "retrieved": {"type": "array", "items": short_key_schema(32), "maxItems": 6},
            "retired": {"type": "array", "items": short_key_schema(32), "maxItems": 6},
            "retired_docs": {"type": "array", "items": doc_id_schema(56), "maxItems": 4},
            "rejected_option_notes": {"type": "array", "items": rejected_note_schema(), "maxItems": 4},
            "active_context_keys": {"type": "array", "items": short_key_schema(32), "maxItems": 4},
            "docs_retrieved": {"type": "array", "items": doc_id_schema(56), "maxItems": 6},
            "active_docs": {"type": "array", "items": doc_id_schema(56), "maxItems": 2},
            "ignored_distractors": {"type": "array", "items": doc_id_schema(56), "maxItems": 4},
            "spoken_rule_hits": spoken_rule_schema(),
            "critical_constraints": {"type": "array", "items": short_key_schema(40), "maxItems": 6},
            "summary": {"type": "string", "maxLength": 180},
        },
        "required": [
            "retrieved",
            "retired",
            "retired_docs",
            "rejected_option_notes",
            "active_context_keys",
            "docs_retrieved",
            "active_docs",
            "ignored_distractors",
            "spoken_rule_hits",
            "critical_constraints",
            "summary",
        ],
        "additionalProperties": False,
    }


def verifier_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "approve": {"type": "boolean"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "retire": {"type": "array", "items": {"type": "string"}},
            "retired_docs": {"type": "array", "items": {"type": "string"}},
            "active_context_keys": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string", "maxLength": 320},
        },
        "required": ["approve", "issues", "retire", "retired_docs", "active_context_keys", "notes"],
        "additionalProperties": False,
    }


def episode_prompt(episode: Dict[str, Any]) -> str:
    lines = [
        f"trip_id: {episode['trip_id']}",
        f"family: {episode['family']}",
        f"city: {episode['city']}",
        f"origin: {episode['origin']}",
        f"traveler_id: {episode['traveler_id']}",
        f"nights: {episode['nights']}",
        f"budget_total: {episode['budget_total']}",
        f"meeting_zone: {episode['meeting_zone']}",
        f"weather: {episode['weather']}",
        f"spoken_rule_density: {episode['spoken_rule_density']}",
    ]
    if episode.get("scenario_hooks"):
        lines.append(f"scenario_hooks: {json.dumps(episode['scenario_hooks'], ensure_ascii=False, sort_keys=True)}")
    if episode.get("scenario_state"):
        lines.append(f"scenario_state: {json.dumps(episode['scenario_state'], ensure_ascii=False, sort_keys=True)}")
    lines.append("turns:")
    lines.extend(f"- {turn['speaker']}: {turn['text']}" for turn in episode["turns"])
    return "\n".join(lines)


def merge_memory_report(
    model_report: Dict[str, Any],
    session: TravelToolSession,
    *,
    active_doc_cap: int,
    active_key_cap: int,
    forced_retired: List[str] | None = None,
    forced_retired_docs: List[str] | None = None,
) -> Dict[str, Any]:
    report = dict(model_report or {})
    report.setdefault("retrieved", [])
    report.setdefault("retired", [])
    report.setdefault("retired_docs", [])
    report.setdefault("rejected_option_notes", [])
    report.setdefault("active_context_keys", [])
    report.setdefault("docs_retrieved", [])
    report.setdefault("active_docs", [])
    report.setdefault("ignored_distractors", [])
    report.setdefault(
        "spoken_rule_hits",
        {
            "must_remember": [],
            "forbidden": [],
            "one_off_only": [],
            "retire": [],
            "do_not_reconsider": [],
            "keep_context_lean": [],
        },
    )
    summary = session.summary()
    report["docs_retrieved"] = dedupe_keep_order(report["docs_retrieved"] + summary["docs_seen"])
    if not report["active_docs"]:
        report["active_docs"] = report["docs_retrieved"][:active_doc_cap]
    report["active_docs"] = dedupe_keep_order(report["active_docs"])[:active_doc_cap]
    report["active_context_keys"] = synthesize_active_context_keys(report, summary, session.episode, active_key_cap)
    report["retired"] = canonicalize_context_keys(report["retired"] + (forced_retired or []), keep_doc_ids=False)
    inferred_retired_docs = [RETIRED_DOC_BY_KEY[key] for key in report["retired"] if key in RETIRED_DOC_BY_KEY]
    report["retired_docs"] = dedupe_keep_order(report["retired_docs"] + inferred_retired_docs + (forced_retired_docs or []))
    report["ignored_distractors"] = dedupe_keep_order(report["ignored_distractors"])
    report["rejected_option_notes"] = dedupe_keep_order([
        canonicalize_rejected_note(note)
        for note in (report["rejected_option_notes"] + summary.get("rejected_option_notes_seen", []))
    ])
    for key in ["must_remember", "forbidden", "one_off_only", "retire", "do_not_reconsider", "keep_context_lean"]:
        report["spoken_rule_hits"][key] = canonicalize_context_keys(report["spoken_rule_hits"].get(key, []), keep_doc_ids=False)
    report["retrieved"] = dedupe_keep_order(
        inferred_retrieved_keys_from_report(report) + summary.get("retrieved_keys_seen", [])
    )[:8]
    return report


def tool_result(
    runner: LLMRunner,
    runner_result: Dict[str, Any],
    session: TravelToolSession,
    *,
    active_doc_cap: int,
    active_key_cap: int,
    forced_retired: List[str] | None = None,
    forced_retired_docs: List[str] | None = None,
) -> Dict[str, Any]:
    payload = dict(runner_result["parsed"])
    payload["memory_report"] = merge_memory_report(
        payload.get("memory_report", {}),
        session,
        active_doc_cap=active_doc_cap,
        active_key_cap=active_key_cap,
        forced_retired=forced_retired,
        forced_retired_docs=forced_retired_docs,
    )
    return {
        "submission": payload,
        "usage": runner.combine_usages(runner_result["usage"], session.usage),
        "response_ids": runner_result.get("response_ids", [runner_result.get("response_id")]),
        "tool_trace": session.summary()["tool_trace"],
        "retrieval": {
            "docs_seen": session.summary()["docs_seen"],
            "rejected_memory_seen": session.summary()["rejected_memory_seen"],
            "tool_call_count": session.summary()["tool_call_count"],
        },
        "api_status": {"success": True},
    }


def session_tools(session: TravelToolSession, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    primitive_only = bool(config.get("primitive_tools_only", False))
    return session.tool_specs(primitive_only=primitive_only)


def run_single_tool_agent(
    runner: LLMRunner,
    toolbox: TravelToolbox,
    episode: Dict[str, Any],
    *,
    role: str,
    model: str,
    config: Dict[str, Any],
    instructions: str,
    active_doc_cap: int,
    active_key_cap: int,
    session_factory=None,
) -> Dict[str, Any]:
    # Built-in baselines can create toolbox sessions directly because
    # run_llm_baselines.py consumes their returned retrieval trace. Dynamic
    # student solvers may use either runtime.new_session(...) or
    # runtime.toolbox.new_session(...) during solve_episode(runtime); the
    # official runner registers toolbox-created sessions via the active runtime.
    if session_factory is None:
        session = toolbox.new_session(
            episode=episode,
            retrieval_strategy=config["retrieval_strategy"],
            embedding_model=config.get("embedding_model"),
            max_results=config["max_tool_results"],
            role=role,
        )
        session.bind_runner(runner)
    else:
        session = session_factory(
            retrieval_strategy=config["retrieval_strategy"],
            embedding_model=config.get("embedding_model"),
            max_results=config["max_tool_results"],
            role=role,
        )
    result = runner.run_tool_agent_json(
        model=model,
        instructions=instructions,
        input_text=episode_prompt(episode),
        json_schema=final_decision_schema(),
        schema_name=f"{role}_decision",
        tools=session_tools(session, config),
        tool_handler=session.dispatch,
        max_output_tokens=config["max_output_tokens"],
        reasoning_effort="low" if model.startswith("gpt-5") else None,
        text_verbosity="low" if model.startswith("gpt-5") else None,
        metadata={"system": config["system_name"], "trip_id": episode["trip_id"], "role": role},
        max_tool_rounds=config.get("max_tool_rounds", 8),
    )
    result["parsed"] = ensure_grounded_submission(
        session,
        episode,
        result.get("parsed", {}),
        initial_report=(result.get("parsed", {}) or {}).get("memory_report", {}),
    )
    return tool_result(
        runner,
        result,
        session,
        active_doc_cap=active_doc_cap,
        active_key_cap=active_key_cap,
    )


def run_single_baseline(
    runner: LLMRunner,
    toolbox: TravelToolbox,
    episode: Dict[str, Any],
    config: Dict[str, Any],
    *,
    session_factory=None,
) -> Dict[str, Any]:
    instructions = (
        "You are a single travel-planning agent. Use tools selectively; do not pull every inventory list broadly. "
        "Start with canonical trip context, then search only the inventories and memory notes you truly need. "
        "Prefer one or two focused searches per category over exhaustive browsing. "
        "Return strict JSON with chosen IDs and a concise memory report. "
        "Do not hallucinate IDs, and do not assume stale notes are retired unless you explicitly decided to retire them. "
        "After enough evidence, stop searching and finalize the bundle. "
        + MEMORY_REPORT_GUIDANCE
    )
    return run_single_tool_agent(
        runner,
        toolbox,
        episode,
        role="single_baseline",
        model=config["model"],
        config=config,
        instructions=instructions,
        active_doc_cap=4,
        active_key_cap=6,
        session_factory=session_factory,
    )


def run_memory_single(
    runner: LLMRunner,
    toolbox: TravelToolbox,
    episode: Dict[str, Any],
    config: Dict[str, Any],
    *,
    session_factory=None,
) -> Dict[str, Any]:
    if config.get("retirement_policy", True):
        instructions = (
            "You are a memory-aware single travel-planning agent. Use tools selectively, but explicitly search for stale notes, one-off overrides, and rejected options. "
            "Retire stale assumptions when the user says they no longer apply. "
            "Avoid carrying distractor notes into active context. "
            "Use the broader memory search before deciding when spoken rules mention old preferences, context hygiene, or not reconsidering past mistakes. If scenario hooks show bundle watch, event sensitivity, active badge access, refund risk, loyalty focus, or stakeholders, inspect the corresponding rich-context tools before finalizing. "
            "Return strict JSON with explicit retirement and spoken-rule handling. "
            + MEMORY_REPORT_GUIDANCE
        )
    else:
        instructions = (
            "You are a single travel-planning agent with access to richer memory search. Use tools selectively and pay attention to spoken rules and rejected options, "
            "but do not spend much effort on explicit memory retirement unless it is unavoidable. "
            "Return strict JSON with the final bundle and concise memory report. "
            + MEMORY_REPORT_GUIDANCE
        )
    return run_single_tool_agent(
        runner,
        toolbox,
        episode,
        role="single_memory",
        model=config["model"],
        config=config,
        instructions=instructions,
        active_doc_cap=3,
        active_key_cap=5,
    )
