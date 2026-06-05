from __future__ import annotations

import contextvars
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dynamic_travel_replanning.rtl_semantic_env import RTLSemanticEnv
from retrieval import RetrievalCorpus, lexical_score


_CONTEXT_KEY_ALIASES = {
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

_SOFT_TAG_TO_KEYS = {
    "low_friction": ["low_friction_transit"],
    "quiet": ["prefer_quiet_hotel"],
    "easy_airport_access": ["prefer_airport_access"],
    "weather_safe": ["weather_safe_backup"],
    "client_ready": ["client_dinner_polished"],
    "conference_ready": ["conference_badge_access", "low_friction_transit"],
}

_ACTIVE_STUDENT_RUNTIME = contextvars.ContextVar("active_student_runtime", default=None)


def set_active_student_runtime(runtime):
    """Set the runtime that should receive toolbox-created session traces.

    The official runner uses this while executing solve_episode(runtime).  It
    allows legacy solvers that call runtime.toolbox.new_session(...) directly
    to receive the same trace credit as solvers that call runtime.new_session(...).
    """
    return _ACTIVE_STUDENT_RUNTIME.set(runtime)


def reset_active_student_runtime(token) -> None:
    _ACTIVE_STUDENT_RUNTIME.reset(token)


def get_active_student_runtime():
    return _ACTIVE_STUDENT_RUNTIME.get()


PRIMITIVE_TOOL_NAMES = [
    "search_memory",
    "get_rejected_options",
    "get_profile_brief",
    "get_venue_brief",
    "get_city_ops_notes",
    "get_policy",
    "search_flights",
    "search_hotels",
    "search_restaurants",
    "search_activities",
]

RICH_CONTEXT_TOOL_NAMES = [
    "get_partner_promotions",
    "get_event_context",
    "get_loyalty_profile",
    "get_stakeholder_brief",
    "get_booking_constraints",
    "get_option_dependencies",
]

_RISK_TAG_TO_KEYS = {
    "noise_risk": ["loud_after_10pm"],
    "late_return_risk": ["low_friction_transit"],
    "weather_transfer_risk": ["weather_safe_backup", "low_friction_transit"],
    "badge_access": ["conference_badge_access"],
    "late_checkin_risk": ["late_checkin_risk"],
    "shuttle_bundle": ["shuttle_bundle", "low_friction_transit"],
}


def primitive_only_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    config = config or {}
    raw = config.get("primitive_tools_only")
    if raw is None:
        raw = os.getenv("RTL_PRIMITIVE_TOOLS_ONLY", "")
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def canonicalize_context_key(value: str | None) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        return ""
    return _CONTEXT_KEY_ALIASES.get(cleaned, cleaned)


def infer_context_keys_from_doc(doc: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    keys.extend(doc.get("stable_prefs", []))
    for tag in doc.get("soft_tags", []):
        keys.extend(_SOFT_TAG_TO_KEYS.get(tag, [tag]))
    for tag in doc.get("risk_tags", []):
        keys.extend(_RISK_TAG_TO_KEYS.get(tag, [tag]))
    keys.extend(doc.get("tags", []))
    out: List[str] = []
    seen = set()
    for key in keys:
        normalized = canonicalize_context_key(key)
        if normalized and normalized not in seen:
            out.append(normalized)
            seen.add(normalized)
    return out


def compact_item(row: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    return {field: row.get(field) for field in fields if field in row}


class TravelToolbox:
    def __init__(self, data_dir: str | Path, max_results: int = 4) -> None:
        self.data_dir = Path(data_dir)
        self.env = RTLSemanticEnv(data_dir)
        self.max_results = max_results
        self.memory_corpus = RetrievalCorpus(self.env.memory_corpus, cache_dir=self.data_dir / ".cache")
        self.rejected_options = list(self.env.rejected_options)

    def new_session(
        self,
        *,
        episode: Dict[str, Any],
        retrieval_strategy: str,
        embedding_model: str | None,
        max_results: int,
        role: str,
        board: Any = None,
        **session_kwargs: Any,
    ) -> "TravelToolSession":
        """Create a tool session.

        Extra keyword arguments are accepted and preserved for compatibility
        with older student solvers that attached custom boards/state to tool
        sessions.  Official scoring only uses the recorded tool trace.
        """
        session = TravelToolSession(
            toolbox=self,
            episode=episode,
            retrieval_strategy=retrieval_strategy,
            embedding_model=embedding_model,
            max_results=max_results,
            role=role,
            board=board,
            **session_kwargs,
        )
        runtime = get_active_student_runtime()
        if runtime is not None and getattr(runtime, "toolbox", None) is self:
            register = getattr(runtime, "_register_session", None)
            if callable(register):
                register(session)
        return session


class TravelToolSession:
    def __init__(
        self,
        *,
        toolbox: TravelToolbox,
        episode: Dict[str, Any],
        retrieval_strategy: str,
        embedding_model: str | None,
        max_results: int,
        role: str,
        board: Any = None,
        **session_kwargs: Any,
    ) -> None:
        self.toolbox = toolbox
        self.episode = episode
        self.retrieval_strategy = retrieval_strategy
        self.embedding_model = embedding_model
        self.max_results = max_results
        self.role = role
        self.board = board
        self.extra_session_kwargs = dict(session_kwargs)
        self.runner = None
        self.tool_trace: List[Dict[str, Any]] = []
        self.docs_seen: List[str] = []
        self.rejected_seen: List[str] = []
        self.rejected_notes_seen: List[str] = []
        self.retrieved_keys_seen: List[str] = []
        self.usage = None

    def bind_runner(self, runner) -> None:
        self.runner = runner
        self.usage = runner.empty_usage()

    def _record(self, *, name: str, arguments: Dict[str, Any], result: Any) -> None:
        preview: Any
        if isinstance(result, dict):
            if "items" in result and isinstance(result["items"], list):
                preview = [item.get("doc_id") or item.get("flight_id") or item.get("hotel_id") or item.get("restaurant_id") or item.get("activity_id") or item.get("memory_id") for item in result["items"][:4]]
            elif "doc_id" in result:
                preview = result["doc_id"]
            else:
                preview = str(result)[:200]
        else:
            preview = str(result)[:200]
        self.tool_trace.append({"tool": name, "arguments": arguments, "preview": preview})
        if self.runner is not None:
            self.runner.trace(
                "tool_result",
                trip_id=self.episode["trip_id"],
                agent=self.role,
                tool=name,
                arguments=arguments,
                preview=preview,
                docs_seen_count=len(self.docs_seen),
                rejected_memory_count=len(self.rejected_seen),
            )

    def _track_docs(self, docs: List[Dict[str, Any]]) -> None:
        for doc in docs:
            doc_id = doc.get("doc_id")
            if doc_id and doc_id not in self.docs_seen:
                self.docs_seen.append(doc_id)
            for key in infer_context_keys_from_doc(doc):
                if key not in self.retrieved_keys_seen:
                    self.retrieved_keys_seen.append(key)

    def _track_rejected(self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            memory_id = row.get("memory_id")
            if memory_id and memory_id not in self.rejected_seen:
                self.rejected_seen.append(memory_id)
            reason_key = row.get("reason_key")
            option_id = row.get("option_id")
            if reason_key and option_id:
                note = f"{reason_key}:{option_id}"
                if note not in self.rejected_notes_seen:
                    self.rejected_notes_seen.append(note)

    def summary(self) -> Dict[str, Any]:
        return {
            "tool_trace": self.tool_trace,
            "docs_seen": self.docs_seen,
            "rejected_memory_seen": self.rejected_seen,
            "rejected_option_notes_seen": self.rejected_notes_seen,
            "retrieved_keys_seen": self.retrieved_keys_seen,
            "tool_call_count": len(self.tool_trace),
        }

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        started_at = time.perf_counter()
        if self.runner is not None:
            self.runner.trace(
                "tool_dispatch_start",
                trip_id=self.episode["trip_id"],
                agent=self.role,
                tool=name,
                arguments=arguments,
            )
        result = getattr(self, name)(**arguments)
        self._record(name=name, arguments=arguments, result=result)
        if self.runner is not None:
            self.runner.trace(
                "tool_dispatch_finish",
                trip_id=self.episode["trip_id"],
                agent=self.role,
                tool=name,
                duration_ms=round((time.perf_counter() - started_at) * 1000, 1),
                result_count=len(result.get("items", [])) if isinstance(result, dict) and isinstance(result.get("items"), list) else None,
            )
        return result

    def primitive_tool_specs(self) -> List[Dict[str, Any]]:
        return [spec for spec in self.tool_specs(primitive_only=False) if spec.get("name") in PRIMITIVE_TOOL_NAMES]

    def tool_specs(self, primitive_only: bool = False) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []

        def add(name: str, description: str, parameters: Dict[str, Any]) -> None:
            specs.append({"type": "function", "name": name, "description": description, "parameters": parameters})

        if self.role in {"single_baseline", "single_memory", "mas_memory_manager"}:
            add(
                "search_memory",
                "Search the broader memory corpus, including heuristics, stale notes, venue playbooks, and profile noise, and return only top relevant notes.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "memory_type": {"type": ["string", "null"]},
                        "include_stale": {"type": "boolean"},
                        "top_k": {"type": "integer"},
                        "scope": {
                            "type": "string",
                            "enum": ["episode", "city", "traveler", "family", "global"],
                            "description": "Search scope. episode is default; city/family/traveler/global intentionally broaden retrieval when needed.",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            )
        if self.role in {"single_memory", "mas_memory_manager", "mas_verifier"}:
            add(
                "get_rejected_options",
                "Retrieve previously rejected options and reasons so they are not reconsidered without a material change.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": ["string", "null"]},
                        "kind": {"type": ["string", "null"]},
                        "max_results": {"type": "integer"},
                        "scope": {
                            "type": "string",
                            "enum": ["episode", "city", "family", "global"],
                            "description": "Rejected-option search scope. episode is default; city/family/global intentionally broaden retrieval when needed.",
                        },
                    },
                    "additionalProperties": False,
                },
            )
        if self.role in {"single_baseline", "single_memory", "mas_memory_manager", "mas_planner"}:
            add(
                "get_profile_brief",
                "Fetch the stable traveler profile brief.",
                {
                    "type": "object",
                    "properties": {"traveler_id": {"type": "string"}},
                    "required": ["traveler_id"],
                    "additionalProperties": False,
                },
            )
            add(
                "get_venue_brief",
                "Fetch the canonical venue brief for the trip city and family.",
                {
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "family": {"type": "string"}},
                    "required": ["city", "family"],
                    "additionalProperties": False,
                },
            )
            add(
                "get_city_ops_notes",
                "Fetch current city ops notes or query a narrower city-ops subset.",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "query": {"type": ["string", "null"]},
                        "include_stale": {"type": "boolean"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
            )
        if self.role in {"single_baseline", "single_memory", "mas_planner", "mas_verifier"}:
            add(
                "get_policy",
                "Fetch current policy thresholds and caps.",
                {"type": "object", "properties": {}, "additionalProperties": False},
            )
            add(
                "search_flights",
                "Search flights incrementally with filters instead of browsing every option at once.",
                {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                        "flight_id": {"type": ["string", "null"]},
                        "max_fare": {"type": ["integer", "null"]},
                        "time_window": {"type": ["string", "null"]},
                        "red_eye_allowed": {"type": "boolean"},
                        "refundable_only": {"type": "boolean"},
                        "nonstop_only": {"type": "boolean"},
                        "exclude_ids": {"type": "array", "items": {"type": "string"}},
                        "sort_by": {"type": ["string", "null"]},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["origin", "destination"],
                    "additionalProperties": False,
                },
            )
            add(
                "search_hotels",
                "Search hotels with filters like zone, quiet score, airport access, and chain preference.",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "hotel_id": {"type": ["string", "null"]},
                        "preferred_zone": {"type": ["string", "null"]},
                        "exclude_zones": {"type": "array", "items": {"type": "string"}},
                        "exclude_ids": {"type": "array", "items": {"type": "string"}},
                        "quiet_min": {"type": ["number", "null"]},
                        "airport_access_min": {"type": ["number", "null"]},
                        "chain_ok": {"type": "boolean"},
                        "max_nightly_price": {"type": ["integer", "null"]},
                        "sort_by": {"type": ["string", "null"]},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
            )
            add(
                "search_restaurants",
                "Search restaurants with filters for quietness, dietary support, client readiness, and area.",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "restaurant_id": {"type": ["string", "null"]},
                        "preferred_area": {"type": ["string", "null"]},
                        "exclude_areas": {"type": "array", "items": {"type": "string"}},
                        "exclude_ids": {"type": "array", "items": {"type": "string"}},
                        "dietary": {"type": ["string", "null"]},
                        "quiet_min": {"type": ["number", "null"]},
                        "client_ready_min": {"type": ["number", "null"]},
                        "max_price_level": {"type": ["integer", "null"]},
                        "sort_by": {"type": ["string", "null"]},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
            )
            add(
                "search_activities",
                "Search activities with filters for weather safety, indoor preference, zone, and cost.",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "activity_id": {"type": ["string", "null"]},
                        "preferred_zone": {"type": ["string", "null"]},
                        "exclude_ids": {"type": "array", "items": {"type": "string"}},
                        "indoor_only": {"type": "boolean"},
                        "weather_safe_required": {"type": "boolean"},
                        "max_price": {"type": ["integer", "null"]},
                        "sort_by": {"type": ["string", "null"]},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
            )
        if self.role in {"single_memory", "mas_memory_manager", "mas_planner", "mas_verifier"}:
            add(
                "get_partner_promotions",
                "Inspect hotel/restaurant/activity partner bundles and discounts instead of assuming options are independent.",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "hotel_id": {"type": ["string", "null"]},
                        "restaurant_id": {"type": ["string", "null"]},
                        "activity_id": {"type": ["string", "null"]},
                        "family": {"type": ["string", "null"]},
                        "query": {"type": ["string", "null"]},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
            )
            add(
                "get_event_context",
                "Inspect event disruptions, noise surges, or weather-linked operational context for the city.",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "query": {"type": ["string", "null"]},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
            )
            add(
                "get_loyalty_profile",
                "Inspect traveler loyalty perks that may change effective bundle value.",
                {
                    "type": "object",
                    "properties": {"traveler_id": {"type": "string"}},
                    "required": ["traveler_id"],
                    "additionalProperties": False,
                },
            )
            add(
                "get_stakeholder_brief",
                "Inspect a stakeholder brief when dinner, hosting, or teammate constraints matter.",
                {
                    "type": "object",
                    "properties": {"stakeholder_id": {"type": "string"}},
                    "required": ["stakeholder_id"],
                    "additionalProperties": False,
                },
            )
            add(
                "get_booking_constraints",
                "Inspect booking constraints like badge gates, refund policy, late-arrival bundle invalidation, and reservation windows.",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": ["string", "null"]},
                        "family": {"type": ["string", "null"]},
                        "query": {"type": ["string", "null"]},
                        "max_results": {"type": "integer"},
                    },
                    "additionalProperties": False,
                },
            )
            add(
                "get_option_dependencies",
                "Inspect cross-option dependencies, such as when one hotel unlocks or weakens a dinner or transfer option.",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "query": {"type": ["string", "null"]},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
            )
        if primitive_only:
            primitive_names = set(PRIMITIVE_TOOL_NAMES)
            return [spec for spec in specs if spec.get("name") in primitive_names]
        return specs

    def _normalize_tool_scope(self, scope: str | None) -> str:
        cleaned = (scope or "episode").strip().lower().replace("-", "_")
        aliases = {
            "trip": "episode",
            "current": "episode",
            "episode_only": "episode",
            "same_city": "city",
            "city_only": "city",
            "same_traveler": "traveler",
            "traveler_only": "traveler",
            "traveller": "traveler",
            "traveller_only": "traveler",
            "same_family": "family",
            "family_only": "family",
            "all": "global",
            "any": "global",
        }
        cleaned = aliases.get(cleaned, cleaned)
        allowed = {"episode", "city", "traveler", "family", "global"}
        if cleaned not in allowed:
            raise ValueError(f"Unsupported tool scope {scope!r}; expected one of {sorted(allowed)}")
        return cleaned

    def _memory_scope_filters(self, scope: str | None) -> Dict[str, str | None]:
        normalized = self._normalize_tool_scope(scope)
        if normalized == "episode":
            return {
                "city": self.episode.get("city"),
                "traveler_id": self.episode.get("traveler_id"),
                "family": self.episode.get("family"),
            }
        if normalized == "city":
            return {"city": self.episode.get("city"), "traveler_id": None, "family": None}
        if normalized == "traveler":
            return {"city": None, "traveler_id": self.episode.get("traveler_id"), "family": None}
        if normalized == "family":
            return {"city": None, "traveler_id": None, "family": self.episode.get("family")}
        return {"city": None, "traveler_id": None, "family": None}

    def _rejected_scope_filters(self, scope: str | None) -> Dict[str, str | None]:
        normalized = self._normalize_tool_scope(scope)
        if normalized == "episode":
            return {"city": self.episode.get("city"), "family": self.episode.get("family")}
        if normalized == "city":
            return {"city": self.episode.get("city"), "family": None}
        if normalized == "family":
            return {"city": None, "family": self.episode.get("family")}
        # Rejected-option rows are not traveler-scoped, so traveler/global both mean broad search.
        return {"city": None, "family": None}

    def search_memory(
        self,
        query: str,
        memory_type: str | None = None,
        include_stale: bool = True,
        top_k: int = 5,
        scope: str = "episode",
    ) -> Dict[str, Any]:
        scope_filters = self._memory_scope_filters(scope)
        payload = self.toolbox.memory_corpus.search(
            runner=self.runner,
            query=query,
            strategy=self.retrieval_strategy,
            embedding_model=self.embedding_model,
            top_k=min(top_k, self.max_results + 1),
            memory_type=memory_type,
            city=scope_filters["city"],
            traveler_id=scope_filters["traveler_id"],
            family=scope_filters["family"],
            include_stale=include_stale,
        )
        self.usage = self.runner.combine_usages(self.usage, payload["usage"])
        self._track_docs(payload["results"])
        return {"strategy": payload["strategy"], "scope": self._normalize_tool_scope(scope), "items": payload["results"]}

    def get_rejected_options(
        self,
        query: str | None = None,
        kind: str | None = None,
        max_results: int = 5,
        scope: str = "episode",
    ) -> Dict[str, Any]:
        normalized_scope = self._normalize_tool_scope(scope)
        scope_filters = self._rejected_scope_filters(normalized_scope)
        query_text = query or f"{self.episode['city']} {self.episode['family']} rejected options"
        query_lower = query_text.lower()
        hinted_kinds = set()
        if any(token in query_lower for token in ["hotel", "room", "noise"]):
            hinted_kinds.add("hotel")
        if any(token in query_lower for token in ["restaurant", "dinner", "vibe", "vegan", "client"]):
            hinted_kinds.add("restaurant")
        if any(token in query_lower for token in ["flight", "red-eye", "red eye", "layover", "departure"]):
            hinted_kinds.add("flight")
        allowed_kinds = set([kind]) if kind else set()
        if len(hinted_kinds) >= 2:
            allowed_kinds |= hinted_kinds
        rows = []
        for row in self.toolbox.rejected_options:
            if scope_filters["city"] is not None and row["city"] != scope_filters["city"]:
                continue
            if scope_filters["family"] is not None and row["family"] != scope_filters["family"]:
                continue
            if allowed_kinds and row["kind"] not in allowed_kinds:
                continue
            item = dict(row)
            item["score"] = lexical_score(query_text, f"{row['option_id']} {row['reason_key']} {row['text']}")
            rows.append(item)
        rows.sort(key=lambda row: row["score"], reverse=True)
        rows = rows[: min(max_results, self.max_results + 1)]
        self._track_rejected(rows)
        return {"scope": normalized_scope, "items": rows}

    def get_profile_brief(self, traveler_id: str) -> Dict[str, Any]:
        item = dict(self.toolbox.env.get_profile_brief(traveler_id))
        item["doc_id"] = item.get("doc_id", f"profile:{traveler_id}")
        self._track_docs([item])
        return item

    def get_venue_brief(self, city: str, family: str) -> Dict[str, Any]:
        item = dict(self.toolbox.env.get_venue_brief(city, family))
        self._track_docs([item])
        return item

    def get_city_ops_notes(
        self,
        city: str,
        query: str | None = None,
        include_stale: bool = False,
        max_results: int = 3,
    ) -> Dict[str, Any]:
        if not query:
            item = dict(self.toolbox.env.get_city_ops_notes(city))
            self._track_docs([item])
            return {"items": [item]}
        payload = self.toolbox.memory_corpus.search(
            runner=self.runner,
            query=query,
            strategy=self.retrieval_strategy,
            embedding_model=self.embedding_model,
            top_k=min(max_results, self.max_results),
            memory_type=None,
            city=city,
            traveler_id=None,
            family=None,
            include_stale=include_stale,
        )
        rows = [row for row in payload["results"] if row.get("memory_type") in {"city_ops", "stale_city_ops"} or "city_ops" in row.get("doc_id", "")]
        self.usage = self.runner.combine_usages(self.usage, payload["usage"])
        self._track_docs(rows)
        return {"strategy": payload["strategy"], "items": rows[: min(max_results, self.max_results)]}

    def get_policy(self) -> Dict[str, Any]:
        return self.toolbox.env.get_policy("default")

    def get_partner_promotions(
        self,
        city: str,
        hotel_id: str | None = None,
        restaurant_id: str | None = None,
        activity_id: str | None = None,
        family: str | None = None,
        query: str | None = None,
        max_results: int = 5,
    ) -> Dict[str, Any]:
        rows = self.toolbox.env.get_partner_promotions(city=city, hotel_id=hotel_id, restaurant_id=restaurant_id, activity_id=activity_id, family=family)
        query_text = query or f"{self.episode['city']} partner bundle quiet private room shuttle badge"
        for row in rows:
            row["score"] = lexical_score(query_text, f"{row.get('promo_id')} {row.get('text', '')} {' '.join(row.get('benefit_tags', []))}")
        rows.sort(key=lambda row: row.get("score", 0.0), reverse=True)
        self._track_docs(rows[: min(max_results, self.max_results + 1)])
        return {"items": rows[: min(max_results, self.max_results + 1)]}

    def get_event_context(
        self,
        city: str,
        query: str | None = None,
        max_results: int = 4,
    ) -> Dict[str, Any]:
        rows = self.toolbox.env.get_event_calendar(city)
        query_text = query or f"{city} noise transfer friction event"
        for row in rows:
            row["score"] = lexical_score(query_text, f"{row.get('event_id')} {row.get('text', '')} {' '.join(row.get('tags', []))}")
        rows.sort(key=lambda row: row.get("score", 0.0), reverse=True)
        self._track_docs(rows[: min(max_results, self.max_results + 1)])
        return {"items": rows[: min(max_results, self.max_results + 1)]}

    def get_loyalty_profile(self, traveler_id: str) -> Dict[str, Any]:
        item = self.toolbox.env.get_loyalty_profile(traveler_id) or {"traveler_id": traveler_id, "doc_id": f"loyalty:{traveler_id}", "text": "No loyalty profile found.", "bonus_tags": []}
        self._track_docs([item])
        return item

    def get_stakeholder_brief(self, stakeholder_id: str) -> Dict[str, Any]:
        item = self.toolbox.env.get_stakeholder_brief(stakeholder_id) or {"stakeholder_id": stakeholder_id, "doc_id": stakeholder_id, "text": "No stakeholder brief found.", "tags": []}
        self._track_docs([item])
        return item

    def get_booking_constraints(
        self,
        city: str | None = None,
        family: str | None = None,
        query: str | None = None,
        max_results: int = 5,
    ) -> Dict[str, Any]:
        rows = self.toolbox.env.get_booking_constraints(city=city or self.episode['city'], family=family or self.episode['family'])
        query_text = query or f"{self.episode['city']} refund badge late arrival constraint"
        for row in rows:
            row["score"] = lexical_score(query_text, f"{row.get('constraint_id')} {row.get('text', '')} {' '.join(row.get('tags', []))}")
        rows.sort(key=lambda row: row.get("score", 0.0), reverse=True)
        self._track_docs(rows[: min(max_results, self.max_results + 1)])
        return {"items": rows[: min(max_results, self.max_results + 1)]}

    def get_option_dependencies(
        self,
        city: str,
        query: str | None = None,
        max_results: int = 5,
    ) -> Dict[str, Any]:
        rows = self.toolbox.env.get_option_dependencies(city)
        query_text = query or f"{city} bundle dependency shuttle private room hotel restaurant"
        for row in rows:
            row["score"] = lexical_score(query_text, f"{row.get('dependency_id')} {row.get('text', '')} {' '.join(row.get('tags', []))}")
        rows.sort(key=lambda row: row.get("score", 0.0), reverse=True)
        self._track_docs(rows[: min(max_results, self.max_results + 1)])
        return {"items": rows[: min(max_results, self.max_results + 1)]}

    def search_flights(
        self,
        origin: str,
        destination: str,
        flight_id: str | None = None,
        max_fare: int | None = None,
        time_window: str | None = None,
        red_eye_allowed: bool = True,
        refundable_only: bool = False,
        nonstop_only: bool = False,
        exclude_ids: List[str] | None = None,
        sort_by: str | None = None,
        max_results: int = 4,
    ) -> Dict[str, Any]:
        rows = self.toolbox.env.search_flights(origin, destination)
        exclude_ids = set(exclude_ids or [])
        out = []
        for row in rows:
            if flight_id and row["flight_id"] != flight_id:
                continue
            if row["flight_id"] in exclude_ids:
                continue
            if max_fare is not None and row["fare_total"] > max_fare:
                continue
            if time_window and row.get("time_window") != time_window:
                continue
            if not red_eye_allowed and row.get("red_eye"):
                continue
            if refundable_only and not row.get("refundable"):
                continue
            if nonstop_only and row.get("stops", 0) != 0:
                continue
            out.append(row)
        sort_key = {
            "fare_total": lambda row: (row["fare_total"], row.get("red_eye", False)),
            "meeting_safe": lambda row: (("meeting_safe" not in row.get("semantic_tags", [])), row["fare_total"]),
            "change_friendly": lambda row: (("change_friendly" not in row.get("semantic_tags", [])), row["fare_total"]),
        }.get(sort_by or "meeting_safe", lambda row: (("meeting_safe" not in row.get("semantic_tags", [])), row["fare_total"]))
        out.sort(key=sort_key)
        return {
            "items": [compact_item(row, ["flight_id", "time_window", "fare_total", "depart_time", "arrival_time", "duration_minutes", "refundable", "stops", "red_eye", "semantic_tags", "description_snippet"]) for row in out[: min(max_results, self.max_results)]]
        }

    def search_hotels(
        self,
        city: str,
        hotel_id: str | None = None,
        preferred_zone: str | None = None,
        exclude_zones: List[str] | None = None,
        exclude_ids: List[str] | None = None,
        quiet_min: float | None = None,
        airport_access_min: float | None = None,
        chain_ok: bool = True,
        max_nightly_price: int | None = None,
        sort_by: str | None = None,
        max_results: int = 4,
    ) -> Dict[str, Any]:
        rows = self.toolbox.env.search_hotels(city)
        exclude_zones = set(exclude_zones or [])
        exclude_ids = set(exclude_ids or [])
        out = []
        for row in rows:
            if hotel_id and row["hotel_id"] != hotel_id:
                continue
            if row["hotel_id"] in exclude_ids or row.get("zone") in exclude_zones:
                continue
            if quiet_min is not None and row.get("quiet_score", 0.0) < quiet_min:
                continue
            if airport_access_min is not None and row.get("airport_access_score", 0.0) < airport_access_min:
                continue
            if not chain_ok and row.get("chain"):
                continue
            if max_nightly_price is not None and row["nightly_price"] > max_nightly_price:
                continue
            out.append(row)
        sort_key = {
            "quiet_score": lambda row: (-row.get("quiet_score", 0.0), row["nightly_price"]),
            "airport_access": lambda row: (-row.get("airport_access_score", 0.0), row["nightly_price"]),
            "price": lambda row: (row["nightly_price"], -row.get("quiet_score", 0.0)),
            "zone_match": lambda row: (row.get("zone") != preferred_zone, row["nightly_price"]),
        }.get(sort_by or "quiet_score", lambda row: (-row.get("quiet_score", 0.0), row["nightly_price"]))
        out.sort(key=sort_key)
        return {
            "items": [compact_item(row, ["hotel_id", "nightly_price", "quiet_score", "zone", "chain", "airport_access_score", "late_checkout", "meeting_shuttle", "semantic_tags", "review_snippet"]) for row in out[: min(max_results, self.max_results)]]
        }

    def search_restaurants(
        self,
        city: str,
        restaurant_id: str | None = None,
        preferred_area: str | None = None,
        exclude_areas: List[str] | None = None,
        exclude_ids: List[str] | None = None,
        dietary: str | None = None,
        quiet_min: float | None = None,
        client_ready_min: float | None = None,
        max_price_level: int | None = None,
        sort_by: str | None = None,
        max_results: int = 4,
    ) -> Dict[str, Any]:
        rows = self.toolbox.env.search_restaurants(city)
        exclude_areas = set(exclude_areas or [])
        exclude_ids = set(exclude_ids or [])
        out = []
        for row in rows:
            if restaurant_id and row["restaurant_id"] != restaurant_id:
                continue
            if row["restaurant_id"] in exclude_ids or row.get("area") in exclude_areas:
                continue
            if dietary and dietary not in row.get("dietary_flags", []):
                continue
            if quiet_min is not None and row.get("quiet_score", 0.0) < quiet_min:
                continue
            if client_ready_min is not None and row.get("client_ready_score", 0.0) < client_ready_min:
                continue
            if max_price_level is not None and row["price_level"] > max_price_level:
                continue
            out.append(row)
        sort_key = {
            "quiet_score": lambda row: (-row.get("quiet_score", 0.0), row["price_level"]),
            "client_ready": lambda row: (-row.get("client_ready_score", 0.0), row["price_level"]),
            "area_match": lambda row: (row.get("area") != preferred_area, row["price_level"]),
            "price": lambda row: (row["price_level"], -row.get("quiet_score", 0.0)),
        }.get(sort_by or "quiet_score", lambda row: (-row.get("quiet_score", 0.0), row["price_level"]))
        out.sort(key=sort_key)
        return {
            "items": [compact_item(row, ["restaurant_id", "cuisine", "price_level", "dietary_flags", "area", "quiet_score", "client_ready_score", "private_room", "booking_cutoff", "badge_only", "semantic_tags", "review_snippet"]) for row in out[: min(max_results, self.max_results)]]
        }

    def search_activities(
        self,
        city: str,
        activity_id: str | None = None,
        preferred_zone: str | None = None,
        exclude_ids: List[str] | None = None,
        indoor_only: bool = False,
        weather_safe_required: bool = False,
        max_price: int | None = None,
        sort_by: str | None = None,
        max_results: int = 4,
    ) -> Dict[str, Any]:
        rows = self.toolbox.env.search_activities(city)
        exclude_ids = set(exclude_ids or [])
        out = []
        for row in rows:
            if activity_id and row["activity_id"] != activity_id:
                continue
            if row["activity_id"] in exclude_ids:
                continue
            if indoor_only and not row.get("indoor"):
                continue
            if weather_safe_required and "weather_safe" not in row.get("semantic_tags", []):
                continue
            if max_price is not None and row.get("price", 0) > max_price:
                continue
            out.append(row)
        sort_key = {
            "zone_match": lambda row: (row.get("location_zone") != preferred_zone, row.get("price", 0)),
            "price": lambda row: (row.get("price", 0), row.get("indoor") is False),
            "weather_safe": lambda row: (("weather_safe" not in row.get("semantic_tags", [])), row.get("price", 0)),
        }.get(sort_by or "weather_safe", lambda row: (("weather_safe" not in row.get("semantic_tags", [])), row.get("price", 0)))
        out.sort(key=sort_key)
        return {
            "items": [compact_item(row, ["activity_id", "category", "location_zone", "indoor", "price", "badge_only", "semantic_tags", "description_snippet"]) for row in out[: min(max_results, self.max_results)]]
        }
