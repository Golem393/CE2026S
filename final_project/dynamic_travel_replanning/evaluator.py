from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set


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

_CORE_CONTEXT_KEYS = {"budget_total", "weather", "meeting_zone"}

_REJECTED_REASON_ALIASES = {
    "rejected_hotel_for_noise": "rejected_hotel_for_noise",
    "noise_rejected_hotel": "rejected_hotel_for_noise",
    "rejected_flight_for_red_eye": "rejected_flight_for_red_eye",
    "red_eye_rejected_flight": "rejected_flight_for_red_eye",
    "rejected_restaurant_for_vibe": "rejected_restaurant_for_vibe",
    "wrong_vibe_restaurant": "rejected_restaurant_for_vibe",
}

_RETIRED_DOC_BY_KEY = {
    "old_budget_cap": "stale:budget_cap_archive",
    "old_local_character_priority": "stale:local_character_default",
    "avoid_chain_hotels_stable": "stale:avoid_chain_hotels_absolute",
    "old_weather_assumption": "stale:dry_weather_ops_assumption",
    "old_social_bundle_default": "stale:partner_social_default",
    "old_bundle_discount_absolute": "stale:bundle_discount_always_wins",
    "late_checkin_irrelevant": "stale:late_checkin_irrelevant",
}


_DOC_ID_TO_KEYS = {
    "heuristic:lean_context_policy": ["relevant_only"],
    "heuristic:rejected_option_memory": [],
    "heuristic:airport_access_one_off": ["prefer_airport_access"],
    "profile:traveler_consultant": ["avoid_red_eye", "prefer_quiet_hotel", "local_character_if_safe"],
    "profile:traveler_exec": ["prefer_airport_access", "prefer_quiet_hotel", "client_dinner_polished"],
    "profile:traveler_research": ["low_friction_transit"],
    "venue:OSA_business_travel": ["low_friction_transit", "prefer_quiet_hotel"],
    "venue:TPE_conference_trip": ["low_friction_transit", "prefer_airport_access", "prefer_quiet_hotel"],
    "venue:SIN_business_travel": ["low_friction_transit", "prefer_airport_access", "weather_safe_backup"],
    "city_ops:OSA": ["loud_after_10pm", "low_friction_transit"],
    "city_ops:TPE": ["low_friction_transit", "prefer_airport_access"],
    "city_ops:SIN": ["weather_safe_backup", "low_friction_transit"],
    "stale:budget_cap_archive": ["old_budget_cap"],
    "stale:avoid_chain_hotels_absolute": ["avoid_chain_hotels_stable"],
    "stale:dry_weather_ops_assumption": ["old_weather_assumption"],
    "stale:partner_social_default": ["old_social_bundle_default"],
    "stale:bundle_discount_always_wins": ["old_bundle_discount_absolute"],
    "stale:late_checkin_irrelevant": ["late_checkin_irrelevant"],
}


def _normalize_key(value: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        return ""
    return _CONTEXT_KEY_ALIASES.get(cleaned, cleaned)


def _normalized_set(values: Iterable[str]) -> Set[str]:
    return {normalized for normalized in (_normalize_key(value) for value in values) if normalized}


def _normalize_retrieved_items(
    retrieved: Iterable[str],
    *,
    docs: Iterable[str] = (),
    active_keys: Iterable[str] = (),
    spoken_hits: Dict[str, Any] | None = None,
) -> List[str]:
    normalized: List[str] = []
    seen: Set[str] = set()

    def add(value: str) -> None:
        key = _normalize_key(value)
        if key and key not in seen:
            normalized.append(key)
            seen.add(key)

    for item in retrieved:
        if (item or "").startswith(("profile:", "venue:", "city_ops:", "heuristic:", "stale:", "distractor:")):
            for mapped in _doc_keys(item):
                add(mapped)
        else:
            add(item)
    for item in active_keys:
        add(item)
    if spoken_hits:
        for bucket in ("must_remember", "forbidden", "one_off_only", "keep_context_lean"):
            for item in spoken_hits.get(bucket, []):
                add(item)
    for doc_id in docs:
        for mapped in _doc_keys(doc_id):
            add(mapped)
    return normalized


def _doc_keys(doc_id: str) -> List[str]:
    if doc_id in _DOC_ID_TO_KEYS:
        return _DOC_ID_TO_KEYS[doc_id]
    if doc_id.startswith("promo:"):
        out = ["bundle_discount_value"]
        if "private_room" in doc_id or "room" in doc_id:
            out.append("private_room_bonus")
        if "badge" in doc_id:
            out.append("conference_badge_access")
        if "shuttle" in doc_id:
            out.extend(["shuttle_bundle", "weather_safe_backup"])
        if "client" in doc_id:
            out.append("client_dinner_polished")
        return out
    if doc_id.startswith("event:"):
        out = ["transfer_friction_risk"]
        if any(token in doc_id for token in ["noise", "fair"]):
            out.append("loud_after_10pm")
        if any(token in doc_id for token in ["thunderstorm", "rain"]):
            out.append("weather_safe_backup")
        if "badge" in doc_id or "expo" in doc_id:
            out.append("conference_badge_access")
        return out
    if doc_id.startswith("loyalty:"):
        return ["loyalty_bundle_value"]
    if doc_id.startswith("stakeholder:"):
        return ["team_dietary_flex"] if "vegan" in doc_id else ["client_dinner_polished"]
    if doc_id.startswith("constraint:"):
        if "refund" in doc_id:
            return ["refundable_priority"]
        if "badge" in doc_id:
            return ["conference_badge_access"]
        if "late_arrival" in doc_id:
            return ["late_checkin_risk", "bundle_discount_value"]
        if "dietary" in doc_id:
            return ["team_dietary_flex"]
    if doc_id.startswith("dependency:"):
        out = ["bundle_discount_value"]
        if "shuttle" in doc_id:
            out.append("shuttle_bundle")
        if "badge" in doc_id:
            out.append("conference_badge_access")
        if "private_room" in doc_id:
            out.append("private_room_bonus")
        return out
    return []


def _overlap(pred: List[str], gold: List[str]) -> float:
    if not gold:
        return 1.0
    return len(set(pred) & set(gold)) / len(set(gold))


def _avoidance(pred: List[str], avoid: List[str]) -> float:
    if not avoid:
        return 1.0
    return 1.0 - (len(set(pred) & set(avoid)) / len(set(avoid)))


def _set_f1(pred: List[str], gold: List[str]) -> float:
    pred_set = _normalized_set(pred)
    gold_set = _normalized_set(gold)
    if not gold_set and not pred_set:
        return 1.0
    if not gold_set:
        return 0.0
    if not pred_set:
        return 0.0
    true_positive = len(pred_set & gold_set)
    if true_positive == 0:
        return 0.0
    precision = true_positive / len(pred_set)
    recall = true_positive / len(gold_set)
    return (2.0 * precision * recall) / (precision + recall)




def _normalize_rejected_reason(value: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        return ""
    prefix = cleaned.split(":", 1)[0]
    return _REJECTED_REASON_ALIASES.get(prefix, prefix)


def _normalize_rejected_reason_set(values: Iterable[str]) -> Set[str]:
    return {normalized for normalized in (_normalize_rejected_reason(value) for value in values) if normalized}


def _infer_retired_docs(retired: List[str], retired_docs: List[str]) -> List[str]:
    inferred = list(retired_docs)
    for key in retired:
        doc_id = _RETIRED_DOC_BY_KEY.get(_normalize_key(key))
        if doc_id and doc_id not in inferred:
            inferred.append(doc_id)
    return inferred


def _target_active_context_keys(gold: Dict[str, Any]) -> Set[str]:
    keys: Set[str] = set(gold.get("should_retrieve", []))
    spoken_gold = gold.get("required_spoken_rules", {})
    for bucket in ("must_remember", "forbidden", "one_off_only", "keep_context_lean"):
        keys.update(spoken_gold.get(bucket, []))
    return _normalized_set(keys)


def _active_context_hygiene(active_keys: List[str], active_docs: List[str], gold: Dict[str, Any]) -> float:
    raw_active = {key for key in active_keys if key}
    normalized_active = _normalized_set(active_keys)
    meaningful_active = normalized_active - _CORE_CONTEXT_KEYS

    if not meaningful_active:
        # Backward-compatible fallback for older runs that only reported core context anchors.
        relevance = len(raw_active & _CORE_CONTEXT_KEYS) / max(len(_CORE_CONTEXT_KEYS), 1)
    else:
        spoken_gold = gold.get("required_spoken_rules", {})
        essential_active = _normalized_set(
            spoken_gold.get("must_remember", [])
            + spoken_gold.get("forbidden", [])
            + spoken_gold.get("one_off_only", [])
            + spoken_gold.get("keep_context_lean", [])
        )
        relevant_pool = _target_active_context_keys(gold)
        precision = len(meaningful_active & relevant_pool) / max(len(meaningful_active), 1)
        if essential_active:
            essential_coverage = len(meaningful_active & essential_active) / len(essential_active)
        else:
            essential_coverage = precision
        relevance = 0.65 * precision + 0.35 * essential_coverage

    overload_penalty = 1.0
    if len(raw_active) > 5:
        overload_penalty *= 0.85
    if len(set(active_docs)) > 3:
        overload_penalty *= 0.8
    return max(0.0, relevance * overload_penalty)


def _id_lookup(items: List[Dict[str, Any]], key: str, value: Any) -> Optional[Dict[str, Any]]:
    for item in items:
        if item[key] == value:
            return item
    return None


def evaluate_episode(env, episode: Dict[str, Any], submission: Dict[str, Any], gold: Dict[str, Any] | None = None) -> Dict[str, Any]:
    gold = gold or episode.get("gold")
    if not gold:
        raise ValueError(f"Gold not provided for episode {episode['trip_id']}")

    city = episode["city"]
    flight = _id_lookup(env.search_flights(episode["origin"], city), "flight_id", submission.get("flight_id"))
    hotel = _id_lookup(env.search_hotels(city), "hotel_id", submission.get("hotel_id"))
    restaurant = _id_lookup(env.search_restaurants(city), "restaurant_id", submission.get("restaurant_id"))
    activity = _id_lookup(env.search_activities(city), "activity_id", submission.get("activity_id"))
    policy = env.get_policy("default")

    total_cost = 0
    hard = {key: False for key in ["under_budget", "meeting_safe_arrival", "quiet_hotel", "weather_safe_activity", "zone_coherence", "team_dietary_support", "refund_safe", "bundle_dependency_valid"]}
    scenario_state = episode.get("scenario_state", {}) or {}
    if flight:
        total_cost += flight["fare_total"]
        hard["meeting_safe_arrival"] = ("meeting_safe" in flight.get("semantic_tags", [])) or (
            "airport_access_more_important_now" in gold.get("episodic_exceptions", [])
        )
        hard["refund_safe"] = (not scenario_state.get("refund_risk")) or bool(flight.get("refundable"))
    else:
        hard["refund_safe"] = not scenario_state.get("refund_risk")
    if hotel:
        total_cost += hotel["nightly_price"] * episode["nights"]
        hard["quiet_hotel"] = "quiet" in hotel.get("semantic_tags", [])
    if restaurant:
        total_cost += restaurant["price_level"] * 25000
        hard["team_dietary_support"] = (not scenario_state.get("teammate_vegan")) or any(flag in restaurant.get("dietary_flags", []) for flag in ["vegan", "vegan_preorder"])
    else:
        hard["team_dietary_support"] = not scenario_state.get("teammate_vegan")
    if activity:
        total_cost += activity["price"]
        hard["weather_safe_activity"] = (episode.get("weather") != "rainy") or ("weather_safe" in activity.get("semantic_tags", []))

    matched_promos = env.get_partner_promotions(city=city, hotel_id=submission.get("hotel_id"), restaurant_id=submission.get("restaurant_id"), activity_id=submission.get("activity_id"), family=episode.get("family"))
    bundle_valid = True
    if scenario_state.get("partner_bundle"):
        bundle_valid = False
        arrival_minutes = None
        if flight and flight.get("arrival_time") and ":" in flight.get("arrival_time"):
            hh, mm = flight["arrival_time"].split(":", 1)
            arrival_minutes = int(hh) * 60 + int(mm)
        for promo in matched_promos:
            cutoff = promo.get("arrival_before")
            cutoff_minutes = None
            if cutoff and ":" in cutoff:
                hh, mm = cutoff.split(":", 1)
                cutoff_minutes = int(hh) * 60 + int(mm)
            badge_ok = (not promo.get("badge_required")) or scenario_state.get("badge_available")
            arrival_ok = cutoff_minutes is None or arrival_minutes is None or arrival_minutes <= cutoff_minutes
            if badge_ok and arrival_ok:
                bundle_valid = True
                break
    hard["bundle_dependency_valid"] = bundle_valid

    hard["under_budget"] = total_cost <= episode["budget_total"]
    zones = [item.get("zone") for item in [hotel] if item] + [item.get("area") for item in [restaurant] if item] + [item.get("location_zone") for item in [activity] if item]
    hard["zone_coherence"] = bool(zones) and all(zone != gold.get("avoid_zone") for zone in zones) and sum(zone == gold.get("preferred_zone") for zone in zones) >= 2
    hard_rate = sum(hard[name] for name in gold.get("required_hard", [])) / max(len(gold.get("required_hard", [])), 1)

    chosen_tags = set()
    for item in [flight, hotel, restaurant, activity]:
        if item:
            chosen_tags.update(item.get("semantic_tags", []))
    for promo in matched_promos:
        chosen_tags.update(promo.get("benefit_tags", []))
    loyalty = env.get_loyalty_profile(episode.get("traveler_id")) or {}
    if hotel and hotel.get("hotel_id") in loyalty.get("hotel_partner_ids", []):
        chosen_tags.update(loyalty.get("bonus_tags", []))
    semantic_fit = _overlap(list(chosen_tags), gold.get("soft_tags", []))
    exactish = (
        int(submission.get("flight_id") in gold.get("acceptable_flights", []))
        + int(submission.get("hotel_id") in gold.get("acceptable_hotels", []))
        + int(submission.get("restaurant_id") in gold.get("good_restaurants", []))
        + int(submission.get("activity_id") in gold.get("good_activities", []))
    ) / 4.0
    coherence = 1.0 if hard["zone_coherence"] else 0.0

    memory = submission.get("memory_report", {})
    retrieved = memory.get("retrieved", [])
    retired = memory.get("retired", [])
    retired_docs = memory.get("retired_docs", [])
    rejected = memory.get("rejected_option_notes", [])
    active_keys = memory.get("active_context_keys", [])
    docs = memory.get("docs_retrieved", [])
    active_docs = memory.get("active_docs", [])
    ignored_distractors = memory.get("ignored_distractors", [])
    spoken_hits = memory.get("spoken_rule_hits", {})

    normalized_retrieved = _normalize_retrieved_items(
        retrieved,
        docs=docs,
        active_keys=active_keys,
        spoken_hits=spoken_hits,
    )
    normalized_retired = _normalized_set(retired)

    update_handling = 1.0
    if "airport_access_more_important_now" in gold.get("episodic_exceptions", []) and "prefer_airport_access" not in normalized_retrieved:
        update_handling *= 0.75
    if "old_budget_cap" not in normalized_retired:
        update_handling *= 0.85
    if "old_local_character_priority" in gold.get("required_spoken_rules", {}).get("retire", []) and "local_character_if_safe" not in normalized_retired:
        update_handling *= 0.85
    if "old_weather_assumption" in gold.get("required_spoken_rules", {}).get("retire", []) and "old_weather_assumption" not in normalized_retired:
        update_handling *= 0.85
    if "old_social_bundle_default" in gold.get("required_spoken_rules", {}).get("retire", []) and "old_social_bundle_default" not in normalized_retired:
        update_handling *= 0.85
    if "old_bundle_discount_absolute" in gold.get("required_spoken_rules", {}).get("retire", []) and "old_bundle_discount_absolute" not in normalized_retired:
        update_handling *= 0.85
    if "late_checkin_irrelevant" in gold.get("required_spoken_rules", {}).get("retire", []) and "late_checkin_irrelevant" not in normalized_retired:
        update_handling *= 0.85

    memory_retrieval_rate = _overlap(normalized_retrieved, gold.get("should_retrieve", []))
    memory_retirement_rate = _overlap(list(normalized_retired), gold.get("should_retire", []))
    normalized_rejected = sorted(_normalize_rejected_reason_set(rejected))
    effective_retired_docs = _infer_retired_docs(retired, retired_docs)
    rejected_option_memory_rate = _overlap(normalized_rejected, gold.get("should_remember_rejected", []))
    distributed_context_rate = _overlap(docs, gold.get("required_docs", []))
    stale_doc_retirement_rate = _overlap(effective_retired_docs, gold.get("stale_docs_to_retire", []))
    distractor_avoidance_rate = _avoidance(active_docs, gold.get("distractor_docs_to_avoid", []))
    if gold.get("distractor_docs_to_avoid"):
        distractor_avoidance_rate = max(distractor_avoidance_rate, _overlap(ignored_distractors, gold.get("distractor_docs_to_avoid", [])))

    active_context_hygiene_rate = _active_context_hygiene(active_keys, active_docs, gold)

    hotel_over_cap = bool(hotel and hotel["nightly_price"] > policy["hotel_soft_cap_per_night"])
    policy_ok = 0.0 if hotel_over_cap and "slight_budget_overage_ok_this_trip" not in gold.get("episodic_exceptions", []) else 1.0

    spoken_scores = []
    spoken_gold = gold.get("required_spoken_rules", {})
    for key in ["must_remember", "forbidden", "one_off_only", "retire", "do_not_reconsider", "keep_context_lean"]:
        spoken_scores.append(_set_f1(spoken_hits.get(key, []), spoken_gold.get(key, [])))
    spoken_rule_compliance_rate = sum(spoken_scores) / max(len(spoken_scores), 1)

    decision_quality = (
        0.22 * hard_rate
        + 0.18 * semantic_fit
        + 0.16 * exactish
        + 0.12 * coherence
        + 0.18 * spoken_rule_compliance_rate
        + 0.07 * stale_doc_retirement_rate
        + 0.07 * distractor_avoidance_rate
    )

    usage = submission.get("usage", {})
    total_tokens = int(usage.get("total_tokens", usage.get("input_tokens", 0) + usage.get("output_tokens", 0)))
    cost = float(usage.get("estimated_cost_usd", 0.0))
    tool_calls = int(submission.get("debug", {}).get("tool_call_count", submission.get("tool_call_count", 0) or 0))

    return {
        "trip_id": episode["trip_id"],
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







def _summary_bucket_means(rows: List[Dict[str, Any]], fixed_baseline_cost: float = 0.03) -> Dict[str, float]:
    mean = lambda key: sum(row[key] for row in rows) / len(rows) if rows else 0.0
    total_cost = sum(row["estimated_cost_usd"] for row in rows)
    feasibility = (mean("hard_constraint_rate") + mean("policy_ok") + mean("bundle_coherence_rate")) / 3.0
    preference_fit = (mean("decision_quality") + mean("semantic_fit_rate") + mean("spoken_rule_compliance_rate")) / 3.0
    adaptation = (
        mean("update_handling_rate")
        + mean("memory_retrieval_rate")
        + mean("memory_retirement_rate")
        + mean("distributed_context_rate")
        + mean("stale_doc_retirement_rate")
        + mean("distractor_avoidance_rate")
        + mean("rejected_option_memory_rate")
        + mean("active_context_hygiene_rate")
    ) / 8.0
    efficiency = min(1.0, fixed_baseline_cost / max(total_cost, 1e-12)) if rows else 0.0
    if mean("decision_quality") < 0.35:
        efficiency *= 0.5
    return {
        "feasibility": feasibility,
        "preference_fit": preference_fit,
        "adaptation": adaptation,
        "efficiency": efficiency,
        "total_cost": total_cost,
    }



def summarize_rows_student(rows: List[Dict[str, Any]], fixed_baseline_cost: float = 0.03) -> Dict[str, Any]:
    buckets = _summary_bucket_means(rows, fixed_baseline_cost=fixed_baseline_cost)
    overall = 0.40 * buckets["feasibility"] + 0.30 * buckets["preference_fit"] + 0.20 * buckets["adaptation"] + 0.10 * buckets["efficiency"]
    return {
        "student_overall_score": round(overall * 100.0, 2),
        "feasibility_constraints": round(buckets["feasibility"] * 100.0, 2),
        "preference_fit": round(buckets["preference_fit"] * 100.0, 2),
        "adaptation_memory": round(buckets["adaptation"] * 100.0, 2),
        "efficiency_bonus": round(buckets["efficiency"] * 100.0, 2),
        "episodes": len(rows),
        "total_cost_usd": round(buckets["total_cost"], 6),
    }



def summarize_rows(rows: List[Dict[str, Any]], fixed_baseline_cost: float = 0.03) -> Dict[str, Any]:
    mean = lambda key: sum(row[key] for row in rows) / len(rows) if rows else 0.0
    buckets = _summary_bucket_means(rows, fixed_baseline_cost=fixed_baseline_cost)
    raw = (
        15.0 * buckets["feasibility"]
        + 14.0 * buckets["preference_fit"]
        + 11.0 * buckets["adaptation"]
        + 5.0 * buckets["efficiency"]
    )

    by_tier = {}
    for tier in sorted(set(row["difficulty_tier"] for row in rows)):
        tier_rows = [row for row in rows if row["difficulty_tier"] == tier]
        by_tier[tier] = round(sum(row["decision_quality"] for row in tier_rows) / len(tier_rows), 4)

    return {
        "student_view": summarize_rows_student(rows, fixed_baseline_cost=fixed_baseline_cost),
        "episodes": len(rows),
        "mean_decision_quality": round(mean("decision_quality"), 4),
        "mean_hard_constraint_rate": round(mean("hard_constraint_rate"), 4),
        "mean_semantic_fit_rate": round(mean("semantic_fit_rate"), 4),
        "mean_bundle_coherence_rate": round(mean("bundle_coherence_rate"), 4),
        "mean_update_handling_rate": round(mean("update_handling_rate"), 4),
        "mean_memory_retrieval_rate": round(mean("memory_retrieval_rate"), 4),
        "mean_memory_retirement_rate": round(mean("memory_retirement_rate"), 4),
        "mean_distributed_context_rate": round(mean("distributed_context_rate"), 4),
        "mean_stale_doc_retirement_rate": round(mean("stale_doc_retirement_rate"), 4),
        "mean_distractor_avoidance_rate": round(mean("distractor_avoidance_rate"), 4),
        "mean_rejected_option_memory_rate": round(mean("rejected_option_memory_rate"), 4),
        "mean_active_context_hygiene_rate": round(mean("active_context_hygiene_rate"), 4),
        "mean_spoken_rule_compliance_rate": round(mean("spoken_rule_compliance_rate"), 4),
        "mean_policy_ok": round(mean("policy_ok"), 4),
        "mean_tool_calls": round(mean("tool_calls"), 2),
        "by_tier_decision_quality": by_tier,
        "total_tokens": int(sum(row["tokens"] for row in rows)),
        "total_cost_usd": round(buckets["total_cost"], 6),
        "feasibility_bucket": round(buckets["feasibility"], 4),
        "preference_fit_bucket": round(buckets["preference_fit"], 4),
        "adaptation_bucket": round(buckets["adaptation"], 4),
        "efficiency_bucket": round(buckets["efficiency"], 4),
        "raw_score_for_ranking": round(raw, 4),
    }
