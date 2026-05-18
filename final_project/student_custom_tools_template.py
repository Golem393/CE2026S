from __future__ import annotations

"""Student-owned helper module.

This file contains deterministic Python helpers that run in the orchestrator
(not as LLM-callable tools). They save LLM tokens by doing scoring, filtering,
budget checking, and constraint validation locally.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
#  ZONE ADJACENCY MAP — zones that are "close" to each other per city
# ═══════════════════════════════════════════════════════════════════════

_ZONE_ADJACENCY: Dict[str, Dict[str, List[str]]] = {
    "OSA": {
        "namba": ["namba", "umeda", "shinsekai"],
        "umeda": ["umeda", "namba", "airport_link"],
        "shinsekai": ["shinsekai", "namba"],
        "airport_link": ["airport_link", "umeda"],
    },
    "TPE": {
        "xinyi": ["xinyi", "songshan", "blue_line_corridor"],
        "songshan": ["songshan", "xinyi", "blue_line_corridor"],
        "ximending": ["ximending", "songshan"],
        "scenic_outer": ["scenic_outer"],
        "blue_line_corridor": ["blue_line_corridor", "xinyi", "songshan"],
    },
    "SIN": {
        "clarke_quay": ["clarke_quay", "chinatown", "bugis", "marina"],
        "chinatown": ["chinatown", "clarke_quay"],
        "bugis": ["bugis", "clarke_quay", "marina"],
        "one_north": ["one_north", "airport_link"],
        "marina": ["marina", "clarke_quay", "bugis"],
        "airport_link": ["airport_link", "one_north"],
        "jurong": ["jurong"],
    },
}

# Spoken-rule keyword → canonical constraint key
_RULE_KEYWORD_MAP: Dict[str, str] = {
    "red eye": "avoid_red_eye",
    "red-eye": "avoid_red_eye",
    "no red eye": "avoid_red_eye",
    "quiet": "prefer_quiet_hotel",
    "quiet hotel": "prefer_quiet_hotel",
    "quiet room": "prefer_quiet_hotel",
    "noise": "loud_after_10pm",
    "loud": "loud_after_10pm",
    "after 10pm": "loud_after_10pm",
    "nightlife": "loud_after_10pm",
    "client dinner": "client_dinner_polished",
    "polished dinner": "client_dinner_polished",
    "client ready": "client_dinner_polished",
    "airport access": "prefer_airport_access",
    "airport": "prefer_airport_access",
    "weather": "weather_safe_backup",
    "rainy": "weather_safe_backup",
    "rain": "weather_safe_backup",
    "indoor": "weather_safe_backup",
    "vegan": "team_dietary_flex",
    "dietary": "team_dietary_flex",
    "refund": "refundable_priority",
    "refundable": "refundable_priority",
    "badge": "conference_badge_access",
    "conference badge": "conference_badge_access",
    "chain": "chain_ok_this_trip",
    "chain hotel": "chain_ok_this_trip",
    "bundle": "bundle_discount_value",
    "discount": "bundle_discount_value",
    "loyalty": "loyalty_bundle_value",
    "late check": "late_checkin_risk",
    "late arrival": "late_checkin_risk",
    "shuttle": "shuttle_bundle",
    "low friction": "low_friction_transit",
    "transit": "low_friction_transit",
    "transfer": "transfer_friction_risk",
    "budget": "budget_cap",
    "nonstop": "prefer_nonstop",
    "direct flight": "prefer_nonstop",
}


# ═══════════════════════════════════════════════════════════════════════
#  BUDGET & COST HELPERS
# ═══════════════════════════════════════════════════════════════════════

def pre_filter_by_budget(
    candidates: List[Dict[str, Any]],
    max_price: float,
    price_key: str = "fare_total",
) -> List[Dict[str, Any]]:
    """Drop candidates exceeding max_price on the given price_key.

    Works for flights (fare_total), hotels (nightly_price),
    restaurants (price_level), activities (price).
    Returns filtered list, preserving original order.
    """
    return [c for c in candidates if c.get(price_key, float("inf")) <= max_price]


def calculate_total_itinerary_cost(
    itinerary: Dict[str, Any],
    nights: int = 1,
) -> float:
    """Sum the total cost of a current itinerary.

    Args:
        itinerary: Dict with flight, hotel, restaurant, activity sub-dicts.
        nights: Number of nights (hotel cost = nightly_price × nights).

    Returns:
        Total estimated cost. Uses price_level × 50000 as restaurant estimate
        when exact price is unavailable.
    """
    total = 0.0

    flight = itinerary.get("flight") or {}
    total += float(flight.get("fare_total", 0))

    hotel = itinerary.get("hotel") or {}
    total += float(hotel.get("nightly_price", 0)) * max(nights, 1)

    restaurant = itinerary.get("restaurant") or {}
    # price_level is 1–4; estimate ~50000 per level as a rough proxy
    total += float(restaurant.get("price_level", 0)) * 50000

    activity = itinerary.get("activity") or {}
    total += float(activity.get("price", 0))

    return total


def get_remaining_budget(
    current_itinerary: Dict[str, Any],
    total_budget: float,
    nights: int = 1,
) -> float:
    """Returns total_budget minus the estimated itinerary cost."""
    spent = calculate_total_itinerary_cost(current_itinerary, nights)
    return total_budget - spent


def calculate_days_nights(
    arrival_date: str,
    departure_date: str,
) -> Dict[str, int]:
    """Parse ISO dates and return {'days': N, 'nights': N-1}.

    Handles common date formats: YYYY-MM-DD, YYYY/MM/DD.
    """
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            arr = datetime.strptime(arrival_date, fmt)
            dep = datetime.strptime(departure_date, fmt)
            delta = (dep - arr).days
            return {"days": max(delta, 0) + 1, "nights": max(delta, 0)}
        except ValueError:
            continue
    return {"days": 0, "nights": 0}


# ═══════════════════════════════════════════════════════════════════════
#  RERANKING & SCORING
# ═══════════════════════════════════════════════════════════════════════

def _score_hotel(hotel: Dict[str, Any], ctx: Dict[str, Any]) -> float:
    """Score a single hotel candidate. Higher = better."""
    score = 0.0
    quiet_weight = ctx.get("quiet_weight", 3.0)
    airport_weight = ctx.get("airport_weight", 1.5)
    price_weight = ctx.get("price_weight", 1.0)
    zone_bonus = ctx.get("zone_bonus", 2.0)

    score += float(hotel.get("quiet_score", 0)) * quiet_weight
    score += float(hotel.get("airport_access_score", 0)) * airport_weight

    # Price penalty: cheaper is better, normalize around 200k
    nightly = float(hotel.get("nightly_price", 250000))
    score -= (nightly / 100000) * price_weight

    # Zone match bonus
    target_zone = ctx.get("target_zone", "")
    if target_zone and hotel.get("zone") == target_zone:
        score += zone_bonus

    # Chain penalty if not allowed
    if not ctx.get("chain_ok", True) and hotel.get("chain"):
        score -= 5.0

    # Rejected penalty
    rejected_ids = set(ctx.get("rejected_ids", []))
    if hotel.get("hotel_id") in rejected_ids:
        score -= 100.0

    # Semantic tag bonuses
    tags = set(hotel.get("semantic_tags", []))
    if "meeting_shuttle" in tags or hotel.get("meeting_shuttle"):
        score += 1.0
    if "late_checkout" in tags or hotel.get("late_checkout"):
        score += 0.5

    return score


def rerank_hotels(
    candidates: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Score and sort hotel candidates by weighted preference fit.

    Context keys:
        quiet_weight (float): Weight for quiet_score (default 3.0)
        airport_weight (float): Weight for airport_access_score (default 1.5)
        price_weight (float): Weight for price penalty (default 1.0)
        zone_bonus (float): Bonus for matching target_zone (default 2.0)
        target_zone (str): Preferred zone
        chain_ok (bool): Whether chain hotels are acceptable
        rejected_ids (List[str]): IDs to deprioritize
    """
    scored = [(c, _score_hotel(c, context)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored]


def _score_restaurant(rest: Dict[str, Any], ctx: Dict[str, Any]) -> float:
    """Score a single restaurant candidate."""
    score = 0.0
    quiet_weight = ctx.get("quiet_weight", 2.5)
    client_weight = ctx.get("client_weight", 2.0)
    price_weight = ctx.get("price_weight", 0.5)

    score += float(rest.get("quiet_score", 0)) * quiet_weight
    score += float(rest.get("client_ready_score", 0)) * client_weight

    # Lower price_level is generally better unless client dinner
    price_level = int(rest.get("price_level", 3))
    if ctx.get("client_dinner"):
        # For client dinners, higher price can be fine
        score += price_level * 0.5
    else:
        score -= price_level * price_weight

    # Area match bonus
    target_area = ctx.get("target_area", "")
    if target_area and rest.get("area") == target_area:
        score += ctx.get("area_bonus", 1.5)

    # Dietary match
    required_dietary = ctx.get("dietary")
    if required_dietary:
        flags = rest.get("dietary_flags", [])
        if required_dietary in flags:
            score += 3.0
        else:
            score -= 10.0  # Hard fail

    # Private room bonus
    if rest.get("private_room") and ctx.get("prefer_private_room"):
        score += 2.0

    # Rejected penalty
    rejected_ids = set(ctx.get("rejected_ids", []))
    if rest.get("restaurant_id") in rejected_ids:
        score -= 100.0

    return score


def rerank_restaurants(
    candidates: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Score and sort restaurant candidates by preference fit.

    Context keys:
        quiet_weight, client_weight, price_weight, target_area, area_bonus,
        dietary (str), client_dinner (bool), prefer_private_room (bool),
        rejected_ids (List[str])
    """
    scored = [(c, _score_restaurant(c, context)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored]


def _score_flight(flight: Dict[str, Any], ctx: Dict[str, Any]) -> float:
    """Score a single flight candidate."""
    score = 0.0

    # Lower fare = better
    fare = float(flight.get("fare_total", 999999))
    score -= (fare / 100000) * ctx.get("price_weight", 1.5)

    # Red-eye penalty
    if flight.get("red_eye") and not ctx.get("red_eye_ok", False):
        score -= 15.0

    # Stops penalty
    stops = int(flight.get("stops", 0))
    score -= stops * ctx.get("stops_penalty", 2.0)

    # Refundable bonus
    if flight.get("refundable"):
        score += ctx.get("refundable_bonus", 2.0)

    # Meeting-safe tag bonus
    tags = set(flight.get("semantic_tags", []))
    if "meeting_safe" in tags:
        score += ctx.get("meeting_safe_bonus", 3.0)
    if "change_friendly" in tags:
        score += 1.0

    # Rejected penalty
    rejected_ids = set(ctx.get("rejected_ids", []))
    if flight.get("flight_id") in rejected_ids:
        score -= 100.0

    return score


def rerank_flights(
    candidates: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Score and sort flight candidates.

    Context keys:
        price_weight (float), red_eye_ok (bool), stops_penalty (float),
        refundable_bonus (float), meeting_safe_bonus (float),
        rejected_ids (List[str])
    """
    scored = [(c, _score_flight(c, context)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored]


def _score_activity(act: Dict[str, Any], ctx: Dict[str, Any]) -> float:
    """Score a single activity candidate."""
    score = 0.0

    # Weather-safe bonus
    tags = set(act.get("semantic_tags", []))
    if "weather_safe" in tags:
        score += ctx.get("weather_safe_bonus", 3.0)

    # Indoor bonus when weather is bad
    if act.get("indoor"):
        if ctx.get("weather_risky", False):
            score += 4.0
        else:
            score += 1.0

    # Zone match
    target_zone = ctx.get("target_zone", "")
    if target_zone and act.get("location_zone") == target_zone:
        score += ctx.get("zone_bonus", 2.0)

    # Lower price = better
    price = float(act.get("price", 0))
    score -= (price / 50000) * ctx.get("price_weight", 0.5)

    # Rejected penalty
    rejected_ids = set(ctx.get("rejected_ids", []))
    if act.get("activity_id") in rejected_ids:
        score -= 100.0

    return score


def rerank_activities(
    candidates: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Score and sort activity candidates.

    Context keys:
        weather_safe_bonus (float), weather_risky (bool),
        target_zone (str), zone_bonus (float),
        price_weight (float), rejected_ids (List[str])
    """
    scored = [(c, _score_activity(c, context)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored]


# ═══════════════════════════════════════════════════════════════════════
#  BUNDLE & COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════

def choose_bundle(
    bundle_candidates: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Score each candidate bundle and pick the best.

    Each bundle is a dict with keys like flight, hotel, restaurant, activity.
    Scoring considers total cost, constraint satisfaction, zone proximity,
    and preference fit.
    """
    if not bundle_candidates:
        return None

    best_score = float("-inf")
    best_bundle = bundle_candidates[0]

    budget = float(context.get("budget_total", float("inf")))
    nights = int(context.get("nights", 1))
    target_zone = context.get("target_zone", "")
    city = context.get("city", "")

    for bundle in bundle_candidates:
        score = 0.0

        # Cost: under budget = good, over = bad
        cost = calculate_total_itinerary_cost(bundle, nights)
        if cost <= budget:
            # Reward being under budget but not too cheap (quality matters)
            utilization = cost / budget if budget > 0 else 0
            score += utilization * 5.0  # Reward using 60-90% of budget
        else:
            score -= 20.0  # Over budget is very bad

        # Zone coherence: how many items are in/near the target zone
        zone_items = 0
        for key, zone_key in [
            ("hotel", "zone"),
            ("restaurant", "area"),
            ("activity", "location_zone"),
        ]:
            item = bundle.get(key) or {}
            item_zone = item.get(zone_key, "")
            if check_location_compatibility(item_zone, target_zone, city):
                zone_items += 1
        score += zone_items * 2.0

        # Quality scores
        hotel = bundle.get("hotel") or {}
        score += float(hotel.get("quiet_score", 0)) * 1.5
        score += float(hotel.get("airport_access_score", 0)) * 0.5

        restaurant = bundle.get("restaurant") or {}
        score += float(restaurant.get("quiet_score", 0)) * 1.0
        score += float(restaurant.get("client_ready_score", 0)) * 1.0

        flight = bundle.get("flight") or {}
        if flight.get("red_eye") and not context.get("red_eye_ok", False):
            score -= 10.0
        if flight.get("refundable"):
            score += 1.0

        if score > best_score:
            best_score = score
            best_bundle = bundle

    return best_bundle


def find_cheapest_compliant_combo(
    flights: List[Dict[str, Any]],
    hotels: List[Dict[str, Any]],
    constraints: List[str],
) -> Dict[str, Any]:
    """Find the cheapest flight+hotel combo satisfying constraint list.

    Brute-force over ≤4×4 candidates. Constraints are canonical key strings.
    Returns {"flight": ..., "hotel": ..., "total_cost": ...} or empty dict.
    """
    best: Dict[str, Any] = {}
    best_cost = float("inf")

    # Limit to top 4 each for performance
    fl = flights[:4]
    hl = hotels[:4]

    for f in fl:
        for h in hl:
            # Check constraints
            violations = _check_combo_constraints(f, h, constraints)
            if violations:
                continue

            cost = float(f.get("fare_total", 0)) + float(h.get("nightly_price", 0))
            if cost < best_cost:
                best_cost = cost
                best = {"flight": f, "hotel": h, "total_cost": cost}

    return best


def _check_combo_constraints(
    flight: Dict[str, Any],
    hotel: Dict[str, Any],
    constraints: List[str],
) -> List[str]:
    """Check a flight+hotel combo against constraint keys."""
    violations = []
    for c in constraints:
        if c == "avoid_red_eye" and flight.get("red_eye"):
            violations.append("flight is red-eye")
        if c == "prefer_quiet_hotel" and float(hotel.get("quiet_score", 0)) < 7.0:
            violations.append(f"hotel quiet_score {hotel.get('quiet_score')} < 7.0")
        if c == "prefer_airport_access" and float(hotel.get("airport_access_score", 0)) < 7.5:
            violations.append(f"hotel airport_access {hotel.get('airport_access_score')} < 7.5")
        if c == "refundable_priority" and not flight.get("refundable"):
            violations.append("flight not refundable")
    return violations


def find_closest_hotels(
    candidates: List[Dict[str, Any]],
    target_zone: str,
) -> List[Dict[str, Any]]:
    """Filter and sort hotels by proximity to target_zone.

    Exact zone match first, then adjacent zones, then everything else.
    """
    exact = []
    adjacent = []
    other = []

    # Build adjacency from all cities
    all_adjacent: set = set()
    for city_zones in _ZONE_ADJACENCY.values():
        if target_zone in city_zones:
            all_adjacent = set(city_zones[target_zone])
            break

    for h in candidates:
        zone = h.get("zone", "")
        if zone == target_zone:
            exact.append(h)
        elif zone in all_adjacent:
            adjacent.append(h)
        else:
            other.append(h)

    return exact + adjacent + other


def check_location_compatibility(
    item_location: str,
    target_zone: str,
    city: str,
) -> bool:
    """Check if item_location matches or is adjacent to target_zone in city."""
    if not item_location or not target_zone:
        return False
    if item_location == target_zone:
        return True

    city_zones = _ZONE_ADJACENCY.get(city, {})
    adjacent = city_zones.get(target_zone, [])
    return item_location in adjacent


# ═══════════════════════════════════════════════════════════════════════
#  SCHEDULE & CONFLICT
# ═══════════════════════════════════════════════════════════════════════

def check_schedule_conflicts(
    flight: Dict[str, Any],
    activity: Dict[str, Any],
) -> List[str]:
    """Check if a flight's arrival conflicts with an activity.

    Returns a list of conflict reasons, empty if no conflicts.
    """
    conflicts = []

    arrival_str = flight.get("arrival_time", "")
    if not arrival_str:
        return conflicts

    try:
        arrival_hour = int(arrival_str.split(":")[0])
    except (ValueError, IndexError):
        return conflicts

    # If flight arrives late (after 20:00), evening activities may be missed
    if arrival_hour >= 20:
        conflicts.append(
            f"flight arrives at {arrival_str}, too late for same-day activity"
        )

    # If flight is red-eye, next-morning activities may be impacted
    if flight.get("red_eye"):
        conflicts.append("red-eye flight may impact next-day activity readiness")

    # Long flights (>4 hours) arriving after 16:00 cause fatigue
    duration = int(flight.get("duration_minutes", 0))
    if duration > 240 and arrival_hour >= 16:
        conflicts.append(
            f"long flight ({duration}min) arriving at {arrival_str} — fatigue risk"
        )

    return conflicts


# ═══════════════════════════════════════════════════════════════════════
#  CONSTRAINT EXTRACTION & RULE CHECKING
# ═══════════════════════════════════════════════════════════════════════

def extract_hard_constraints_from_rules(
    spoken_rules: List[str],
) -> List[str]:
    """Parse spoken-rule strings into canonical constraint keys.

    Scans each rule for known keywords and maps them to benchmark keys
    like 'avoid_red_eye', 'prefer_quiet_hotel', etc.

    Returns deduplicated list of canonical constraint keys.
    """
    found: List[str] = []
    seen: set = set()

    for rule in spoken_rules:
        rule_lower = rule.lower()
        for keyword, canonical in _RULE_KEYWORD_MAP.items():
            if keyword in rule_lower and canonical not in seen:
                found.append(canonical)
                seen.add(canonical)

    return found


def automated_rule_checker(
    itinerary: Dict[str, Any],
    hard_constraints: List[str],
) -> List[str]:
    """Check an itinerary dict against hard constraints.

    Returns a list of violation descriptions. Empty = all good.
    Runs BEFORE the Verifier agent to catch obvious issues early and
    save LLM round-trips.
    """
    violations = []
    flight = itinerary.get("flight") or {}
    hotel = itinerary.get("hotel") or {}
    restaurant = itinerary.get("restaurant") or {}
    activity = itinerary.get("activity") or {}

    for constraint in hard_constraints:
        if constraint == "avoid_red_eye":
            if flight.get("red_eye"):
                violations.append(
                    f"VIOLATION: Flight {flight.get('flight_id')} is a red-eye "
                    f"but avoid_red_eye is a hard constraint"
                )

        elif constraint == "prefer_quiet_hotel":
            qs = float(hotel.get("quiet_score", 0))
            if qs < 7.0 and hotel.get("hotel_id"):
                violations.append(
                    f"VIOLATION: Hotel {hotel.get('hotel_id')} quiet_score={qs} "
                    f"is below threshold (7.0)"
                )

        elif constraint == "loud_after_10pm":
            qs = float(hotel.get("quiet_score", 0))
            if qs < 6.0 and hotel.get("hotel_id"):
                violations.append(
                    f"VIOLATION: Hotel {hotel.get('hotel_id')} quiet_score={qs} "
                    f"— noise risk after 10pm"
                )

        elif constraint == "prefer_airport_access":
            aa = float(hotel.get("airport_access_score", 0))
            if aa < 7.5 and hotel.get("hotel_id"):
                violations.append(
                    f"VIOLATION: Hotel {hotel.get('hotel_id')} airport_access={aa} "
                    f"is below threshold (7.5)"
                )

        elif constraint == "client_dinner_polished":
            cr = float(restaurant.get("client_ready_score", 0))
            if cr < 7.0 and restaurant.get("restaurant_id"):
                violations.append(
                    f"VIOLATION: Restaurant {restaurant.get('restaurant_id')} "
                    f"client_ready={cr} — not polished enough"
                )

        elif constraint == "team_dietary_flex":
            flags = restaurant.get("dietary_flags", [])
            if restaurant.get("restaurant_id") and "vegan" not in flags:
                violations.append(
                    f"WARNING: Restaurant {restaurant.get('restaurant_id')} "
                    f"may not support vegan dietary needs"
                )

        elif constraint == "refundable_priority":
            if not flight.get("refundable") and flight.get("flight_id"):
                violations.append(
                    f"VIOLATION: Flight {flight.get('flight_id')} is not refundable "
                    f"but refundable_priority is set"
                )

        elif constraint == "weather_safe_backup":
            act_tags = set(activity.get("semantic_tags", []))
            if (
                activity.get("activity_id")
                and "weather_safe" not in act_tags
                and not activity.get("indoor")
            ):
                violations.append(
                    f"WARNING: Activity {activity.get('activity_id')} "
                    f"is outdoor and not weather-safe"
                )

    return violations


def summarize_failed_searches(
    failed_list: List[str],
) -> List[str]:
    """Deduplicate and compress failed search notes.

    Merges similar entries, caps at 6 items to keep context lean.
    """
    seen: set = set()
    deduped: List[str] = []
    for item in failed_list:
        # Normalize whitespace
        normalized = " ".join(item.strip().split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)

    # Cap at 6 to keep the memory board lean
    return deduped[:6]
