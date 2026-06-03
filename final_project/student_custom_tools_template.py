from __future__ import annotations

"""Student-owned helper module.

This file contains deterministic Python helpers that run in the orchestrator
(not as LLM-callable tools). They save LLM tokens by doing scoring, filtering,
budget checking, and constraint validation locally.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple



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
        Total estimated cost, matching the evaluator's cost formula exactly
        (restaurant = price_level × 25000, evaluator.py:541).
    """
    total = 0.0

    flight = itinerary.get("flight") or {}
    total += float(flight.get("fare_total", 0))

    hotel = itinerary.get("hotel") or {}
    total += float(hotel.get("nightly_price", 0)) * max(nights, 1)

    restaurant = itinerary.get("restaurant") or {}
    total += float(restaurant.get("price_level", 0)) * 25000

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


def automated_rule_checker(
    itinerary: Dict[str, Any],
    hard_constraints: List[str],
    budget_total: float = 0.0,
    nights: int = 1,
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

    if flight and activity:
        conflicts = check_schedule_conflicts(flight, activity)
        violations.extend(conflicts)

    # Budget is a hard constraint (evaluator under_budget). Flag any overage so the
    # revision loop swaps for cheaper options.
    if budget_total:
        total_cost = calculate_total_itinerary_cost(itinerary, nights)
        if total_cost > budget_total:
            violations.append(
                f"VIOLATION: total cost {int(total_cost)} exceeds budget "
                f"{int(budget_total)} by {int(total_cost - budget_total)} — "
                f"swap for cheaper flight/hotel that still meets tags"
            )

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


# ═══════════════════════════════════════════════════════════════════════
#  DETERMINISTIC FEASIBLE-ITINERARY SELECTOR
# ═══════════════════════════════════════════════════════════════════════

def select_feasible_itinerary(
    env: Any,
    episode: Dict[str, Any],
    *,
    require_refundable: bool = False,
    require_vegan: bool = False,
    require_client_dinner: bool = False,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Pick the cheapest itinerary that satisfies the hard constraints.

    Hidden-safe: relies only on observable signals (meeting_zone, weather,
    item semantic_tags). zone_coherence (>=2/3 venues in meeting_zone) is met by
    placing restaurant + activity in meeting_zone; the hotel also targets
    meeting_zone when a quiet option exists there, maximising zone coherence.
    Avoids 'nightlife_strip' zones (the env's proxy for the gold avoid_zone).

    After initial selection, a bundle-aware pass checks partner promotions and
    swaps items if a valid bundle is feasible under budget.

    Conditional needs (refundable flight, vegan restaurant) are passed in by the
    caller, inferred from the user turns by the Memory agent. Weather-safe
    activities are required when episode weather is rainy.

    Returns {"flight": .., "hotel": .., "restaurant": .., "activity": ..} with
    full item dicts (or None when a category is empty).
    """
    city = episode["city"]
    mz = episode.get("meeting_zone")
    nights = max(int(episode.get("nights", 1)), 1)
    budget = float(episode.get("budget_total", 0) or 0)
    weather = episode.get("weather")
    family = episode.get("family", "")

    flights = env.search_flights(episode["origin"], city)
    hotels = env.search_hotels(city)
    rests = env.search_restaurants(city)
    acts = env.search_activities(city)

    def cheapest(cands, fallback, key):
        if cands:
            return min(cands, key=key)
        return min(fallback, key=key) if fallback else None

    # Flight: meeting_safe (morning). Prefer refundable when required.
    f_ms = [f for f in flights if "meeting_safe" in f.get("semantic_tags", [])]
    f_ref = [f for f in f_ms if f.get("refundable")]
    flight_cheap = cheapest(f_ms, flights, lambda f: f["fare_total"])
    flight_pref = cheapest(f_ref, f_ms or flights, lambda f: f["fare_total"]) if require_refundable else flight_cheap

    # Hotel: quiet, not nightlife_strip. Prefer meeting_zone for zone coherence.
    h_quiet = [
        h for h in hotels
        if "quiet" in h.get("semantic_tags", []) and "nightlife_strip" not in h.get("semantic_tags", [])
    ]
    # Try zone-matched quiet hotel first for 3/3 zone coherence
    h_quiet_zone = [h for h in h_quiet if h.get("zone") == mz] if mz else []
    hotel = cheapest(
        h_quiet_zone,
        h_quiet or [h for h in hotels if "quiet" in h.get("semantic_tags", [])] or hotels,
        lambda h: h["nightly_price"],
    )

    # Restaurant in meeting_zone. Prefer vegan when required, client-ready when needed.
    r_zone = [r for r in rests if r.get("area") == mz]
    r_vegan = [r for r in r_zone if any(x in r.get("dietary_flags", []) for x in ["vegan", "vegan_preorder"])]
    r_client = [r for r in r_zone if float(r.get("client_ready_score", 0)) >= 7.5]
    rest_cheap = cheapest(r_zone, rests, lambda r: r["price_level"])
    if require_vegan and require_client_dinner:
        # Both vegan + client-ready: find overlap, fall back to vegan, then client, then zone
        r_both = [r for r in r_vegan if float(r.get("client_ready_score", 0)) >= 7.5]
        rest_pref = cheapest(r_both, r_vegan or r_client or r_zone or rests, lambda r: r["price_level"])
    elif require_vegan:
        rest_pref = cheapest(r_vegan, r_zone or rests, lambda r: r["price_level"])
    elif require_client_dinner:
        rest_pref = cheapest(r_client, r_zone or rests, lambda r: r["price_level"])
    else:
        rest_pref = rest_cheap

    # Activity in meeting_zone. Weather-safe when rainy.
    a_zone = [a for a in acts if a.get("location_zone") == mz]
    if weather == "rainy":
        a_zone = [a for a in a_zone if "weather_safe" in a.get("semantic_tags", [])] or a_zone
    activity = cheapest(a_zone, acts, lambda a: a["price"])

    def total(f, h, r, a):
        c = 0.0
        if f:
            c += f["fare_total"]
        if h:
            c += h["nightly_price"] * nights
        if r:
            c += r["price_level"] * 25000
        if a:
            c += a["price"]
        return c

    # Start from preferred (refundable/vegan); relax to cheapest if over budget.
    flight, restaurant = flight_pref, rest_pref
    if budget and total(flight, hotel, restaurant, activity) > budget:
        restaurant = rest_cheap
    if budget and total(flight, hotel, restaurant, activity) > budget:
        flight = flight_cheap
    # If zone hotel pushed us over budget, fall back to any quiet hotel
    if budget and total(flight, hotel, restaurant, activity) > budget and h_quiet_zone:
        hotel = cheapest(
            h_quiet or [h for h in hotels if "quiet" in h.get("semantic_tags", [])] or hotels,
            hotels,
            lambda h: h["nightly_price"],
        )

    # ── Exhaustive budget fallback ───────────────────────────────────
    # If greedy relaxation still busts the budget, brute-force the cheapest
    # flight × hotel combo that fits (keeping restaurant + activity fixed).
    if budget and total(flight, hotel, restaurant, activity) > budget:
        # Sort all flights and quiet hotels by cost
        all_flights_sorted = sorted(
            [f for f in flights if not f.get("red_eye")],
            key=lambda f: f["fare_total"],
        )
        all_hotels_sorted = sorted(
            h_quiet or hotels,
            key=lambda h: h["nightly_price"],
        )
        # Also try cheapest restaurant if still over
        r_all_sorted = sorted(rests, key=lambda r: r["price_level"])
        r_candidates = [restaurant] + [r for r in r_all_sorted if r != restaurant]

        found = False
        for rc in r_candidates[:3]:
            for fc in all_flights_sorted[:5]:
                for hc in all_hotels_sorted[:5]:
                    if total(fc, hc, rc, activity) <= budget:
                        flight, hotel, restaurant = fc, hc, rc
                        found = True
                        break
                if found:
                    break
            if found:
                break

    # ── Bundle-aware pass: check partner promotions ──────────────────
    # Try to find a bundle combo that matches a partner promotion while
    # staying under budget. Only swap if the new combo is still feasible.
    result = {"flight": flight, "hotel": hotel, "restaurant": restaurant, "activity": activity}
    try:
        result = _try_bundle_upgrade(env, episode, result, nights, budget)
    except Exception:
        pass  # Bundle optimization is best-effort; never break the baseline

    return result


def _try_bundle_upgrade(
    env: Any,
    episode: Dict[str, Any],
    current: Dict[str, Optional[Dict[str, Any]]],
    nights: int,
    budget: float,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Deterministic bundle optimization: check partner promotions and try
    to swap items to form a valid bundle that's still under budget.

    No LLM calls — uses env.get_partner_promotions() directly.
    """
    city = episode["city"]
    family = episode.get("family", "")
    mz = episode.get("meeting_zone")
    weather = episode.get("weather")

    promos = env.get_partner_promotions(city=city, family=family)
    if not promos:
        return current

    hotels = env.search_hotels(city)
    rests = env.search_restaurants(city)
    acts = env.search_activities(city)

    flight = current["flight"]
    # Compute arrival_minutes for cutoff check
    arrival_minutes = None
    if flight and flight.get("arrival_time") and ":" in str(flight.get("arrival_time", "")):
        parts = str(flight["arrival_time"]).split(":", 1)
        try:
            arrival_minutes = int(parts[0]) * 60 + int(parts[1])
        except (ValueError, IndexError):
            pass

    h_index = {h["hotel_id"]: h for h in hotels}
    r_index = {r["restaurant_id"]: r for r in rests}
    a_index = {a["activity_id"]: a for a in acts}

    def _total(f, h, r, a):
        c = 0.0
        if f: c += f.get("fare_total", 0)
        if h: c += h.get("nightly_price", 0) * nights
        if r: c += r.get("price_level", 0) * 25000
        if a: c += a.get("price", 0)
        return c

    def _is_quiet(h):
        return "quiet" in h.get("semantic_tags", [])

    def _is_weather_safe(a):
        return weather != "rainy" or "weather_safe" in a.get("semantic_tags", [])

    best = dict(current)
    best_cost = _total(flight, current.get("hotel"), current.get("restaurant"), current.get("activity"))

    for promo in promos:
        # Check arrival cutoff
        cutoff = promo.get("arrival_before")
        if cutoff and ":" in cutoff and arrival_minutes is not None:
            hh, mm = cutoff.split(":", 1)
            try:
                cutoff_min = int(hh) * 60 + int(mm)
                if arrival_minutes > cutoff_min:
                    continue
            except (ValueError, IndexError):
                pass

        # Determine which items the promotion links
        p_hotel_ids = [pid for pid in (promo.get("hotel_ids") or []) if pid in h_index]
        p_rest_ids = [pid for pid in (promo.get("restaurant_ids") or []) if pid in r_index]
        p_act_ids = [pid for pid in (promo.get("activity_ids") or []) if pid in a_index]

        # Try combinations: keep current items where promo doesn't specify alternatives
        candidate_hotels = [h_index[hid] for hid in p_hotel_ids if _is_quiet(h_index[hid])] if p_hotel_ids else [current["hotel"]] if current["hotel"] else []
        candidate_rests = [r_index[rid] for rid in p_rest_ids] if p_rest_ids else [current["restaurant"]] if current["restaurant"] else []
        candidate_acts = [a_index[aid] for aid in p_act_ids if _is_weather_safe(a_index[aid])] if p_act_ids else [current["activity"]] if current["activity"] else []

        for ch in candidate_hotels:
            for cr in candidate_rests:
                for ca in candidate_acts:
                    cost = _total(flight, ch, cr, ca)
                    if budget and cost > budget:
                        continue
                    # Check zone coherence: >=2 of 3 venues in meeting_zone
                    zones = [ch.get("zone"), cr.get("area"), ca.get("location_zone")]
                    zone_hits = sum(1 for z in zones if z == mz)
                    if zone_hits < 2:
                        continue
                    # Prefer this bundle combo if it's cheaper
                    if cost <= best_cost:
                        best = {"flight": flight, "hotel": ch, "restaurant": cr, "activity": ca}
                        best_cost = cost

    return best


# ═══════════════════════════════════════════════════════════════════════
#  DYNAMIC SESSION PATCHING
# ═══════════════════════════════════════════════════════════════════════

import types

def patch_session_tools(session: Any) -> None:
    """Monkey-patch the session object to use custom scoring and sorting logic."""
    from llm_tools import compact_item

    def _build_scoring_context(self) -> Dict[str, Any]:
        ctx = {}
        if not getattr(self, "board", None):
            return ctx
            
        td = getattr(self.board, "trip_details", None)
        if td:
            ctx["budget_total"] = getattr(td, "budget_total", 0.0)
            ctx["nights"] = getattr(td, "nights", 1)
            ctx["city"] = getattr(td, "city", "")
            ctx["target_zone"] = getattr(td, "meeting_zone", "")
            
        hc = getattr(self.board, "hard_constraints", [])
        if "avoid_red_eye" in hc:
            ctx["red_eye_ok"] = False
            ctx["meeting_safe_bonus"] = 5.0
        if "prefer_quiet_hotel" in hc:
            ctx["quiet_weight"] = 5.0
        if "team_dietary_flex" in hc:
            ctx["dietary_vegan"] = True
        if "budget_cap" in hc:
            ctx["price_weight"] = 2.0
            
        rejected_ids = []
        for r in self.toolbox.rejected_options:
            if "hotel_id" in r: rejected_ids.append(r["hotel_id"])
            if "flight_id" in r: rejected_ids.append(r["flight_id"])
            if "restaurant_id" in r: rejected_ids.append(r["restaurant_id"])
            if "activity_id" in r: rejected_ids.append(r["activity_id"])
        ctx["rejected_ids"] = rejected_ids
        
        if hasattr(self, "logger") and self.logger:
            self.logger._print(f"   \033[38;5;180m🔧 Reranking Context:\033[0m {ctx}")
            import datetime
            self.logger._write_json({
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "event": "rerank_context",
                "trip_id": getattr(self.logger, "trip_id", ""),
                "context": ctx
            })
        
        return ctx

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
        exclude_ids_set = set(exclude_ids or [])
        out = []
        for row in rows:
            if flight_id and row["flight_id"] != flight_id:
                continue
            if row["flight_id"] in exclude_ids_set:
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
        
        ctx = _build_scoring_context(self)
        for row in out:
            row["heuristic_score"] = round(_score_flight(row, ctx), 2)
            
        sort_key = {
            "fare_total": lambda row: (row["fare_total"], row.get("red_eye", False)),
            "meeting_safe": lambda row: (("meeting_safe" not in row.get("semantic_tags", [])), row["fare_total"]),
            "change_friendly": lambda row: (("change_friendly" not in row.get("semantic_tags", [])), row["fare_total"]),
            "heuristic_score": lambda row: -row.get("heuristic_score", 0),
        }.get(sort_by or "heuristic_score", lambda row: -row.get("heuristic_score", 0))
        out.sort(key=sort_key)
        return {
            "items": [compact_item(row, ["flight_id", "time_window", "fare_total", "depart_time", "arrival_time", "duration_minutes", "refundable", "stops", "red_eye", "semantic_tags", "description_snippet", "heuristic_score"]) for row in out[: min(max_results, self.max_results)]]
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
        exclude_zones_set = set(exclude_zones or [])
        exclude_ids_set = set(exclude_ids or [])
        out = []
        for row in rows:
            if hotel_id and row["hotel_id"] != hotel_id:
                continue
            if row["hotel_id"] in exclude_ids_set:
                continue
            if row.get("zone") in exclude_zones_set:
                continue
            if quiet_min is not None and row.get("quiet_score", 0) < quiet_min:
                continue
            if airport_access_min is not None and row.get("airport_access_score", 0) < airport_access_min:
                continue
            if not chain_ok and row.get("chain"):
                continue
            if max_nightly_price is not None and row["nightly_price"] > max_nightly_price:
                continue
            out.append(row)
            
        ctx = _build_scoring_context(self)
        for row in out:
            row["heuristic_score"] = round(_score_hotel(row, ctx), 2)
            
        sort_key = {
            "quiet_score": lambda row: (-row.get("quiet_score", 0.0), row["nightly_price"]),
            "airport_access": lambda row: (-row.get("airport_access_score", 0.0), row["nightly_price"]),
            "price": lambda row: (row["nightly_price"], -row.get("quiet_score", 0.0)),
            "zone_match": lambda row: (row.get("zone") != preferred_zone, row["nightly_price"]),
            "heuristic_score": lambda row: -row.get("heuristic_score", 0),
        }.get(sort_by or "heuristic_score", lambda row: -row.get("heuristic_score", 0))
        out.sort(key=sort_key)
        return {
            "items": [compact_item(row, ["hotel_id", "nightly_price", "quiet_score", "zone", "chain", "airport_access_score", "late_checkout", "meeting_shuttle", "semantic_tags", "review_snippet", "heuristic_score"]) for row in out[: min(max_results, self.max_results)]]
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
        exclude_areas_set = set(exclude_areas or [])
        exclude_ids_set = set(exclude_ids or [])
        out = []
        for row in rows:
            if restaurant_id and row["restaurant_id"] != restaurant_id:
                continue
            if row["restaurant_id"] in exclude_ids_set:
                continue
            if row.get("area") in exclude_areas_set:
                continue
            if dietary and dietary not in row.get("dietary_flags", []):
                continue
            if quiet_min is not None and row.get("quiet_score", 0) < quiet_min:
                continue
            if client_ready_min is not None and row.get("client_ready_score", 0) < client_ready_min:
                continue
            if max_price_level is not None and row["price_level"] > max_price_level:
                continue
            out.append(row)
            
        ctx = _build_scoring_context(self)
        for row in out:
            row["heuristic_score"] = round(_score_restaurant(row, ctx), 2)
            
        sort_key = {
            "quiet_score": lambda row: (-row.get("quiet_score", 0.0), row["price_level"]),
            "client_ready": lambda row: (-row.get("client_ready_score", 0.0), row["price_level"]),
            "area_match": lambda row: (row.get("area") != preferred_area, row["price_level"]),
            "price": lambda row: (row["price_level"], -row.get("quiet_score", 0.0)),
            "heuristic_score": lambda row: -row.get("heuristic_score", 0),
        }.get(sort_by or "heuristic_score", lambda row: -row.get("heuristic_score", 0))
        out.sort(key=sort_key)
        return {
            "items": [compact_item(row, ["restaurant_id", "cuisine", "price_level", "dietary_flags", "area", "quiet_score", "client_ready_score", "private_room", "booking_cutoff", "badge_only", "semantic_tags", "review_snippet", "heuristic_score"]) for row in out[: min(max_results, self.max_results)]]
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
        exclude_ids_set = set(exclude_ids or [])
        out = []
        for row in rows:
            if activity_id and row["activity_id"] != activity_id:
                continue
            if row["activity_id"] in exclude_ids_set:
                continue
            if indoor_only and not row.get("indoor"):
                continue
            if weather_safe_required and "weather_safe" not in row.get("semantic_tags", []):
                continue
            if max_price is not None and row.get("price", 0) > max_price:
                continue
            out.append(row)
            
        ctx = _build_scoring_context(self)
        for row in out:
            row["heuristic_score"] = round(_score_activity(row, ctx), 2)
            
        sort_key = {
            "zone_match": lambda row: (row.get("location_zone") != preferred_zone, row.get("price", 0)),
            "price": lambda row: (row.get("price", 0), row.get("indoor") is False),
            "weather_safe": lambda row: (("weather_safe" not in row.get("semantic_tags", [])), row.get("price", 0)),
            "heuristic_score": lambda row: -row.get("heuristic_score", 0),
        }.get(sort_by or "heuristic_score", lambda row: -row.get("heuristic_score", 0))
        out.sort(key=sort_key)
        return {
            "items": [compact_item(row, ["activity_id", "category", "location_zone", "indoor", "price", "badge_only", "semantic_tags", "description_snippet", "heuristic_score"]) for row in out[: min(max_results, self.max_results)]]
        }

    session.search_flights = types.MethodType(search_flights, session)
    session.search_hotels = types.MethodType(search_hotels, session)
    session.search_restaurants = types.MethodType(search_restaurants, session)
    session.search_activities = types.MethodType(search_activities, session)
