from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rtl_semantic_env import RTLSemanticEnv

HERE = Path(__file__).resolve().parent


ID_FIELDS = {
    "inventory_flights.json": "flight_id",
    "inventory_hotels.json": "hotel_id",
    "inventory_restaurants.json": "restaurant_id",
    "inventory_activities.json": "activity_id",
    "memory_corpus.json": "doc_id",
    "rejected_options_memory.json": "memory_id",
    "partner_promotions.json": "promo_id",
    "event_calendar.json": "event_id",
    "booking_constraints.json": "constraint_id",
    "option_dependencies.json": "dependency_id",
}


def write_json(name: str, payload: Any) -> None:
    (HERE / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _load_seed(name: str, default: Any) -> Any:
    path = HERE / name
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _merge_unique(rows: List[Dict[str, Any]], extras: List[Dict[str, Any]], id_field: str) -> List[Dict[str, Any]]:
    merged = {row[id_field]: dict(row) for row in rows if id_field in row}
    for row in extras:
        merged[row[id_field]] = dict(row)
    return list(merged.values())


def _merge_dict(seed: Dict[str, Any], extras: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(seed)
    out.update(extras)
    return out


ROUTE_CONFIGS = {
    "OSA_business_travel": {"city": "OSA", "family": "business_travel", "origin": "GMP", "base_budget": 610000, "weather": "clear"},
    "OSA_board_visit": {"city": "OSA", "family": "board_visit", "origin": "GMP", "base_budget": 650000, "weather": "clear"},
    "TPE_conference_trip": {"city": "TPE", "family": "conference_trip", "origin": "SEL", "base_budget": 590000, "weather": "clear"},
    "TPE_partner_summit": {"city": "TPE", "family": "partner_summit", "origin": "SEL", "base_budget": 635000, "weather": "clear"},
    "SIN_business_travel": {"city": "SIN", "family": "business_travel", "origin": "PUS", "base_budget": 610000, "weather": "rainy"},
    "SIN_roadshow_trip": {"city": "SIN", "family": "roadshow_trip", "origin": "PUS", "base_budget": 645000, "weather": "rainy"},
}

BASE_TURNS = {
    "business_travel": [
        "Please build a short trip plan that keeps me functional for work, not just technically on time.",
        "A teammate may join dinner, so dietary flexibility matters more than a flashy venue.",
    ],
    "conference_trip": [
        "I need to arrive ready for an early conference morning, not merely arrive somehow.",
        "A small networking dinner is welcome only if it does not damage the next day.",
    ],
    "board_visit": [
        "This trip includes a board-facing session, so the plan should feel reliable and polished, not improvised.",
        "I can accept a slightly less local option if it clearly reduces friction around prep and client optics.",
    ],
    "partner_summit": [
        "This is a partner summit, so I care about a calm but polished evening room more than novelty.",
        "If a venue badge unlocks a genuinely better bundle, count it, but only if the full plan still stays lean and low-risk.",
    ],
    "roadshow_trip": [
        "This roadshow has moving parts, so I want a bundle that stays resilient when timing shifts.",
        "Please optimize the whole path — arrival, hotel, dinner, and buffer activity — instead of picking isolated winners.",
    ],
}

RULE_TEMPLATES = {
    "profile_use": "My usual travel habits are in my profile. Please use them, but do not let one-off exceptions rewrite them.",
    "must_remember_quiet": "Please remember that a genuinely quiet room matters to me; do not treat it as cosmetic.",
    "must_remember_client_ready": "Remember that client-facing or partner-facing dinners should feel polished enough for conversation, not merely convenient.",
    "forbid_red_eye": "Do not send me through a red-eye just because the fare is lower.",
    "forbid_loud_zone": "Please filter out places that feel loud after 10pm; nightlife spillover is a real problem for this trip.",
    "allow_chain_once": "This trip only, a chain hotel is acceptable if it clearly reduces risk; do not rewrite my normal profile with that exception.",
    "airport_priority_once": "For this trip only, airport access matters more than local character.",
    "retire_old_budget": "The older budget assumption is no longer valid, so retire it rather than repeating it.",
    "retire_old_local_char": "After the risk update, stop carrying the old local-character preference in active context unless it becomes relevant again.",
    "remember_rejection_noise": "If you reject a hotel for noise, keep that reason around and do not surface it again unless something material changes.",
    "remember_rejection_vibe": "If a dinner option is out for the wrong vibe, keep it out instead of rediscovering the same mistake later.",
    "retrieval_discipline": "Bring back only the relevant old preferences, not the whole history; keep active context lean.",
    "compressed_followup": "Keep the useful parts, drop stale assumptions, and adjust only what the new situation forces.",
    "weather_shift": "Weather looks worse now, so brittle outdoor ideas should probably drop out.",
    "client_dinner_now": "This dinner should feel polished enough for a client or partner conversation, not just convenient.",
    "no_recycling": "Bring back only the relevant bits from memory; if a candidate already failed for noise or vibe, do not recycle it.",
    "bundle_value": "I heard some hotel+dinner or shuttle bundles may exist. Count them only if they improve the full plan, not just the sticker price.",
    "badge_unlock": "I will have the event badge this time, so if it unlocks a calmer or lower-friction option, that is relevant for this trip only.",
    "refund_risk": "Timing may still move, so refundable bookings deserve real weight instead of being treated as a trivial nice-to-have.",
    "late_arrival": "Do not assume a nominal dinner bundle still works after a late arrival; some perks disappear after check-in cutoffs.",
    "teammate_vegan": "A teammate may join dinner and may need vegan-capable options, so group flexibility matters.",
    "loyalty_value": "Please consider whether a loyalty perk changes the effective value of a bundle, not just the list price.",
    "event_disruption": "There is a city event tonight that may change transfer friction or noise in one zone, so do not rely on stale assumptions.",
}


def inventory_flights() -> List[Dict[str, Any]]:
    seed = _load_seed("inventory_flights.json", [])
    extras = [
        {"flight_id": "FL106", "origin": "GMP", "destination": "OSA", "depart_date": "2026-10-12", "return_date": "2026-10-14", "time_window": "morning", "fare_total": 445000, "airline": "KE", "depart_time": "07:20", "arrival_time": "08:55", "duration_minutes": 95, "refundable": True, "baggage_included": True, "stops": 0, "red_eye": False},
        {"flight_id": "FL107", "origin": "GMP", "destination": "OSA", "depart_date": "2026-10-12", "return_date": "2026-10-14", "time_window": "afternoon", "fare_total": 365000, "airline": "7C", "depart_time": "16:10", "arrival_time": "17:45", "duration_minutes": 95, "refundable": False, "baggage_included": True, "stops": 0, "red_eye": False},
        {"flight_id": "FL706", "origin": "SEL", "destination": "TPE", "depart_date": "2026-12-03", "return_date": "2026-12-05", "time_window": "afternoon", "fare_total": 485000, "airline": "KE", "depart_time": "13:25", "arrival_time": "15:10", "duration_minutes": 105, "refundable": True, "baggage_included": True, "stops": 0, "red_eye": False},
        {"flight_id": "FL707", "origin": "SEL", "destination": "TPE", "depart_date": "2026-12-03", "return_date": "2026-12-05", "time_window": "evening", "fare_total": 435000, "airline": "TW", "depart_time": "18:45", "arrival_time": "20:35", "duration_minutes": 110, "refundable": False, "baggage_included": True, "stops": 0, "red_eye": False},
        {"flight_id": "FL806", "origin": "PUS", "destination": "SIN", "depart_date": "2026-11-12", "return_date": "2026-11-15", "time_window": "afternoon", "fare_total": 635000, "airline": "SQ", "depart_time": "15:05", "arrival_time": "20:25", "duration_minutes": 320, "refundable": True, "baggage_included": True, "stops": 0, "return_depart_time": "13:00", "red_eye": False},
        {"flight_id": "FL807", "origin": "PUS", "destination": "SIN", "depart_date": "2026-11-12", "return_date": "2026-11-15", "time_window": "evening", "fare_total": 560000, "airline": "TR", "depart_time": "18:50", "arrival_time": "00:10", "duration_minutes": 320, "refundable": False, "baggage_included": False, "stops": 0, "return_depart_time": "08:15", "red_eye": False},
    ]
    return _merge_unique(seed, extras, "flight_id")


def inventory_hotels() -> List[Dict[str, Any]]:
    seed = _load_seed("inventory_hotels.json", [])
    extras = [
        {"hotel_id": "HT207", "city": "OSA", "nightly_price": 218000, "rating": 4.7, "quiet_score": 8.8, "distance_to_center_km": 1.0, "zone": "namba", "chain": False, "airport_access_score": 7.6, "vegan_option": False, "late_checkout": True, "meeting_shuttle": False},
        {"hotel_id": "HT208", "city": "OSA", "nightly_price": 205000, "rating": 4.3, "quiet_score": 7.9, "distance_to_center_km": 1.5, "zone": "umeda", "chain": True, "airport_access_score": 8.7, "vegan_option": False, "late_checkout": True, "meeting_shuttle": True},
        {"hotel_id": "HT806", "city": "TPE", "nightly_price": 240000, "rating": 4.7, "quiet_score": 8.9, "distance_to_center_km": 1.1, "zone": "xinyi", "chain": False, "airport_access_score": 8.8, "vegan_option": False, "late_checkout": False, "meeting_shuttle": False},
        {"hotel_id": "HT807", "city": "TPE", "nightly_price": 215000, "rating": 4.2, "quiet_score": 8.1, "distance_to_center_km": 2.0, "zone": "blue_line_corridor", "chain": True, "airport_access_score": 9.3, "vegan_option": False, "late_checkout": True, "meeting_shuttle": True},
        {"hotel_id": "HT907", "city": "SIN", "nightly_price": 245000, "rating": 4.6, "quiet_score": 8.5, "distance_to_center_km": 1.1, "zone": "one_north", "chain": True, "airport_access_score": 8.8, "vegan_option": False, "late_checkout": True, "meeting_shuttle": True},
        {"hotel_id": "HT908", "city": "SIN", "nightly_price": 225000, "rating": 4.3, "quiet_score": 8.0, "distance_to_center_km": 1.6, "zone": "airport_link", "chain": True, "airport_access_score": 9.5, "vegan_option": False, "late_checkout": True, "meeting_shuttle": False},
    ]
    return _merge_unique(seed, extras, "hotel_id")


def inventory_restaurants() -> List[Dict[str, Any]]:
    seed = _load_seed("inventory_restaurants.json", [])
    extras = [
        {"restaurant_id": "RS3004", "city": "OSA", "cuisine": "kaiseki", "price_level": 4, "dietary_flags": ["vegan_preorder"], "area": "namba", "quiet_score": 8.6, "client_ready_score": 8.7, "private_room": True, "booking_cutoff": "20:30"},
        {"restaurant_id": "RS3005", "city": "OSA", "cuisine": "bistro", "price_level": 2, "dietary_flags": ["vegan"], "area": "umeda", "quiet_score": 7.9, "client_ready_score": 6.9, "private_room": False, "booking_cutoff": "21:30"},
        {"restaurant_id": "RS4004", "city": "TPE", "cuisine": "taiwanese_set", "price_level": 4, "dietary_flags": ["vegan_preorder"], "area": "xinyi", "quiet_score": 8.4, "client_ready_score": 8.6, "private_room": True, "booking_cutoff": "20:45", "badge_only": True},
        {"restaurant_id": "RS4005", "city": "TPE", "cuisine": "plant_forward", "price_level": 3, "dietary_flags": ["vegan"], "area": "blue_line_corridor", "quiet_score": 8.0, "client_ready_score": 7.1, "private_room": False, "booking_cutoff": "21:00"},
        {"restaurant_id": "RS2004", "city": "SIN", "cuisine": "modern_asian", "price_level": 3, "dietary_flags": ["vegan_preorder"], "area": "one_north", "quiet_score": 8.3, "client_ready_score": 7.8, "private_room": True, "booking_cutoff": "20:15"},
        {"restaurant_id": "RS2005", "city": "SIN", "cuisine": "marina_grill", "price_level": 4, "dietary_flags": ["vegan_preorder"], "area": "marina", "quiet_score": 7.8, "client_ready_score": 8.8, "private_room": True, "booking_cutoff": "21:00"},
    ]
    return _merge_unique(seed, extras, "restaurant_id")


def inventory_activities() -> List[Dict[str, Any]]:
    seed = _load_seed("inventory_activities.json", [])
    extras = [
        {"activity_id": "ACT104_indoor_tea", "city": "OSA", "category": "tea_house", "location_zone": "namba", "indoor": True, "price": 28000},
        {"activity_id": "ACT105_partner_lounge", "city": "OSA", "category": "meeting_buffer", "location_zone": "umeda", "indoor": True, "price": 38000},
        {"activity_id": "ACT404_partner_lounge", "city": "TPE", "category": "meeting_buffer", "location_zone": "xinyi", "indoor": True, "price": 42000, "badge_only": True},
        {"activity_id": "ACT405_innovation_gallery", "city": "TPE", "category": "gallery", "location_zone": "blue_line_corridor", "indoor": True, "price": 26000},
        {"activity_id": "ACT304_innovation_center", "city": "SIN", "category": "innovation_center_tour", "location_zone": "one_north", "indoor": True, "price": 40000},
        {"activity_id": "ACT305_boardroom_buffer", "city": "SIN", "category": "meeting_buffer", "location_zone": "airport_link", "indoor": True, "price": 36000},
    ]
    return _merge_unique(seed, extras, "activity_id")


def profile_briefs() -> Dict[str, Dict[str, Any]]:
    seed = _load_seed("profile_briefs.json", {})
    extras = {
        "traveler_sales": {
            "traveler_id": "traveler_sales",
            "doc_id": "profile:traveler_sales",
            "text": "Stable profile: values polished but calm dinners, accepts bundles if they reduce friction, prefers refundable plans when partner schedules look volatile, and dislikes carrying too many conditional notes in active context.",
            "stable_prefs": ["client_dinner_polished", "low_friction_transit", "refundable_priority", "relevant_only"],
            "avoid_zones": ["nightlife_strip"],
            "preferred_zones": ["namba", "xinyi", "one_north", "marina"],
        },
        "traveler_ops": {
            "traveler_id": "traveler_ops",
            "doc_id": "profile:traveler_ops",
            "text": "Stable profile: airport access, transfer resilience, and quiet sleep matter. Will accept a chain when it meaningfully lowers operational risk, but does not want stale exceptions to rewrite the normal profile.",
            "stable_prefs": ["prefer_airport_access", "low_friction_transit", "prefer_quiet_hotel", "chain_exception_this_trip"],
            "avoid_zones": ["nightlife_strip"],
            "preferred_zones": ["airport_link", "blue_line_corridor", "one_north"],
        },
    }
    return _merge_dict(seed, extras)


def venue_briefs() -> Dict[str, Dict[str, Any]]:
    seed = _load_seed("venue_briefs.json", {})
    extras = {
        "OSA_board_visit": {
            "doc_id": "venue:OSA_board_visit",
            "text": "Board-visit brief: meetings concentrate around Umeda with one evening stakeholder dinner. Quiet prep time and a room that supports polished conversation matter. A private-room bundle can be valuable if it stays calm and close.",
            "soft_tags": ["business_friendly", "quiet", "low_friction", "client_ready", "central_bundle"],
            "preferred_zone": "umeda",
        },
        "TPE_partner_summit": {
            "doc_id": "venue:TPE_partner_summit",
            "text": "Partner summit brief: sessions cluster around Xinyi and the blue-line corridor. Badge-enabled partner rooms can reduce noise and transit friction, but scenic detours often become brittle when schedules move.",
            "soft_tags": ["quiet", "easy_airport_access", "low_friction", "conference_ready", "client_ready"],
            "preferred_zone": "xinyi",
        },
        "SIN_roadshow_trip": {
            "doc_id": "venue:SIN_roadshow_trip",
            "text": "Roadshow brief: the day oscillates between One-North and the airport corridor. Rain or a late arrival can quickly invalidate fragile dinner bundles, so resilient shuttle-linked plans score best.",
            "soft_tags": ["easy_airport_access", "weather_safe", "low_friction", "business_friendly", "central_bundle"],
            "preferred_zone": "one_north",
        },
    }
    return _merge_dict(seed, extras)


def city_ops_notes() -> Dict[str, Dict[str, Any]]:
    seed = _load_seed("city_ops_notes.json", {})
    extras = {
        "OSA": {
            "doc_id": "city_ops:OSA",
            "text": "Ops note: Namba is efficient for central meetings, while Shinsekai tends to be louder after 22:00. Umeda can support polished dinners, but some partner bundles stop working after late check-in.",
            "avoid_zone": "shinsekai",
            "risk_tags": ["noise_risk", "late_checkin_risk"],
        },
        "TPE": {
            "doc_id": "city_ops:TPE",
            "text": "Ops note: Xinyi works best for conference mornings. Airport access is smoother from the blue-line corridor, and expo nights can make scenic outer areas slower than they look. Badge-enabled partner rooms can matter.",
            "avoid_zone": "scenic_outer",
            "risk_tags": ["late_return_risk", "badge_access"],
        },
        "SIN": {
            "doc_id": "city_ops:SIN",
            "text": "Ops note: One-North is dependable for business travel during wet weather. Marina dinners can look polished but add transfer overhead, while shuttle-linked bundles hold up better on rainy nights.",
            "avoid_zone": "clarke_quay",
            "risk_tags": ["weather_transfer_risk", "shuttle_bundle"],
        },
    }
    return _merge_dict(seed, extras)


def partner_promotions() -> List[Dict[str, Any]]:
    return [
        {
            "promo_id": "promo:OSA_ht206_rs3001_private_room",
            "doc_id": "promo:OSA_ht206_rs3001_private_room",
            "city": "OSA",
            "family": "business_travel",
            "hotel_id": "HT206",
            "restaurant_id": "RS3001",
            "activity_id": None,
            "discount_krw": 35000,
            "score_bonus": 3.2,
            "benefit_tags": ["bundle_discount_value", "private_room_bonus", "late_checkin_risk"],
            "badge_required": False,
            "arrival_before": "20:30",
            "text": "HT206 guests can access a quieter private-room seating at RS3001 with a set-menu credit, but the bundle expires after a late check-in.",
        },
        {
            "promo_id": "promo:OSA_ht207_rs3004_board_bundle",
            "doc_id": "promo:OSA_ht207_rs3004_board_bundle",
            "city": "OSA",
            "family": "board_visit",
            "hotel_id": "HT207",
            "restaurant_id": "RS3004",
            "activity_id": None,
            "discount_krw": 42000,
            "score_bonus": 3.8,
            "benefit_tags": ["bundle_discount_value", "private_room_bonus"],
            "badge_required": False,
            "arrival_before": "21:00",
            "text": "For board-facing trips, HT207 and RS3004 run a quiet private-room dinner bundle close to the meeting zone.",
        },
        {
            "promo_id": "promo:TPE_ht801_rs4004_badge_room",
            "doc_id": "promo:TPE_ht801_rs4004_badge_room",
            "city": "TPE",
            "family": "conference_trip",
            "hotel_id": "HT801",
            "restaurant_id": "RS4004",
            "activity_id": None,
            "discount_krw": 30000,
            "score_bonus": 3.6,
            "benefit_tags": ["bundle_discount_value", "conference_badge_access", "private_room_bonus"],
            "badge_required": True,
            "arrival_before": "21:00",
            "text": "Conference badge holders staying at HT801 can access RS4004's calmer partner room with a fixed-menu credit.",
        },
        {
            "promo_id": "promo:TPE_ht807_rs4005_airport_corridor",
            "doc_id": "promo:TPE_ht807_rs4005_airport_corridor",
            "city": "TPE",
            "family": "partner_summit",
            "hotel_id": "HT807",
            "restaurant_id": "RS4005",
            "activity_id": "ACT405_innovation_gallery",
            "discount_krw": 28000,
            "score_bonus": 2.8,
            "benefit_tags": ["bundle_discount_value", "low_friction_transit"],
            "badge_required": False,
            "arrival_before": None,
            "text": "The blue-line corridor bundle links HT807, RS4005, and ACT405 for a lower-friction partner summit evening.",
        },
        {
            "promo_id": "promo:SIN_ht907_rs2004_shuttle_bundle",
            "doc_id": "promo:SIN_ht907_rs2004_shuttle_bundle",
            "city": "SIN",
            "family": "roadshow_trip",
            "hotel_id": "HT907",
            "restaurant_id": "RS2004",
            "activity_id": "ACT304_innovation_center",
            "discount_krw": 32000,
            "score_bonus": 3.5,
            "benefit_tags": ["bundle_discount_value", "shuttle_bundle", "weather_safe_backup"],
            "badge_required": False,
            "arrival_before": "20:30",
            "text": "HT907 runs a shuttle-linked One-North bundle with RS2004 and ACT304 that stays resilient under rain, but late arrivals erode the dinner benefit.",
        },
        {
            "promo_id": "promo:SIN_ht908_rs2005_client_reception",
            "doc_id": "promo:SIN_ht908_rs2005_client_reception",
            "city": "SIN",
            "family": "business_travel",
            "hotel_id": "HT908",
            "restaurant_id": "RS2005",
            "activity_id": None,
            "discount_krw": 25000,
            "score_bonus": 2.4,
            "benefit_tags": ["bundle_discount_value", "client_dinner_polished"],
            "badge_required": False,
            "arrival_before": None,
            "text": "Airport-link guests at HT908 get a modest credit at RS2005, but the marina transfer still needs to be justified on a work trip.",
        },
    ]


def event_calendar() -> List[Dict[str, Any]]:
    return [
        {
            "event_id": "event:OSA_namba_food_fair",
            "doc_id": "event:OSA_namba_food_fair",
            "city": "OSA",
            "affected_zone": "namba",
            "effect_type": "noise_surge",
            "blocked_restaurant_id": None,
            "boosted_hotel_id": None,
            "text": "A Namba food fair brings more crowd spillover after 21:30. Quiet rooms and private dining matter more than usual.",
            "tags": ["loud_after_10pm", "transfer_friction_risk"],
        },
        {
            "event_id": "event:OSA_umeda_private_room_night",
            "doc_id": "event:OSA_umeda_private_room_night",
            "city": "OSA",
            "affected_zone": "umeda",
            "effect_type": "private_room_boost",
            "blocked_restaurant_id": None,
            "boosted_hotel_id": "HT207",
            "text": "A quiet private-room night in Umeda makes polished dinners unusually viable if the bundle still stays lean.",
            "tags": ["private_room_bonus"],
        },
        {
            "event_id": "event:TPE_xinyi_expo_surge",
            "doc_id": "event:TPE_xinyi_expo_surge",
            "city": "TPE",
            "affected_zone": "xinyi",
            "effect_type": "transit_congestion",
            "blocked_restaurant_id": None,
            "boosted_hotel_id": "HT807",
            "text": "Expo traffic slows Xinyi edge routes, which makes blue-line corridor access and pre-booked quiet rooms more valuable.",
            "tags": ["transfer_friction_risk", "prefer_airport_access"],
        },
        {
            "event_id": "event:TPE_riverside_buyout",
            "doc_id": "event:TPE_riverside_buyout",
            "city": "TPE",
            "affected_zone": "scenic_outer",
            "effect_type": "buyout",
            "blocked_restaurant_id": "RS4002",
            "boosted_hotel_id": None,
            "text": "A riverside buyout removes one scenic dinner option and increases the cost of relying on outdated nightlife assumptions.",
            "tags": ["transfer_friction_risk"],
        },
        {
            "event_id": "event:SIN_one_north_thunderstorm",
            "doc_id": "event:SIN_one_north_thunderstorm",
            "city": "SIN",
            "affected_zone": "one_north",
            "effect_type": "rain_transfer",
            "blocked_restaurant_id": None,
            "boosted_hotel_id": "HT907",
            "text": "Thunderstorms increase transfer friction, so covered routes and shuttle-linked bundles outperform brittle scenic plans.",
            "tags": ["weather_safe_backup", "shuttle_bundle", "transfer_friction_risk"],
        },
        {
            "event_id": "event:SIN_marina_reception",
            "doc_id": "event:SIN_marina_reception",
            "city": "SIN",
            "affected_zone": "marina",
            "effect_type": "client_optics",
            "blocked_restaurant_id": None,
            "boosted_hotel_id": None,
            "text": "A marina reception temporarily improves client optics there, but the transfer burden still needs to be justified.",
            "tags": ["client_dinner_polished"],
        },
    ]


def loyalty_programs() -> Dict[str, Dict[str, Any]]:
    return {
        "traveler_exec": {
            "traveler_id": "traveler_exec",
            "doc_id": "loyalty:traveler_exec",
            "hotel_partner_ids": ["HT202", "HT206", "HT807", "HT907", "HT908"],
            "preferred_flight_ids": ["FL105", "FL705", "FL806"],
            "hotel_credit_krw": 18000,
            "bonus_tags": ["loyalty_bundle_value", "prefer_airport_access"],
            "text": "Executive loyalty profile: selected chain stays include breakfast or late checkout, slightly improving effective value when the operational fit is already good.",
        },
        "traveler_sales": {
            "traveler_id": "traveler_sales",
            "doc_id": "loyalty:traveler_sales",
            "hotel_partner_ids": ["HT208", "HT807", "HT907"],
            "preferred_flight_ids": ["FL106", "FL706", "FL806"],
            "hotel_credit_krw": 22000,
            "bonus_tags": ["loyalty_bundle_value", "refundable_priority"],
            "text": "Sales loyalty profile: premium but flexible chain bookings get extra value when schedule volatility is non-trivial.",
        },
        "traveler_ops": {
            "traveler_id": "traveler_ops",
            "doc_id": "loyalty:traveler_ops",
            "hotel_partner_ids": ["HT208", "HT807", "HT908"],
            "preferred_flight_ids": ["FL106", "FL706", "FL806"],
            "hotel_credit_krw": 20000,
            "bonus_tags": ["loyalty_bundle_value", "low_friction_transit"],
            "text": "Ops loyalty profile: airport-linked chain stays receive value from priority check-in and operational support when the trip gets messy.",
        },
    }


def stakeholder_briefs() -> Dict[str, Dict[str, Any]]:
    return {
        "stakeholder:teammate_vegan": {
            "stakeholder_id": "stakeholder:teammate_vegan",
            "doc_id": "stakeholder:teammate_vegan",
            "text": "Potential dinner stakeholder: a teammate may join and needs genuinely vegan-capable options, not token substitutions.",
            "tags": ["team_dietary_flex"],
        },
        "stakeholder:client_polished": {
            "stakeholder_id": "stakeholder:client_polished",
            "doc_id": "stakeholder:client_polished",
            "text": "Potential dinner stakeholder: the guest values a polished but calm room suitable for sustained conversation.",
            "tags": ["client_dinner_polished", "private_room_bonus"],
        },
        "stakeholder:ops_host": {
            "stakeholder_id": "stakeholder:ops_host",
            "doc_id": "stakeholder:ops_host",
            "text": "Operational host note: transfer reliability matters more than novelty if the schedule is moving.",
            "tags": ["low_friction_transit", "refundable_priority"],
        },
    }


def booking_constraints() -> List[Dict[str, Any]]:
    return [
        {
            "constraint_id": "constraint:badge_private_room_access",
            "doc_id": "constraint:badge_private_room_access",
            "city": "TPE",
            "family": None,
            "tags": ["conference_badge_access"],
            "text": "Some partner rooms are accessible only when the conference or summit badge is active.",
        },
        {
            "constraint_id": "constraint:late_arrival_voids_bundle",
            "doc_id": "constraint:late_arrival_voids_bundle",
            "city": None,
            "family": None,
            "tags": ["late_checkin_risk", "bundle_discount_value"],
            "text": "Certain hotel+dinner bundles expire after late check-in, so late arrivals can erase nominal discounts.",
        },
        {
            "constraint_id": "constraint:refund_priority_due_schedule_risk",
            "doc_id": "constraint:refund_priority_due_schedule_risk",
            "city": None,
            "family": None,
            "tags": ["refundable_priority"],
            "text": "When schedule volatility is high, refundable bookings should receive real weight rather than a token bonus.",
        },
        {
            "constraint_id": "constraint:team_dietary_support",
            "doc_id": "constraint:team_dietary_support",
            "city": None,
            "family": None,
            "tags": ["team_dietary_flex"],
            "text": "If a teammate joins dinner, the restaurant should be meaningfully vegan-capable or flexible for group ordering.",
        },
    ]


def option_dependencies() -> List[Dict[str, Any]]:
    return [
        {
            "dependency_id": "dependency:OSA_ht206_rs3001_private_room",
            "doc_id": "dependency:OSA_ht206_rs3001_private_room",
            "city": "OSA",
            "hotel_id": "HT206",
            "restaurant_id": "RS3001",
            "activity_id": None,
            "condition_tags": ["private_room_bonus"],
            "effect": "boost",
            "score_bonus": 1.7,
            "text": "HT206 makes RS3001 more viable because the private-room package reduces late noise exposure if check-in is early enough.",
            "tags": ["bundle_discount_value", "private_room_bonus"],
        },
        {
            "dependency_id": "dependency:TPE_ht801_rs4004_badge_bundle",
            "doc_id": "dependency:TPE_ht801_rs4004_badge_bundle",
            "city": "TPE",
            "hotel_id": "HT801",
            "restaurant_id": "RS4004",
            "activity_id": "ACT404_partner_lounge",
            "condition_tags": ["conference_badge_access"],
            "effect": "boost",
            "score_bonus": 2.0,
            "text": "With the summit badge active, HT801 + RS4004 + ACT404 becomes a calmer, tighter partner bundle.",
            "tags": ["conference_badge_access", "bundle_discount_value", "private_room_bonus"],
        },
        {
            "dependency_id": "dependency:SIN_ht907_rs2004_shuttle",
            "doc_id": "dependency:SIN_ht907_rs2004_shuttle",
            "city": "SIN",
            "hotel_id": "HT907",
            "restaurant_id": "RS2004",
            "activity_id": "ACT304_innovation_center",
            "condition_tags": ["shuttle_bundle", "weather_safe_backup"],
            "effect": "boost",
            "score_bonus": 2.1,
            "text": "HT907, RS2004, and ACT304 reinforce each other under rain because the shuttle and covered transfers lower friction materially.",
            "tags": ["shuttle_bundle", "weather_safe_backup", "low_friction_transit"],
        },
    ]


def memory_corpus() -> List[Dict[str, Any]]:
    seed = _load_seed("memory_corpus.json", [])
    extras: List[Dict[str, Any]] = []
    for profile in profile_briefs().values():
        extras.append({**profile, "memory_type": "profile", "status": "current", "city": None, "family": None, "traveler_id": profile["traveler_id"], "tags": profile.get("stable_prefs", [])})
    for venue_key, venue in venue_briefs().items():
        city, family = venue_key.split("_", 1)
        extras.append({**venue, "memory_type": "venue", "status": "current", "city": city, "family": family, "traveler_id": None, "tags": venue.get("soft_tags", [])})
    for city, ops in city_ops_notes().items():
        extras.append({**ops, "memory_type": "city_ops", "status": "current", "city": city, "family": None, "traveler_id": None, "tags": ops.get("risk_tags", [])})
    extras.extend(
        [
            {"doc_id": "heuristic:partner_bundle_reasoning", "memory_type": "heuristic", "status": "current", "city": None, "family": None, "traveler_id": None, "tags": ["bundle_discount_value", "private_room_bonus", "late_checkin_risk", "shuttle_bundle"], "text": "Partner-bundle rule: value a bundle only if the combined hotel+dinner+transfer path improves materially. Do not let a nominal discount outrank quietness, arrival quality, or stakeholder fit."},
            {"doc_id": "heuristic:stakeholder_balance", "memory_type": "heuristic", "status": "current", "city": None, "family": None, "traveler_id": None, "tags": ["team_dietary_flex", "client_dinner_polished"], "text": "Stakeholder balance rule: teammate dietary needs and client-facing tone can coexist, but only if the room stays calm enough for conversation."},
            {"doc_id": "heuristic:refundable_schedule_risk", "memory_type": "heuristic", "status": "current", "city": None, "family": None, "traveler_id": None, "tags": ["refundable_priority"], "text": "Volatile-schedule rule: refundable bookings deserve meaningful weight when the meeting window is still moving."},
            {"doc_id": "heuristic:badge_unlock_logic", "memory_type": "heuristic", "status": "current", "city": "TPE", "family": None, "traveler_id": None, "tags": ["conference_badge_access", "private_room_bonus"], "text": "Badge-unlock rule: treat badge-only rooms as trip-specific advantages, not stable preferences."},
            {"doc_id": "stale:partner_social_default", "memory_type": "stale_policy", "status": "stale", "city": None, "family": None, "traveler_id": None, "tags": ["old_social_bundle_default"], "text": "Old social note: after-work bundles used to be treated as broadly good. Retire that assumption when prep quality, rain, or stakeholder fit dominates."},
            {"doc_id": "stale:bundle_discount_always_wins", "memory_type": "stale_policy", "status": "stale", "city": None, "family": None, "traveler_id": None, "tags": ["old_bundle_discount_absolute"], "text": "Old discount note: a nominal discount always beats a quieter fit. This is stale and should be retired when the bundle is conditional or noisy."},
            {"doc_id": "stale:late_checkin_irrelevant", "memory_type": "stale_policy", "status": "stale", "city": None, "family": None, "traveler_id": None, "tags": ["late_checkin_irrelevant"], "text": "Outdated late-checkin note: arrival time does not affect dinner viability. Retire this whenever a bundle has a real cutoff."},
        ]
    )
    for promo in partner_promotions():
        extras.append({**promo, "memory_type": "partner_promo", "status": "current", "traveler_id": None, "tags": promo.get("benefit_tags", [])})
    for event in event_calendar():
        extras.append({**event, "memory_type": "event_ops", "status": "current", "family": None, "traveler_id": None, "tags": event.get("tags", [])})
    for loyalty in loyalty_programs().values():
        extras.append({**loyalty, "memory_type": "loyalty", "status": "current", "city": None, "family": None, "tags": loyalty.get("bonus_tags", [])})
    for stakeholder in stakeholder_briefs().values():
        extras.append({**stakeholder, "memory_type": "stakeholder", "status": "current", "city": None, "family": None, "traveler_id": None, "tags": stakeholder.get("tags", [])})
    for constraint in booking_constraints():
        extras.append({**constraint, "memory_type": "booking_constraint", "status": "current", "traveler_id": None})
    for dep in option_dependencies():
        extras.append({**dep, "memory_type": "dependency", "status": "current", "family": None, "traveler_id": None})
    return _merge_unique(seed, extras, "doc_id")


def rejected_options_memory() -> List[Dict[str, Any]]:
    seed = _load_seed("rejected_options_memory.json", [])
    extras = [
        {"memory_id": "reject:OSA:restaurant:RS3005", "city": "OSA", "family": "board_visit", "traveler_id": None, "kind": "restaurant", "option_id": "RS3005", "reason_key": "rejected_restaurant_for_vibe", "text": "RS3005 can be fine casually, but the room tone is too casual for a board-facing dinner."},
        {"memory_id": "reject:TPE:restaurant:RS4002", "city": "TPE", "family": "partner_summit", "traveler_id": None, "kind": "restaurant", "option_id": "RS4002", "reason_key": "rejected_restaurant_for_vibe", "text": "RS4002 is too exposed to late crowd energy for a summit dinner."},
        {"memory_id": "reject:SIN:restaurant:RS2005", "city": "SIN", "family": "roadshow_trip", "traveler_id": None, "kind": "restaurant", "option_id": "RS2005", "reason_key": "rejected_restaurant_for_vibe", "text": "RS2005 can look polished, but the transfer overhead makes it wrong for a rainy roadshow night unless the client optics are unusually important."},
    ]
    return _merge_unique(seed, extras, "memory_id")


def write_support_assets() -> None:
    write_json("inventory_flights.json", inventory_flights())
    write_json("inventory_hotels.json", inventory_hotels())
    write_json("inventory_restaurants.json", inventory_restaurants())
    write_json("inventory_activities.json", inventory_activities())
    write_json("profile_briefs.json", profile_briefs())
    write_json("venue_briefs.json", venue_briefs())
    write_json("city_ops_notes.json", city_ops_notes())
    write_json("partner_promotions.json", partner_promotions())
    write_json("event_calendar.json", event_calendar())
    write_json("loyalty_programs.json", loyalty_programs())
    write_json("stakeholder_briefs.json", stakeholder_briefs())
    write_json("booking_constraints.json", booking_constraints())
    write_json("option_dependencies.json", option_dependencies())
    write_json("memory_corpus.json", memory_corpus())
    write_json("rejected_options_memory.json", rejected_options_memory())


def _time_to_minutes(value: str | None) -> int | None:
    if not value or ":" not in value:
        return None
    hh, mm = value.split(":", 1)
    return int(hh) * 60 + int(mm)


def preferred_docs(
    episode: Dict[str, Any],
    scenario: Dict[str, Any],
    matched: Dict[str, Any],
) -> Tuple[List[str], List[str], List[str]]:
    city = episode["city"]
    family = episode["family"]
    traveler_id = episode["traveler_id"]
    required = [
        f"profile:{traveler_id}",
        f"venue:{city}_{family}",
        f"city_ops:{city}",
        "heuristic:lean_context_policy",
        "heuristic:rejected_option_memory",
    ]
    if scenario.get("partner_bundle"):
        required.append("heuristic:partner_bundle_reasoning")
    if scenario.get("refund_risk"):
        required.append("heuristic:refundable_schedule_risk")
    if scenario.get("stakeholder_ids"):
        required.append("heuristic:stakeholder_balance")
        required.extend(scenario["stakeholder_ids"])
    if scenario.get("badge_available"):
        required.extend(["heuristic:badge_unlock_logic", "constraint:badge_private_room_access"])
    if scenario.get("late_arrival_risk"):
        required.append("constraint:late_arrival_voids_bundle")
    if matched.get("promo_id"):
        required.append(matched["promo_id"])
    if matched.get("dependency_id"):
        required.append(matched["dependency_id"])
    required.extend(matched.get("event_ids", []))
    if scenario.get("loyalty_focus") and scenario.get("traveler_has_loyalty_doc"):
        required.append(scenario["traveler_has_loyalty_doc"])

    stale = ["stale:budget_cap_archive"]
    if scenario.get("airport_priority"):
        stale.append("stale:local_character_default")
    if scenario.get("chain_exception"):
        stale.append("stale:avoid_chain_hotels_absolute")
    if scenario.get("rainy"):
        stale.append("stale:dry_weather_ops_assumption")
    if scenario.get("partner_bundle"):
        stale.append("stale:bundle_discount_always_wins")
    if scenario.get("event_disruption"):
        stale.append("stale:partner_social_default")
    if scenario.get("late_arrival_risk"):
        stale.append("stale:late_checkin_irrelevant")

    distractors = {
        "OSA": ["playbook:OSA_business_travel_afterwork_social", "distractor:traveler_consultant_weekend_food_crawl", "stale:OSA_shinsekai_shortcut"],
        "TPE": ["playbook:TPE_conference_afterparty", "distractor:traveler_exec_scenic_photo_walk", "stale:TPE_scenic_outer_transfer_note"],
        "SIN": ["playbook:SIN_marina_client_evening", "stale:SIN_clarke_quay_social_note", "distractor:traveler_research_airport_lounge_habit"],
    }[city]
    if scenario.get("partner_bundle"):
        distractors = distractors + ["stale:partner_social_default"]
    return sorted(set(required)), sorted(set(stale)), sorted(set(distractors))


def _promo_is_valid(promo: Dict[str, Any], flight: Dict[str, Any], scenario: Dict[str, Any]) -> bool:
    arrival_before = _time_to_minutes(promo.get("arrival_before"))
    arrival_time = _time_to_minutes(flight.get("arrival_time"))
    if arrival_before is not None and arrival_time is not None and arrival_time > arrival_before:
        return False
    if promo.get("badge_required") and not scenario.get("badge_available"):
        return False
    return True


def _bundle_matches(promo: Dict[str, Any], hotel: Dict[str, Any], restaurant: Dict[str, Any], activity: Dict[str, Any], episode: Dict[str, Any]) -> bool:
    if promo.get("city") != episode["city"]:
        return False
    if promo.get("family") and promo.get("family") != episode["family"]:
        return False
    if promo.get("hotel_id") and promo.get("hotel_id") != hotel["hotel_id"]:
        return False
    if promo.get("restaurant_id") and promo.get("restaurant_id") != restaurant["restaurant_id"]:
        return False
    if promo.get("activity_id") and promo.get("activity_id") != activity["activity_id"]:
        return False
    return True


def score_bundle(env: RTLSemanticEnv, episode: Dict[str, Any], scenario: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    flights = env.search_flights(episode["origin"], episode["city"])
    hotels = env.search_hotels(episode["city"])
    restaurants = env.search_restaurants(episode["city"])
    activities = env.search_activities(episode["city"])
    venue = env.get_venue_brief(episode["city"], episode["family"])
    ops = env.get_city_ops_notes(episode["city"])
    preferred_zone = venue["preferred_zone"]
    avoid_zone = ops["avoid_zone"]

    promos = env.get_partner_promotions(city=episode["city"])
    events = env.get_event_calendar(episode["city"])
    constraints = env.get_booking_constraints(city=episode["city"], family=episode["family"])
    deps = env.get_option_dependencies(episode["city"])
    loyalty = env.get_loyalty_profile(episode["traveler_id"]) or {}
    stakeholders = [env.get_stakeholder_brief(sid) for sid in scenario.get("stakeholder_ids", []) if env.get_stakeholder_brief(sid)]

    best_score = float("-inf")
    best_bundle: Dict[str, Any] | None = None
    best_meta: Dict[str, Any] | None = None

    for flight in flights:
        for hotel in hotels:
            for restaurant in restaurants:
                for activity in activities:
                    score = 0.0
                    effective_discount = 0
                    matched_promo = None
                    matched_dep = None
                    event_hits: List[str] = []
                    chosen_tags = set()

                    if not flight.get("red_eye"):
                        score += 4.0
                    else:
                        score -= 7.0
                    if "meeting_safe" in flight.get("semantic_tags", []):
                        score += 3.5
                    if flight.get("stops", 0) == 0:
                        score += 1.0
                    if scenario.get("refund_risk"):
                        score += 2.2 if flight.get("refundable") else -4.4
                    score -= flight["fare_total"] / 320000.0

                    score += max(hotel.get("quiet_score", 0) - 7.5, 0.0) * 2.0
                    if hotel["zone"] == preferred_zone:
                        score += 3.0
                    if hotel["zone"] == avoid_zone:
                        score -= 2.5
                    score += hotel.get("airport_access_score", 0.0) * (0.55 if scenario.get("airport_priority") else 0.18)
                    if not hotel.get("chain"):
                        score += 0.7
                    if scenario.get("chain_exception") and hotel.get("chain"):
                        score += 1.2
                    score -= hotel["nightly_price"] * episode["nights"] / 330000.0
                    if scenario.get("loyalty_focus") and hotel["hotel_id"] in loyalty.get("hotel_partner_ids", []):
                        score += 2.0
                        effective_discount += int(loyalty.get("hotel_credit_krw", 0))
                        chosen_tags.add("loyalty_bundle_value")

                    if restaurant["area"] == preferred_zone:
                        score += 2.0
                    if restaurant["area"] == avoid_zone:
                        score -= 3.0
                    score += restaurant.get("quiet_score", 0) * 0.35
                    if "vegan" in restaurant.get("dietary_flags", []) or "vegan_preorder" in restaurant.get("dietary_flags", []):
                        score += 0.8
                    if scenario.get("teammate_vegan"):
                        if any(flag in restaurant.get("dietary_flags", []) for flag in ["vegan", "vegan_preorder"]):
                            score += 2.2
                            chosen_tags.add("team_dietary_flex")
                        else:
                            score -= 5.0
                    if scenario.get("client_dinner"):
                        score += restaurant.get("client_ready_score", 0) * 0.5
                    else:
                        score += 0.2 * restaurant.get("quiet_score", 0)
                    score -= restaurant["price_level"] * 0.15
                    if restaurant.get("private_room") and scenario.get("client_dinner"):
                        score += 1.1
                        chosen_tags.add("private_room_bonus")

                    if scenario.get("rainy"):
                        score += 3.0 if activity.get("indoor") else -4.0
                    if activity["location_zone"] == preferred_zone:
                        score += 1.6
                    if activity["location_zone"] == avoid_zone:
                        score -= 0.7
                    if activity.get("indoor"):
                        score += 0.4

                    for promo in promos:
                        if not _bundle_matches(promo, hotel, restaurant, activity, episode):
                            continue
                        if _promo_is_valid(promo, flight, scenario):
                            promo_bonus = float(promo.get("score_bonus", 0.0))
                            score += promo_bonus
                            effective_discount += int(promo.get("discount_krw", 0))
                            chosen_tags.update(promo.get("benefit_tags", []))
                            matched_promo = promo
                        else:
                            score -= 4.5

                    for dep in deps:
                        if dep.get("hotel_id") == hotel["hotel_id"] and dep.get("restaurant_id") == restaurant["restaurant_id"] and dep.get("activity_id") in {None, activity["activity_id"]}:
                            score += float(dep.get("score_bonus", 0.0))
                            chosen_tags.update(dep.get("tags", []))
                            matched_dep = dep

                    for event in events:
                        zone_hit = event.get("affected_zone") in {hotel.get("zone"), restaurant.get("area"), activity.get("location_zone")}
                        option_blocked = event.get("blocked_restaurant_id") == restaurant["restaurant_id"]
                        if option_blocked:
                            score -= 8.0
                            event_hits.append(event["event_id"])
                            continue
                        if event.get("effect_type") == "noise_surge" and zone_hit:
                            score -= 2.5
                            if restaurant.get("private_room"):
                                score += 1.0
                            event_hits.append(event["event_id"])
                        elif event.get("effect_type") == "transit_congestion" and zone_hit:
                            if hotel.get("airport_access_score", 0.0) >= 8.5 or hotel.get("zone") == "blue_line_corridor":
                                score += 1.5
                            else:
                                score -= 2.0
                            event_hits.append(event["event_id"])
                        elif event.get("effect_type") == "rain_transfer" and zone_hit:
                            if activity.get("indoor") and hotel.get("meeting_shuttle"):
                                score += 2.2
                            else:
                                score -= 2.4
                            event_hits.append(event["event_id"])
                        elif event.get("effect_type") in {"private_room_boost", "client_optics"} and zone_hit:
                            score += 1.2
                            event_hits.append(event["event_id"])

                    if scenario.get("badge_available") and (restaurant.get("badge_only") or activity.get("badge_only")):
                        score += 1.8
                        chosen_tags.add("conference_badge_access")
                    if not scenario.get("badge_available") and (restaurant.get("badge_only") or activity.get("badge_only")):
                        score -= 6.5

                    total_cost = (
                        flight["fare_total"]
                        + hotel["nightly_price"] * episode["nights"]
                        + restaurant["price_level"] * 25000
                        + activity["price"]
                        - effective_discount
                    )
                    if total_cost <= episode["budget_total"]:
                        score += 1.0
                    else:
                        score -= (total_cost - episode["budget_total"]) / 120000.0

                    chosen_tags.add("low_friction_transit") if hotel.get("airport_access_score", 0.0) >= 8.5 else None
                    if scenario.get("rainy") and activity.get("indoor"):
                        chosen_tags.add("weather_safe_backup")
                    if scenario.get("refund_risk") and flight.get("refundable"):
                        chosen_tags.add("refundable_priority")

                    if best_bundle is None or score > best_score:
                        best_score = score
                        best_bundle = {"flight": flight, "hotel": hotel, "restaurant": restaurant, "activity": activity}
                        best_meta = {
                            "preferred_zone": preferred_zone,
                            "avoid_zone": avoid_zone,
                            "promo_id": matched_promo.get("promo_id") if matched_promo else None,
                            "dependency_id": matched_dep.get("dependency_id") if matched_dep else None,
                            "event_ids": sorted(set(event_hits)),
                            "effective_discount": effective_discount,
                            "chosen_tags": sorted(chosen_tags),
                            "total_cost": total_cost,
                        }

    assert best_bundle is not None and best_meta is not None
    return best_bundle, best_meta


def build_gold(env: RTLSemanticEnv, episode: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
    bundle, meta = score_bundle(env, episode, scenario)
    required_docs, stale_docs, distractor_docs = preferred_docs(episode, scenario, meta)
    should_retrieve = ["avoid_red_eye", "prefer_quiet_hotel", "low_friction_transit"]
    if scenario.get("airport_priority"):
        should_retrieve.append("prefer_airport_access")
    else:
        should_retrieve.append("local_character_if_safe")
    if scenario.get("client_dinner"):
        should_retrieve.append("client_dinner_polished")
    if scenario.get("rainy"):
        should_retrieve.append("weather_safe_backup")
    if scenario.get("chain_exception"):
        should_retrieve.append("chain_exception_this_trip")
    if scenario.get("partner_bundle"):
        should_retrieve.extend(["bundle_discount_value", "private_room_bonus"])
    if scenario.get("refund_risk"):
        should_retrieve.append("refundable_priority")
    if scenario.get("teammate_vegan"):
        should_retrieve.append("team_dietary_flex")
    if scenario.get("badge_available"):
        should_retrieve.append("conference_badge_access")
    if scenario.get("late_arrival_risk"):
        should_retrieve.append("late_checkin_risk")
    if scenario.get("loyalty_focus"):
        should_retrieve.append("loyalty_bundle_value")
    if any(tag == "shuttle_bundle" for tag in meta.get("chosen_tags", [])):
        should_retrieve.append("shuttle_bundle")

    should_retire = ["old_budget_cap"]
    if scenario.get("airport_priority"):
        should_retire.append("local_character_if_safe")
    if scenario.get("rainy"):
        should_retire.append("old_weather_assumption")
    if scenario.get("chain_exception"):
        should_retire.append("avoid_chain_hotels_stable")
    if scenario.get("partner_bundle"):
        should_retire.append("old_bundle_discount_absolute")
    if scenario.get("event_disruption"):
        should_retire.append("old_social_bundle_default")
    if scenario.get("late_arrival_risk"):
        should_retire.append("late_checkin_irrelevant")

    required_hard = ["under_budget", "meeting_safe_arrival", "quiet_hotel", "zone_coherence"] + (["weather_safe_activity"] if scenario.get("rainy") else [])
    if scenario.get("teammate_vegan"):
        required_hard.append("team_dietary_support")
    if scenario.get("refund_risk"):
        required_hard.append("refund_safe")
    if scenario.get("partner_bundle"):
        required_hard.append("bundle_dependency_valid")

    soft_tags = ["quiet", "low_friction"]
    soft_tags += ["easy_airport_access"] if scenario.get("airport_priority") else ["local_character"]
    soft_tags += ["client_ready"] if scenario.get("client_dinner") else ["vegan_friendly"]
    soft_tags += ["weather_safe"] if scenario.get("rainy") else []

    retire_rules = ["old_budget_cap"]
    if scenario.get("airport_priority"):
        retire_rules.append("old_local_character_priority")
    if scenario.get("rainy"):
        retire_rules.append("old_weather_assumption")
    if scenario.get("chain_exception"):
        retire_rules.append("old_chain_absolute_rule")
    if scenario.get("partner_bundle"):
        retire_rules.append("old_bundle_discount_absolute")
    if scenario.get("event_disruption"):
        retire_rules.append("old_social_bundle_default")
    if scenario.get("late_arrival_risk"):
        retire_rules.append("late_checkin_irrelevant")

    return {
        "required_hard": required_hard,
        "soft_tags": soft_tags,
        "acceptable_flights": [bundle["flight"]["flight_id"]],
        "acceptable_hotels": [bundle["hotel"]["hotel_id"]],
        "good_restaurants": [bundle["restaurant"]["restaurant_id"]],
        "good_activities": [bundle["activity"]["activity_id"]],
        "required_docs": required_docs,
        "stale_docs_to_retire": stale_docs,
        "distractor_docs_to_avoid": distractor_docs,
        "should_retrieve": sorted(set(should_retrieve)),
        "should_retire": sorted(set(should_retire)),
        "should_remember_rejected": ["rejected_hotel_for_noise", "rejected_flight_for_red_eye", "rejected_restaurant_for_vibe"],
        "required_spoken_rules": {
            "must_remember": ["quiet_matters"] + (["client_ready_dinner"] if scenario.get("client_dinner") else []),
            "forbidden": ["red_eye", "loud_after_10pm"],
            "one_off_only": [x for x in ["chain_ok_this_trip" if scenario.get("chain_exception") else None, "airport_access_more_important_now" if scenario.get("airport_priority") else None] if x],
            "retire": sorted(set(retire_rules)),
            "do_not_reconsider": ["noise_rejected_hotel", "wrong_vibe_restaurant"],
            "keep_context_lean": ["relevant_only"],
        },
        "preferred_zone": meta["preferred_zone"],
        "avoid_zone": meta["avoid_zone"],
        "episodic_exceptions": [x for x in ["airport_access_more_important_now" if scenario.get("airport_priority") else None, "chain_ok_this_trip" if scenario.get("chain_exception") else None] if x],
    }


def spoken_turns(scenario: Dict[str, Any], dense: bool) -> List[str]:
    turns = [
        RULE_TEMPLATES["profile_use"],
        RULE_TEMPLATES["must_remember_quiet"],
        RULE_TEMPLATES["forbid_red_eye"],
        RULE_TEMPLATES["forbid_loud_zone"],
        RULE_TEMPLATES["retrieval_discipline"],
        RULE_TEMPLATES["remember_rejection_noise"],
        RULE_TEMPLATES["remember_rejection_vibe"],
    ]
    if scenario.get("client_dinner"):
        turns.extend([RULE_TEMPLATES["must_remember_client_ready"], RULE_TEMPLATES["client_dinner_now"]])
    if scenario.get("airport_priority"):
        turns.extend([RULE_TEMPLATES["airport_priority_once"], RULE_TEMPLATES["retire_old_local_char"]])
    if scenario.get("chain_exception"):
        turns.append(RULE_TEMPLATES["allow_chain_once"])
    if scenario.get("rainy"):
        turns.append(RULE_TEMPLATES["weather_shift"])
    if scenario.get("partner_bundle"):
        turns.append(RULE_TEMPLATES["bundle_value"])
    if scenario.get("badge_available"):
        turns.append(RULE_TEMPLATES["badge_unlock"])
    if scenario.get("refund_risk"):
        turns.append(RULE_TEMPLATES["refund_risk"])
    if scenario.get("late_arrival_risk"):
        turns.append(RULE_TEMPLATES["late_arrival"])
    if scenario.get("teammate_vegan"):
        turns.append(RULE_TEMPLATES["teammate_vegan"])
    if scenario.get("loyalty_focus"):
        turns.append(RULE_TEMPLATES["loyalty_value"])
    if scenario.get("event_disruption"):
        turns.append(RULE_TEMPLATES["event_disruption"])
    turns.extend([RULE_TEMPLATES["retire_old_budget"], RULE_TEMPLATES["compressed_followup"]])
    return turns if dense else turns[: max(8, len(turns) - 3)]




def derive_task_family(scenario: Dict[str, Any]) -> str:
    if scenario.get("teammate_vegan") and scenario.get("client_dinner"):
        return "stakeholder_tradeoff"
    if scenario.get("loyalty_focus") and scenario.get("partner_bundle"):
        return "loyalty_effective_value"
    if scenario.get("refund_risk") and scenario.get("late_arrival_risk"):
        return "refund_volatility"
    if scenario.get("partner_bundle") and scenario.get("late_arrival_risk"):
        return "bundle_cutoff_reasoning"
    if scenario.get("badge_available") and scenario.get("partner_bundle"):
        return "badge_unlocked_bundle"
    if scenario.get("rainy") and scenario.get("event_disruption"):
        return "weather_disruption"
    if scenario.get("event_disruption"):
        return "event_sensitive_replanning"
    if scenario.get("airport_priority") and scenario.get("chain_exception"):
        return "one_off_airport_override"
    if scenario.get("client_dinner"):
        return "client_ready_dinner"
    if scenario.get("teammate_vegan"):
        return "dietary_flex"
    return "core_memory_replanning"



def validate_episode_bank(env: RTLSemanticEnv, public_eps: List[Dict[str, Any]], hidden_inputs: List[Dict[str, Any]], hidden_gold: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if len(public_eps) != 20:
        raise ValueError(f"Expected 20 public episodes, found {len(public_eps)}")
    if len(hidden_inputs) != 30:
        raise ValueError(f"Expected 30 hidden episodes, found {len(hidden_inputs)}")

    seen_trip_ids = set()
    families = set()
    rich_hook_count = 0

    def turn_signature(episode: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
        return tuple((turn["speaker"], turn["text"]) for turn in episode.get("turns", []))

    def duplicate_prompt_groups(episodes: List[Dict[str, Any]]) -> List[List[str]]:
        groups: Dict[Tuple[Tuple[str, str], ...], List[str]] = {}
        for episode in episodes:
            groups.setdefault(turn_signature(episode), []).append(episode["trip_id"])
        return [trip_ids for trip_ids in groups.values() if len(trip_ids) > 1]

    def check_episode(episode: Dict[str, Any], gold: Dict[str, Any], split: str) -> None:
        nonlocal rich_hook_count
        trip_id = episode["trip_id"]
        if trip_id in seen_trip_ids:
            raise ValueError(f"Duplicate trip_id: {trip_id}")
        seen_trip_ids.add(trip_id)
        scenario = episode.get("scenario_state", {})
        families.add(episode.get("benchmark_family") or derive_task_family(scenario))
        hooks = episode.get("scenario_hooks", {})
        if hooks.get("event_sensitive") or hooks.get("bundle_watch") or hooks.get("schedule_volatility") == "high":
            rich_hook_count += 1

        accepted_sets = [gold.get("acceptable_flights", []), gold.get("acceptable_hotels", []), gold.get("good_restaurants", []), gold.get("good_activities", [])]
        if any(not rows for rows in accepted_sets):
            raise ValueError(f"{trip_id} on {split} has an empty acceptable bundle component")

        required_docs = gold.get("required_docs", [])
        if scenario.get("partner_bundle") and not (
            any(doc.startswith(("promo:", "dependency:", "constraint:")) for doc in required_docs)
            or "bundle_discount_value" in gold.get("should_retrieve", [])
        ):
            raise ValueError(f"{trip_id} uses partner_bundle without bundle reasoning evidence")
        if scenario.get("badge_available") and not any("badge" in doc for doc in required_docs):
            raise ValueError(f"{trip_id} uses badge_available without badge-linked docs")
        if scenario.get("event_disruption") and not any(doc.startswith("event:") for doc in required_docs):
            raise ValueError(f"{trip_id} uses event_disruption without event docs")
        if scenario.get("refund_risk") and "refund_safe" not in gold.get("required_hard", []):
            raise ValueError(f"{trip_id} uses refund_risk without refund_safe hard requirement")
        if scenario.get("teammate_vegan") and "team_dietary_support" not in gold.get("required_hard", []):
            raise ValueError(f"{trip_id} uses teammate_vegan without team_dietary_support")
        if scenario.get("late_arrival_risk") and "late_checkin_risk" not in gold.get("should_retrieve", []):
            raise ValueError(f"{trip_id} uses late_arrival_risk without late_checkin_risk retrieval target")

    for episode in public_eps:
        check_episode(episode, episode["gold"], "public")
    for episode in hidden_inputs:
        gold = hidden_gold.get(episode["trip_id"])
        if not gold:
            raise ValueError(f"Missing hidden gold for {episode['trip_id']}")
        check_episode(episode, gold, "hidden")

    public_family_counts: Dict[str, int] = {}
    hidden_family_counts: Dict[str, int] = {}
    for episode in public_eps:
        public_family_counts[episode["benchmark_family"]] = public_family_counts.get(episode["benchmark_family"], 0) + 1
    for episode in hidden_inputs:
        hidden_family_counts[episode["benchmark_family"]] = hidden_family_counts.get(episode["benchmark_family"], 0) + 1

    public_duplicates = duplicate_prompt_groups(public_eps)
    hidden_duplicates = duplicate_prompt_groups(hidden_inputs)

    if len(families) < 7:
        raise ValueError(f"Expected at least 7 benchmark families, found {len(families)}")
    if rich_hook_count < 18:
        raise ValueError(f"Expected at least 18 rich-hook tasks, found {rich_hook_count}")
    if public_duplicates:
        raise ValueError(f"Public split contains exact duplicate prompt groups: {public_duplicates}")
    if hidden_duplicates:
        raise ValueError(f"Hidden split contains exact duplicate prompt groups: {hidden_duplicates}")
    if public_family_counts and max(public_family_counts.values()) > 7:
        raise ValueError(f"Public family balance is too concentrated: {public_family_counts}")
    if hidden_family_counts and max(hidden_family_counts.values()) > 10:
        raise ValueError(f"Hidden family balance is too concentrated: {hidden_family_counts}")

    return {
        "public_count": len(public_eps),
        "hidden_count": len(hidden_inputs),
        "family_count": len(families),
        "benchmark_families": sorted(families),
        "rich_hook_tasks": rich_hook_count,
        "public_family_counts": public_family_counts,
        "hidden_family_counts": hidden_family_counts,
        "public_exact_duplicate_prompt_groups": public_duplicates,
        "hidden_exact_duplicate_prompt_groups": hidden_duplicates,
        "public_tier_counts": {
            tier: sum(1 for episode in public_eps if episode["difficulty_tier"] == tier)
            for tier in ["easy", "medium", "hard"]
        },
        "hidden_tier_counts": {
            tier: sum(1 for episode in hidden_inputs if episode["difficulty_tier"] == tier)
            for tier in ["easy", "medium", "hard"]
        },
    }




def build_task_catalog(public_eps: List[Dict[str, Any]], hidden_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split, episodes in (("public", public_eps), ("hidden", hidden_inputs)):
        for episode in episodes:
            scenario = episode.get("scenario_state", {})
            rows.append({
                "trip_id": episode["trip_id"],
                "split": split,
                "difficulty_tier": episode.get("difficulty_tier"),
                "family": episode.get("family"),
                "benchmark_family": episode.get("benchmark_family") or derive_task_family(scenario),
                "city": episode.get("city"),
                "traveler_id": episode.get("traveler_id"),
                "scenario_hooks": episode.get("scenario_hooks", {}),
                "scenario_state": scenario,
                "retiered_after_rewrite": False,
            })
    return rows


PUBLIC_SPECS: List[Dict[str, Any]] = [
    {"tier": "easy", "route": "OSA_business_travel", "traveler_id": "traveler_consultant", "airport_priority": False, "rainy": False, "client_dinner": False, "chain_exception": False, "partner_bundle": False, "event_disruption": False, "badge_available": False, "refund_risk": False, "loyalty_focus": False, "late_arrival_risk": False, "teammate_vegan": False, "stakeholder_ids": []},
    {"tier": "easy", "route": "TPE_conference_trip", "traveler_id": "traveler_exec", "airport_priority": False, "rainy": False, "client_dinner": True, "chain_exception": False, "partner_bundle": False, "event_disruption": False, "badge_available": False, "refund_risk": False, "loyalty_focus": False, "late_arrival_risk": False, "teammate_vegan": False, "stakeholder_ids": ["stakeholder:client_polished"]},
    {"tier": "medium", "route": "OSA_business_travel", "traveler_id": "traveler_ops", "airport_priority": True, "rainy": False, "client_dinner": False, "chain_exception": False, "partner_bundle": True, "event_disruption": False, "badge_available": False, "refund_risk": True, "loyalty_focus": True, "late_arrival_risk": True, "teammate_vegan": False, "stakeholder_ids": []},
    {"tier": "medium", "route": "OSA_business_travel", "traveler_id": "traveler_consultant", "airport_priority": True, "rainy": False, "client_dinner": False, "chain_exception": False, "partner_bundle": True, "event_disruption": False, "badge_available": False, "refund_risk": False, "loyalty_focus": False, "late_arrival_risk": True, "teammate_vegan": False, "stakeholder_ids": []},
    {"tier": "medium", "route": "SIN_business_travel", "traveler_id": "traveler_consultant", "airport_priority": False, "rainy": True, "client_dinner": False, "chain_exception": False, "partner_bundle": False, "event_disruption": True, "badge_available": False, "refund_risk": False, "loyalty_focus": False, "late_arrival_risk": False, "teammate_vegan": False, "stakeholder_ids": []},
    {"tier": "medium", "route": "TPE_partner_summit", "traveler_id": "traveler_sales", "airport_priority": False, "rainy": False, "client_dinner": True, "chain_exception": False, "partner_bundle": True, "event_disruption": True, "badge_available": True, "refund_risk": False, "loyalty_focus": False, "late_arrival_risk": False, "teammate_vegan": True, "stakeholder_ids": ["stakeholder:teammate_vegan", "stakeholder:client_polished"]},
    {"tier": "medium", "route": "TPE_partner_summit", "traveler_id": "traveler_exec", "airport_priority": True, "rainy": False, "client_dinner": True, "chain_exception": True, "partner_bundle": True, "event_disruption": True, "badge_available": True, "refund_risk": True, "loyalty_focus": False, "late_arrival_risk": False, "teammate_vegan": True, "stakeholder_ids": ["stakeholder:teammate_vegan", "stakeholder:client_polished"]},
    {"tier": "medium", "route": "SIN_business_travel", "traveler_id": "traveler_research", "airport_priority": True, "rainy": False, "client_dinner": False, "chain_exception": True, "partner_bundle": False, "event_disruption": False, "badge_available": False, "refund_risk": True, "loyalty_focus": False, "late_arrival_risk": True, "teammate_vegan": False, "stakeholder_ids": ["stakeholder:ops_host"]},
    {"tier": "medium", "route": "OSA_board_visit", "traveler_id": "traveler_sales", "airport_priority": False, "rainy": False, "client_dinner": True, "chain_exception": False, "partner_bundle": False, "event_disruption": False, "badge_available": False, "refund_risk": False, "loyalty_focus": True, "late_arrival_risk": False, "teammate_vegan": False, "stakeholder_ids": ["stakeholder:client_polished"]},
    {"tier": "hard", "route": "TPE_conference_trip", "traveler_id": "traveler_exec", "airport_priority": True, "rainy": False, "client_dinner": False, "chain_exception": True, "partner_bundle": True, "event_disruption": True, "badge_available": True, "refund_risk": True, "loyalty_focus": False, "late_arrival_risk": True, "teammate_vegan": False, "stakeholder_ids": []},
    {"tier": "hard", "route": "SIN_roadshow_trip", "traveler_id": "traveler_ops", "airport_priority": True, "rainy": True, "client_dinner": False, "chain_exception": True, "partner_bundle": True, "event_disruption": True, "badge_available": False, "refund_risk": False, "loyalty_focus": True, "late_arrival_risk": True, "teammate_vegan": False, "stakeholder_ids": ["stakeholder:ops_host"]},
    {"tier": "hard", "route": "OSA_board_visit", "traveler_id": "traveler_sales", "airport_priority": False, "rainy": True, "client_dinner": True, "chain_exception": False, "partner_bundle": True, "event_disruption": True, "badge_available": False, "refund_risk": True, "loyalty_focus": False, "late_arrival_risk": True, "teammate_vegan": True, "stakeholder_ids": ["stakeholder:teammate_vegan", "stakeholder:client_polished"]},
    {"tier": "hard", "route": "TPE_partner_summit", "traveler_id": "traveler_exec", "airport_priority": True, "rainy": False, "client_dinner": False, "chain_exception": True, "partner_bundle": True, "event_disruption": True, "badge_available": True, "refund_risk": False, "loyalty_focus": False, "late_arrival_risk": False, "teammate_vegan": False, "stakeholder_ids": []},
    {"tier": "hard", "route": "SIN_business_travel", "traveler_id": "traveler_consultant", "airport_priority": True, "rainy": True, "client_dinner": False, "chain_exception": False, "partner_bundle": True, "event_disruption": True, "badge_available": False, "refund_risk": False, "loyalty_focus": False, "late_arrival_risk": False, "teammate_vegan": False, "stakeholder_ids": []},
    {"tier": "hard", "route": "OSA_business_travel", "traveler_id": "traveler_ops", "airport_priority": True, "rainy": False, "client_dinner": False, "chain_exception": True, "partner_bundle": False, "event_disruption": False, "badge_available": False, "refund_risk": False, "loyalty_focus": False, "late_arrival_risk": False, "teammate_vegan": False, "stakeholder_ids": []},
    {"tier": "hard", "route": "TPE_partner_summit", "traveler_id": "traveler_sales", "airport_priority": True, "rainy": False, "client_dinner": True, "chain_exception": True, "partner_bundle": True, "event_disruption": True, "badge_available": True, "refund_risk": True, "loyalty_focus": False, "late_arrival_risk": True, "teammate_vegan": True, "stakeholder_ids": ["stakeholder:teammate_vegan", "stakeholder:client_polished"]},
    {"tier": "hard", "route": "SIN_roadshow_trip", "traveler_id": "traveler_sales", "airport_priority": True, "rainy": True, "client_dinner": True, "chain_exception": False, "partner_bundle": True, "event_disruption": True, "badge_available": False, "refund_risk": True, "loyalty_focus": False, "late_arrival_risk": True, "teammate_vegan": False, "stakeholder_ids": ["stakeholder:client_polished"]},
    {"tier": "hard", "route": "OSA_board_visit", "traveler_id": "traveler_ops", "airport_priority": True, "rainy": False, "client_dinner": True, "chain_exception": True, "partner_bundle": False, "event_disruption": True, "badge_available": False, "refund_risk": False, "loyalty_focus": True, "late_arrival_risk": False, "teammate_vegan": False, "stakeholder_ids": ["stakeholder:ops_host", "stakeholder:client_polished"]},
    {"tier": "hard", "route": "TPE_conference_trip", "traveler_id": "traveler_consultant", "airport_priority": True, "rainy": False, "client_dinner": True, "chain_exception": False, "partner_bundle": False, "event_disruption": True, "badge_available": False, "refund_risk": False, "loyalty_focus": False, "late_arrival_risk": False, "teammate_vegan": True, "stakeholder_ids": ["stakeholder:teammate_vegan", "stakeholder:client_polished"]},
    {"tier": "hard", "route": "SIN_business_travel", "traveler_id": "traveler_ops", "airport_priority": True, "rainy": True, "client_dinner": False, "chain_exception": True, "partner_bundle": True, "event_disruption": True, "badge_available": False, "refund_risk": False, "loyalty_focus": False, "late_arrival_risk": True, "teammate_vegan": False, "stakeholder_ids": ["stakeholder:ops_host"]},
]


# Hidden split construction is TA-only.  The public generator intentionally
# exposes the benchmark grammar, scenario axes, and public episode generation
# philosophy, but not hidden episode-level specs or gold-label generation.

ROUTE_LOOKUP_BY_CITY_FAMILY = {(cfg["city"], cfg["family"]): route_name for route_name, cfg in ROUTE_CONFIGS.items()}





def _time_to_minutes(value: str | None) -> int | None:
    if not value or ":" not in value:
        return None
    try:
        hh, mm = value.split(":", 1)
        return int(hh) * 60 + int(mm)
    except Exception:
        return None


def _bundle_valid_for_candidate(env: RTLSemanticEnv, episode: Dict[str, Any], flight: Dict[str, Any] | None, hotel: Dict[str, Any] | None, restaurant: Dict[str, Any] | None, activity: Dict[str, Any] | None) -> bool:
    scenario_state = episode.get("scenario_state", {}) or {}
    if not scenario_state.get("partner_bundle"):
        return True
    matched_promos = env.get_partner_promotions(
        city=episode.get("city"),
        hotel_id=(hotel or {}).get("hotel_id"),
        restaurant_id=(restaurant or {}).get("restaurant_id"),
        activity_id=(activity or {}).get("activity_id"),
        family=episode.get("family"),
    )
    arrival_minutes = _time_to_minutes((flight or {}).get("arrival_time"))
    for promo in matched_promos:
        cutoff_minutes = _time_to_minutes(promo.get("arrival_before"))
        badge_ok = (not promo.get("badge_required")) or bool(scenario_state.get("badge_available"))
        arrival_ok = cutoff_minutes is None or arrival_minutes is None or arrival_minutes <= cutoff_minutes
        if badge_ok and arrival_ok:
            return True
    return False


def _candidate_total_cost(episode: Dict[str, Any], flight: Dict[str, Any] | None, hotel: Dict[str, Any] | None, restaurant: Dict[str, Any] | None, activity: Dict[str, Any] | None) -> int:
    total = 0
    if flight:
        total += int(flight["fare_total"])
    if hotel:
        total += int(hotel["nightly_price"]) * int(episode.get("nights", 1))
    if restaurant:
        total += int(restaurant["price_level"]) * 25000
    if activity:
        total += int(activity["price"])
    return total


def _candidate_hard_map(env: RTLSemanticEnv, episode: Dict[str, Any], gold: Dict[str, Any], flight: Dict[str, Any] | None, hotel: Dict[str, Any] | None, restaurant: Dict[str, Any] | None, activity: Dict[str, Any] | None, *, budget_override: int | None = None) -> Dict[str, bool]:
    scenario_state = episode.get("scenario_state", {}) or {}
    budget_total = int(budget_override if budget_override is not None else episode.get("budget_total", 0))
    hard = {key: False for key in ["under_budget", "meeting_safe_arrival", "quiet_hotel", "weather_safe_activity", "zone_coherence", "team_dietary_support", "refund_safe", "bundle_dependency_valid"]}

    if flight:
        hard["meeting_safe_arrival"] = ("meeting_safe" in flight.get("semantic_tags", [])) or ("airport_access_more_important_now" in gold.get("episodic_exceptions", []))
        hard["refund_safe"] = (not scenario_state.get("refund_risk")) or bool(flight.get("refundable"))
    else:
        hard["refund_safe"] = not scenario_state.get("refund_risk")
    if hotel:
        hard["quiet_hotel"] = "quiet" in hotel.get("semantic_tags", [])
    if restaurant:
        hard["team_dietary_support"] = (not scenario_state.get("teammate_vegan")) or any(flag in restaurant.get("dietary_flags", []) for flag in ["vegan", "vegan_preorder"])
    else:
        hard["team_dietary_support"] = not scenario_state.get("teammate_vegan")
    if activity:
        hard["weather_safe_activity"] = (episode.get("weather") != "rainy") or ("weather_safe" in activity.get("semantic_tags", []))

    hard["bundle_dependency_valid"] = _bundle_valid_for_candidate(env, episode, flight, hotel, restaurant, activity)
    hard["under_budget"] = _candidate_total_cost(episode, flight, hotel, restaurant, activity) <= budget_total
    zones = [item.get("zone") for item in [hotel] if item] + [item.get("area") for item in [restaurant] if item] + [item.get("location_zone") for item in [activity] if item]
    hard["zone_coherence"] = bool(zones) and all(zone != gold.get("avoid_zone") for zone in zones) and sum(zone == gold.get("preferred_zone") for zone in zones) >= 2
    return hard


def _public_candidate_score(env: RTLSemanticEnv, episode: Dict[str, Any], gold: Dict[str, Any], flight: Dict[str, Any], hotel: Dict[str, Any], restaurant: Dict[str, Any], activity: Dict[str, Any]) -> float:
    chosen_tags = set()
    for item in [flight, hotel, restaurant, activity]:
        chosen_tags.update(item.get("semantic_tags", []))
    for promo in env.get_partner_promotions(city=episode.get("city"), hotel_id=hotel.get("hotel_id"), restaurant_id=restaurant.get("restaurant_id"), activity_id=activity.get("activity_id"), family=episode.get("family")):
        chosen_tags.update(promo.get("benefit_tags", []))
    loyalty = env.get_loyalty_profile(episode.get("traveler_id")) or {}
    if hotel.get("hotel_id") in loyalty.get("hotel_partner_ids", []):
        chosen_tags.update(loyalty.get("bonus_tags", []))
    soft_tags = set(gold.get("soft_tags", []))
    semantic = len(chosen_tags & soft_tags) / max(len(soft_tags), 1)
    zone_bonus = sum(zone == gold.get("preferred_zone") for zone in [hotel.get("zone"), restaurant.get("area"), activity.get("location_zone")]) / 3.0
    cost = _candidate_total_cost(episode, flight, hotel, restaurant, activity)
    # Prefer semantically faithful and coherent plans; use lower cost only as a tie-breaker.
    return 10.0 * semantic + 1.5 * zone_bonus - (cost / 1_000_000.0)


def repair_public_episode_for_feasibility(env: RTLSemanticEnv, episode: Dict[str, Any]) -> None:
    """Keep the public task philosophy but ensure public gold is evaluator-feasible.

    Students are expected to run the public evaluator while iterating.  Therefore
    public labels must satisfy the same hard-feasibility contract as hidden
    labels: at least one gold-listed combination should achieve a full hard rate
    under the episode's current hard constraints.  This repair is intentionally
    conservative: it only raises impossible budgets, selects a feasible gold
    bundle, and drops bundle_dependency_valid from required_hard when no valid
    partner promo exists for any candidate under the current scenario state.
    """
    gold = episode.get("gold") or {}
    flights = env.search_flights(episode["origin"], episode["city"])
    hotels = env.search_hotels(episode["city"])
    restaurants = env.search_restaurants(episode["city"])
    activities = env.search_activities(episode["city"])
    original_required = list(gold.get("required_hard", []))

    def feasible_candidates(required: List[str]):
        out = []
        for flight in flights:
            for hotel in hotels:
                for restaurant in restaurants:
                    for activity in activities:
                        hard = _candidate_hard_map(env, episode, gold, flight, hotel, restaurant, activity, budget_override=10**9)
                        if all(hard.get(key, False) for key in required if key != "under_budget"):
                            out.append((flight, hotel, restaurant, activity))
        return out

    required = list(original_required)
    candidates = feasible_candidates(required)
    if not candidates and "bundle_dependency_valid" in required:
        required = [key for key in required if key != "bundle_dependency_valid"]
        candidates = feasible_candidates(required)
    if not candidates:
        # Last-resort safety valve for future public-spec edits: preserve the
        # task instead of emitting an impossible public answer key.
        candidates = feasible_candidates([key for key in required if key not in {"zone_coherence"}])
        if "zone_coherence" in required and candidates:
            required = [key for key in required if key != "zone_coherence"]
    if not candidates:
        raise RuntimeError(f"Could not repair public episode {episode.get('trip_id')} to a feasible gold bundle")

    best = max(candidates, key=lambda items: _public_candidate_score(env, episode, gold, *items))
    flight, hotel, restaurant, activity = best
    cost = _candidate_total_cost(episode, flight, hotel, restaurant, activity)
    if "under_budget" in required and cost > int(episode.get("budget_total", 0)):
        # Keep the budget meaningful but feasible; round to a clean 5,000 KRW.
        episode["budget_total"] = int(((cost + 14_999) // 5_000) * 5_000)

    gold["required_hard"] = required
    gold["acceptable_flights"] = [flight["flight_id"]]
    gold["acceptable_hotels"] = [hotel["hotel_id"]]
    gold["good_restaurants"] = [restaurant["restaurant_id"]]
    gold["good_activities"] = [activity["activity_id"]]
    episode["gold"] = gold

def materialize_episode(env: RTLSemanticEnv, spec: Dict[str, Any], trip_id: str, *, extra_turns: List[str] | None = None) -> Dict[str, Any]:
    cfg = ROUTE_CONFIGS[spec["route"]]
    dense_rules = spec["tier"] != "easy" or sum(bool(spec.get(key)) for key in ["partner_bundle", "event_disruption", "badge_available", "refund_risk", "loyalty_focus", "late_arrival_risk", "teammate_vegan"]) >= 3
    scenario = dict(spec)
    scenario["city"] = cfg["city"]
    scenario["family"] = cfg["family"]
    scenario["rainy"] = bool(spec.get("rainy"))
    scenario["traveler_has_loyalty_doc"] = loyalty_programs().get(spec["traveler_id"], {}).get("doc_id")
    budget = spec.get("budget_total_override")
    if budget is None:
        budget = cfg["base_budget"] - (40000 if spec["tier"] == "medium" else 80000 if spec["tier"] == "hard" else 0)
    episode = {
        "trip_id": trip_id,
        "difficulty_tier": spec["tier"],
        "benchmark_family": derive_task_family(scenario),
        "family": cfg["family"],
        "city": cfg["city"],
        "origin": cfg["origin"],
        "traveler_id": spec["traveler_id"],
        "nights": 2,
        "budget_total": budget,
        "meeting_zone": env.get_venue_brief(cfg["city"], cfg["family"])["preferred_zone"],
        "weather": "rainy" if scenario.get("rainy") else cfg["weather"],
        "spoken_rule_density": "dense" if dense_rules else "moderate",
        "scenario_hooks": {
            "stakeholders": scenario.get("stakeholder_ids", []),
            "badge_status": "active" if scenario.get("badge_available") else "inactive",
            "schedule_volatility": "high" if scenario.get("refund_risk") else "normal",
            "bundle_watch": bool(scenario.get("partner_bundle")),
            "late_arrival_risk": bool(scenario.get("late_arrival_risk")),
            "event_sensitive": bool(scenario.get("event_disruption")),
        },
        "scenario_state": {
            key: scenario.get(key)
            for key in [
                "airport_priority",
                "rainy",
                "client_dinner",
                "chain_exception",
                "partner_bundle",
                "event_disruption",
                "badge_available",
                "refund_risk",
                "loyalty_focus",
                "late_arrival_risk",
                "teammate_vegan",
                "stakeholder_ids",
            ]
        },
    }
    turns = BASE_TURNS[cfg["family"]] + spoken_turns(scenario, dense_rules)
    if extra_turns:
        turns = turns + extra_turns
    episode["turns"] = [{"speaker": "user", "text": text} for text in turns]
    episode["gold"] = build_gold(env, episode, scenario)
    repair_public_episode_for_feasibility(env, episode)
    return episode



def build_public_dataset(env: RTLSemanticEnv) -> List[Dict[str, Any]]:
    return [materialize_episode(env, spec, f"rtl7_public_{spec['tier']}_{idx:03d}") for idx, spec in enumerate(PUBLIC_SPECS, start=1)]



def _spec_from_public_episode(episode: Dict[str, Any]) -> Dict[str, Any]:
    scenario = dict(episode.get("scenario_state", {}))
    scenario["tier"] = episode["difficulty_tier"]
    scenario["route"] = ROUTE_LOOKUP_BY_CITY_FAMILY[(episode["city"], episode["family"])]
    scenario["traveler_id"] = episode["traveler_id"]
    return scenario



def build_hidden_split(public_eps: List[Dict[str, Any]], seed: int = 17) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Hidden split generation is TA-only.

    Hidden episodes follow the same high-level benchmark families and are
    validated for hard feasibility and contrastive replanning, but exact hidden
    specs, mutations, and gold labels are not part of the student-facing repo.
    """
    raise RuntimeError(
        "Hidden split generation is TA-only. Use final_project/ta_only/"
        "dynamic_travel_replanning/hidden_episode_generator.py in the staff repo."
    )


def build_and_write(seed: int = 17) -> None:
    """Regenerate student-facing public travel assets only.

    This preserves transparency about the public data generation philosophy while
    keeping TA-only hidden specs/gold separate.
    """
    write_support_assets()
    env = RTLSemanticEnv(HERE)
    public_eps = build_public_dataset(env)
    public_family_counts: Dict[str, int] = {}
    for episode in public_eps:
        public_family_counts[episode["benchmark_family"]] = public_family_counts.get(episode["benchmark_family"], 0) + 1
    validation = {
        "public_count": len(public_eps),
        "hidden_count": 30,
        "family_count": len(public_family_counts),
        "benchmark_families": sorted(public_family_counts),
        "rich_hook_tasks": sum(1 for e in public_eps if e.get("scenario_hooks", {}).get("event_sensitive") or e.get("scenario_hooks", {}).get("bundle_watch") or e.get("scenario_hooks", {}).get("schedule_volatility") == "high"),
        "public_family_counts": public_family_counts,
        "hidden_family_counts": "TA-only",
        "public_tier_counts": {tier: sum(1 for e in public_eps if e["difficulty_tier"] == tier) for tier in ["easy", "medium", "hard"]},
        "hidden_tier_counts": {"easy": 4, "medium": 13, "hard": 13},
        "public_gold_hard_feasibility": "repaired_against_student_evaluator",
    }
    write_json("episodes_public_example.json", public_eps)
    public_tier_counts = {tier: sum(1 for e in public_eps if e["difficulty_tier"] == tier) for tier in ["easy", "medium", "hard"]}
    write_json("tier_manifest.json", {
        "public_easy": [e["trip_id"] for e in public_eps if e["difficulty_tier"] == "easy"],
        "public_medium": [e["trip_id"] for e in public_eps if e["difficulty_tier"] == "medium"],
        "public_hard": [e["trip_id"] for e in public_eps if e["difficulty_tier"] == "hard"],
        "public_all": [e["trip_id"] for e in public_eps],
        "public_count": len(public_eps),
        "public_tier_counts": public_tier_counts,
        "hidden_count": 30,
        "hidden_tier_counts": {"easy": 4, "medium": 13, "hard": 13},
        "hidden_note": "Hidden episodes use the same high-level benchmark families, but exact hidden specs and gold labels are TA-only.",
    })
    write_json("task_catalog.json", build_task_catalog(public_eps, []))
    write_json("task_bank_validation.json", {
        **validation,
        "hidden_note": "Hidden validation is performed in the TA-only repository.",
        "hidden_count": 30,
        "hidden_tier_counts": {"easy": 4, "medium": 13, "hard": 13},
    })
    write_json("evaluation_tracks.json", {
        "tracks": {
            "public_full": [e["trip_id"] for e in public_eps],
            "public_easy": [e["trip_id"] for e in public_eps if e["difficulty_tier"] == "easy"],
            "public_medium": [e["trip_id"] for e in public_eps if e["difficulty_tier"] == "medium"],
            "public_hard": [e["trip_id"] for e in public_eps if e["difficulty_tier"] == "hard"],
        },
        "hidden_summary": {
            "hidden_count": 30,
            "hidden_tier_counts": {"easy": 4, "medium": 13, "hard": 13},
            "note": "Hidden tracks are held in final_project/ta_only and are not student-facing.",
        },
    })
    write_json("scenario_grammar.json", {
        "spoken_rule_categories": {
            "must_remember": {"description": "Long-lived rule or preference the system should explicitly keep available."},
            "one_off_exception": {"description": "A temporary override that must not overwrite the stable profile."},
            "forbidden_filter": {"description": "A natural-language ban or filter to apply to candidate options."},
            "retire_instruction": {"description": "An instruction that stale assumptions must be removed from active context."},
            "rejected_option_memory": {"description": "The system should remember why an option was rejected and avoid reconsidering it."},
            "retrieval_discipline": {"description": "Only relevant memory should be activated; irrelevant memory should remain retrievable but inactive."},
            "dependency_reasoning": {"description": "Some options materially change the value or feasibility of others, such as partner bundles, badge gates, or late-arrival cutoffs."},
        },
        "episode_axes": {
            "family": sorted({cfg["family"] for cfg in ROUTE_CONFIGS.values()}),
            "difficulty": ["easy", "medium", "hard"],
            "dependency_layers": ["partner_bundle", "badge_unlock", "event_disruption", "refund_risk", "loyalty_value", "late_arrival_cutoff", "stakeholder_tradeoff"],
        },
        "hidden_summary": {
            "hidden_count": 30,
            "hidden_tier_counts": {"easy": 4, "medium": 13, "hard": 13},
            "note": "Hidden episodes are generated from TA-only specs using the same high-level grammar.",
        },
    })

if __name__ == "__main__":
    build_and_write()
