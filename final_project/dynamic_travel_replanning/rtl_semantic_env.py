from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class RTLSemanticEnv:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.flights = json.loads((self.data_dir / "inventory_flights.json").read_text())
        self.hotels = json.loads((self.data_dir / "inventory_hotels.json").read_text())
        self.restaurants = json.loads((self.data_dir / "inventory_restaurants.json").read_text())
        self.activities = json.loads((self.data_dir / "inventory_activities.json").read_text())
        self.policy_rules = json.loads((self.data_dir / "policy_rules.json").read_text())
        self.transit = json.loads((self.data_dir / "transit_matrix.json").read_text())
        self.weather = json.loads((self.data_dir / "weather_buckets.json").read_text())
        self.profile_briefs = json.loads((self.data_dir / "profile_briefs.json").read_text())
        self.venue_briefs = json.loads((self.data_dir / "venue_briefs.json").read_text())
        self.city_ops = json.loads((self.data_dir / "city_ops_notes.json").read_text())
        self.memory_corpus = json.loads((self.data_dir / "memory_corpus.json").read_text()) if (self.data_dir / "memory_corpus.json").exists() else []
        self.rejected_options = json.loads((self.data_dir / "rejected_options_memory.json").read_text()) if (self.data_dir / "rejected_options_memory.json").exists() else []
        self.partner_promotions = json.loads((self.data_dir / "partner_promotions.json").read_text()) if (self.data_dir / "partner_promotions.json").exists() else []
        self.event_calendar = json.loads((self.data_dir / "event_calendar.json").read_text()) if (self.data_dir / "event_calendar.json").exists() else []
        self.loyalty_profiles = json.loads((self.data_dir / "loyalty_programs.json").read_text()) if (self.data_dir / "loyalty_programs.json").exists() else {}
        self.stakeholder_briefs = json.loads((self.data_dir / "stakeholder_briefs.json").read_text()) if (self.data_dir / "stakeholder_briefs.json").exists() else {}
        self.booking_constraints = json.loads((self.data_dir / "booking_constraints.json").read_text()) if (self.data_dir / "booking_constraints.json").exists() else []
        self.option_dependencies = json.loads((self.data_dir / "option_dependencies.json").read_text()) if (self.data_dir / "option_dependencies.json").exists() else []

    def search_flights(self, origin: str, destination: str) -> List[Dict[str, Any]]:
        out = [dict(row) for row in self.flights if row["origin"] == origin and row["destination"] == destination]
        for row in out:
            tags = []
            if row.get("red_eye"):
                tags.append("red_eye")
            if row.get("time_window") == "morning":
                tags += ["meeting_safe", "easy_arrival"]
            if row.get("stops", 0) == 0:
                tags += ["simple_itinerary", "low_friction"]
            if row.get("refundable"):
                tags += ["change_friendly", "refundable_priority"]
            row["semantic_tags"] = tags
            row["description_snippet"] = (
                "Safer for next-morning work and easier to recover from."
                if not row.get("red_eye")
                else "Cheaper but often leaves travellers foggy before important meetings."
            ) + (
                " Direct routing lowers disruption risk."
                if row.get("stops", 0) == 0
                else " Connection risk is higher than the headline fare suggests."
            ) + (
                " Refundability helps when the schedule is still moving."
                if row.get("refundable")
                else " Non-refundable terms can become expensive if the meeting shifts."
            )
        return out

    def search_hotels(self, city: str) -> List[Dict[str, Any]]:
        out = [dict(row) for row in self.hotels if row["city"] == city]
        for row in out:
            tags = []
            if row.get("quiet_score", 0) >= 8.0:
                tags += ["quiet", "business_friendly"]
            if row.get("airport_access_score", 0) >= 7.5:
                tags += ["easy_airport_access", "low_friction"]
            if not row.get("chain"):
                tags.append("local_character")
            if row.get("zone") in {"namba", "xinyi", "one_north", "umeda", "blue_line_corridor"}:
                tags.append("central_bundle")
            if row.get("zone") in {"clarke_quay", "shinsekai", "scenic_outer", "ximending"}:
                tags.append("nightlife_strip")
            if row.get("zone") in {"marina"}:
                tags.append("polished_client_area")
            if row.get("late_checkout"):
                tags.append("loyalty_bundle_value")
            if row.get("meeting_shuttle"):
                tags.append("shuttle_bundle")
            row["semantic_tags"] = tags
            row["review_snippet"] = " ".join(
                [
                    "Guests say the room stays quiet enough for prep-heavy work."
                    if row.get("quiet_score", 0) >= 8.0
                    else "Reviews repeatedly mention corridor noise or late-night spillover.",
                    "Transfer to airport and venue feels low-friction."
                    if row.get("airport_access_score", 0) >= 7.5
                    else "Transfer looks manageable on paper but can feel slower during changeovers.",
                    "Late checkout or meeting shuttle support improves bundle resilience."
                    if row.get("late_checkout") or row.get("meeting_shuttle")
                    else "Operational perks are limited, so the room has to win on pure fit.",
                ]
            )
        return out

    def search_restaurants(self, city: str) -> List[Dict[str, Any]]:
        out = [dict(row) for row in self.restaurants if row["city"] == city]
        for row in out:
            tags = []
            dietary = set(row.get("dietary_flags", []))
            if "vegan" in dietary or "vegan_preorder" in dietary:
                tags.append("vegan_friendly")
            if row.get("price_level", 3) <= 2:
                tags.append("budget_gentle")
            if row.get("area") in {"namba", "xinyi", "one_north", "marina", "umeda", "blue_line_corridor"}:
                tags.append("central_bundle")
            if row.get("area") in {"clarke_quay", "shinsekai", "scenic_outer", "ximending"}:
                tags.append("nightlife_strip")
            if row.get("quiet_score", 0) >= 7.8:
                tags += ["quiet", "team_friendly"]
            if row.get("client_ready_score", 0) >= 7.8 or row.get("area") in {"marina", "xinyi", "umeda"}:
                tags.append("client_ready")
            if row.get("private_room"):
                tags.append("private_room_bonus")
            if row.get("badge_only"):
                tags.append("conference_badge_access")
            row["semantic_tags"] = tags
            tone = "Dietary requests are handled carefully and the room tone stays calm." if "quiet" in tags else "Great energy, but tired travellers may find the vibe louder than expected."
            private_room = " A private room is available and can materially change the fit for client-facing dinners." if row.get("private_room") else ""
            badge_note = " Access may depend on an event badge or partner booking." if row.get("badge_only") else ""
            row["review_snippet"] = tone + private_room + badge_note
        return out

    def search_activities(self, city: str) -> List[Dict[str, Any]]:
        out = [dict(row) for row in self.activities if row["city"] == city]
        for row in out:
            tags = ["weather_safe" if row.get("indoor") else "weather_sensitive"]
            if row.get("category") in {"museum", "gallery", "tea_house", "innovation_center_tour", "meeting_buffer", "client_visit"}:
                tags += ["calm", "local_character"]
            if row.get("location_zone") in {"namba", "xinyi", "one_north", "umeda", "blue_line_corridor"}:
                tags.append("central_bundle")
            if row.get("badge_only"):
                tags.append("conference_badge_access")
            row["semantic_tags"] = tags
            row["description_snippet"] = (
                "Reliable even when weather shifts."
                if row.get("indoor")
                else "Memorable when conditions cooperate, but brittle under rain or fatigue."
            ) + (" Access depends on an active badge or partner booking." if row.get("badge_only") else "")
        return out

    def get_policy(self, profile: str = "default") -> Dict[str, Any]:
        return self.policy_rules[profile]

    def get_profile_brief(self, traveler_id: str) -> Dict[str, Any]:
        return self.profile_briefs[traveler_id]

    def get_venue_brief(self, city: str, family: str) -> Dict[str, Any]:
        return self.venue_briefs[f"{city}_{family}"]

    def get_city_ops_notes(self, city: str) -> Dict[str, Any]:
        return self.city_ops[city]

    def get_partner_promotions(
        self,
        city: str | None = None,
        hotel_id: str | None = None,
        restaurant_id: str | None = None,
        activity_id: str | None = None,
        family: str | None = None,
    ) -> List[Dict[str, Any]]:
        out = []
        for row in self.partner_promotions:
            if city and row.get("city") != city:
                continue
            if family and row.get("family") not in {None, family}:
                continue
            if hotel_id and row.get("hotel_id") not in {None, hotel_id}:
                continue
            if restaurant_id and row.get("restaurant_id") not in {None, restaurant_id}:
                continue
            if activity_id and row.get("activity_id") not in {None, activity_id}:
                continue
            out.append(dict(row))
        return out

    def get_event_calendar(self, city: str) -> List[Dict[str, Any]]:
        return [dict(row) for row in self.event_calendar if row.get("city") == city]

    def get_loyalty_profile(self, traveler_id: str) -> Dict[str, Any] | None:
        payload = self.loyalty_profiles.get(traveler_id)
        return dict(payload) if payload else None

    def get_stakeholder_brief(self, stakeholder_id: str) -> Dict[str, Any] | None:
        payload = self.stakeholder_briefs.get(stakeholder_id)
        return dict(payload) if payload else None

    def get_booking_constraints(self, city: str | None = None, family: str | None = None) -> List[Dict[str, Any]]:
        out = []
        for row in self.booking_constraints:
            if city and row.get("city") not in {None, city}:
                continue
            if family and row.get("family") not in {None, family}:
                continue
            out.append(dict(row))
        return out

    def get_option_dependencies(self, city: str | None = None) -> List[Dict[str, Any]]:
        out = []
        for row in self.option_dependencies:
            if city and row.get("city") not in {None, city}:
                continue
            out.append(dict(row))
        return out
