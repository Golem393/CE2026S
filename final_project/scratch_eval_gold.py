import json
from pathlib import Path
from dynamic_travel_replanning.rtl_semantic_env import RTLSemanticEnv
from dynamic_travel_replanning.evaluator import evaluate_episode

env = RTLSemanticEnv("dynamic_travel_replanning")
episodes = json.loads(Path("dynamic_travel_replanning/episodes_public_example.json").read_text())
ep = next(e for e in episodes if e["trip_id"] == "rtl7_public_easy_001")

gold_submission = {
    "flight_id": ep["gold"]["acceptable_flights"][0],
    "hotel_id": ep["gold"]["acceptable_hotels"][0],
    "restaurant_id": ep["gold"]["good_restaurants"][0],
    "activity_id": ep["gold"]["good_activities"][0],
}

# We want to print the hard constraint status. Let's inspect them inside a custom evaluation run.
city = ep["city"]
flight = next(f for f in env.search_flights(ep["origin"], city) if f["flight_id"] == gold_submission["flight_id"])
hotel = next(h for h in env.search_hotels(city) if h["hotel_id"] == gold_submission["hotel_id"])
restaurant = next(r for r in env.search_restaurants(city) if r["restaurant_id"] == gold_submission["restaurant_id"])
activity = next(a for a in env.search_activities(city) if a["activity_id"] == gold_submission["activity_id"])

total_cost = flight["fare_total"] + hotel["nightly_price"] * ep["nights"] + restaurant["price_level"] * 25000 + activity["price"]
print(f"Total cost: {total_cost}")
print(f"Budget total: {ep['budget_total']}")
print(f"Is under budget? {total_cost <= ep['budget_total']}")

res = evaluate_episode(env, ep, gold_submission, ep["gold"])
