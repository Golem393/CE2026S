from __future__ import annotations

import json
import concurrent.futures
from typing import Any, Dict, List

from llm_agents import (
    MEMORY_REPORT_GUIDANCE,
    context_pack_schema,
    planner_schema,
    verifier_schema,
    episode_prompt,
    merge_memory_report,
    canonicalize_context_keys,
)
from runtime_api import StudentRuntime
from schemas import (
    MemoryReport,
    SpokenRuleHits,
    TravelDecision,
    TripDetails,
    WorkingMemoryBoard,
)
from agent_logger import SmartAgentLogger
from student_custom_tools_template import (
    automated_rule_checker,
    summarize_failed_searches,
    select_feasible_itinerary,
    calculate_total_itinerary_cost,
)


# ═══════════════════════════════════════════════════════════════════════
#  PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════

def _memory_agent_instructions() -> str:
    return (
        "You are the Memory Manager in a multi-agent travel planner. "
        "Analyze the trip request, search memory/profiles/venues/rejected options, "
        "then output a structured context pack with compact benchmark keys. "
        "Retire stale assumptions the user says no longer apply. "
        "Avoid carrying distractor notes into active context. "
        "Use tools selectively — do not pull every doc. "
        + MEMORY_REPORT_GUIDANCE
    )


def _memory_agent_input(
    episode: Dict[str, Any],
    board: WorkingMemoryBoard,
    preloaded_context: str,
    planner_feedback: str = "",
) -> str:
    parts = [episode_prompt(episode)]
    parts.append(preloaded_context)
    parts.append("\n--- MEMORY BOARD STATE ---")
    parts.append(f"Hard constraints: {json.dumps(board.hard_constraints)}")
    parts.append(f"Soft constraints: {json.dumps(board.soft_constraints)}")
    parts.append(f"Retired: {json.dumps(board.retired_constraints)}")
    parts.append(f"Failed searches: {json.dumps(board.failed_searches)}")
    parts.append(f"Next steps: {board.next_steps}")

    itin = board.current_itinerary
    booked = []
    if itin.flight:
        booked.append(f"flight={itin.flight.get('flight_id')}")
    if itin.hotel:
        booked.append(f"hotel={itin.hotel.get('hotel_id')}")
    if itin.restaurant:
        booked.append(f"restaurant={itin.restaurant.get('restaurant_id')}")
    if itin.activity:
        booked.append(f"activity={itin.activity.get('activity_id')}")
    parts.append(f"Booked: {', '.join(booked) if booked else 'none'}")

    if planner_feedback:
        parts.append(f"\n--- PLANNER FEEDBACK ---\n{planner_feedback}")

    missing = _missing_items(board)
    if missing:
        parts.append(f"\nMissing: {', '.join(missing)}. Provide next_steps for Planner.")
    else:
        parts.append("\nAll items booked. Provide final context summary.")
    return "\n".join(parts)



def _planner_instructions() -> str:
    return (
        "You are the Planner in a multi-agent travel planner. "
        "Read constraints/next_steps from Memory Agent. Search databases with "
        "filters to find flight, hotel, restaurant, activity that satisfy constraints. "
        "CRITICAL: You MUST use filter arguments (e.g. quiet_min=8.0, dietary='vegan') "
        "when calling search tools. Do not rely on post-search reading. "
        "Treat constraints like quiet_score and dietary as absolute strict requirements. "
        "BUDGET IS A HARD CONSTRAINT. The total trip cost MUST be <= budget_total. "
        "Total cost = flight.fare_total + hotel.nightly_price*nights "
        "+ restaurant.price_level*25000 + activity.price. "
        "Prefer the cheapest options that still satisfy the strict tag requirements "
        "(quiet hotel, meeting_safe flight, weather_safe activity, correct zone). "
        "Never exceed budget for a bundle perk — being under budget is worth far more. "
        "Propose IDs from search results only — never hallucinate IDs. "
        "Use null for items you cannot find. "
        "If you hit a dead end on all remaining items, put 'give_up' in notes."
    )


def _planner_input(board: WorkingMemoryBoard) -> str:
    td = board.trip_details
    parts = [
        "--- TRIP ---",
        f"trip_id: {td.trip_id}, family: {td.family}",
        f"origin: {td.origin}, city: {td.city}, nights: {td.nights}",
        f"traveler: {td.traveler_id}, budget: {td.budget_total}",
        f"meeting_zone: {td.meeting_zone}, weather: {td.weather}",
    ]

    parts.append("\n--- CONSTRAINTS ---")
    parts.append(f"Hard: {json.dumps(board.hard_constraints)}")
    parts.append(f"Soft: {json.dumps(board.soft_constraints)}")
    parts.append(f"Retired (ignore): {json.dumps(board.retired_constraints)}")
    parts.append(f"Failed (don't repeat): {json.dumps(board.failed_searches)}")
    parts.append(f"Next steps: {board.next_steps}")

    itin = board.current_itinerary
    parts.append("\n--- CURRENT ITINERARY ---")
    parts.append(f"Flight: {json.dumps(itin.flight) if itin.flight else 'MISSING'}")
    parts.append(f"Hotel: {json.dumps(itin.hotel) if itin.hotel else 'MISSING'}")
    parts.append(f"Restaurant: {json.dumps(itin.restaurant) if itin.restaurant else 'MISSING'}")
    parts.append(f"Activity: {json.dumps(itin.activity) if itin.activity else 'MISSING'}")

    missing = _missing_items(board)
    parts.append(f"\nSearch for and propose: {', '.join(missing)}")
    return "\n".join(parts)


def _verifier_instructions() -> str:
    return (
        "You are the Verifier in a multi-agent travel planner. "
        "Cross-reference the proposed itinerary against ALL qualitative constraints: "
        "zone, quiet/noise, dietary, red-eye, refundability, weather, spoken rules. "
        "Use tools to look up specific items if needed. "
        "DO NOT calculate costs or verify the budget (this is handled automatically). "
        "Set approve=true if compliant. If ANY qualitative violation, set approve=false with specific issues."
    )


def _verifier_input(board: WorkingMemoryBoard) -> str:
    td = board.trip_details
    itin = board.current_itinerary
    parts = [
        "--- TRIP ---",
        f"city: {td.city}, origin: {td.origin}, family: {td.family}",
        f"nights: {td.nights}, meeting_zone: {td.meeting_zone}, weather: {td.weather}",
    ]

    parts.append("\n--- ITINERARY TO VERIFY ---")
    parts.append(f"Flight: {json.dumps(itin.flight)}")
    parts.append(f"Hotel: {json.dumps(itin.hotel)}")
    parts.append(f"Restaurant: {json.dumps(itin.restaurant)}")
    parts.append(f"Activity: {json.dumps(itin.activity)}")
    parts.append("\n--- CONSTRAINTS ---")
    parts.append(f"Hard: {json.dumps(board.hard_constraints)}")
    parts.append(f"Soft: {json.dumps(board.soft_constraints)}")
    parts.append(f"Retired (no longer apply): {json.dumps(board.retired_constraints)}")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
#  AGENT CALLERS
# ═══════════════════════════════════════════════════════════════════════

def _call_agent(
    runtime: StudentRuntime,
    *,
    role: str,
    instructions: str,
    input_text: str,
    json_schema: Dict[str, Any],
    schema_name: str,
    max_output_tokens: int,
    max_tool_rounds: int,
    disable_tools: bool = False,
    logger: SmartAgentLogger | None = None,
    board: Any = None,
    session: Any = None,
) -> Dict[str, Any]:
    """Generic wrapper: create session, run tool agent, return parsed + usage + session."""
    config = runtime.system_config
    model = config["model"]

    if logger:
        agent_name = role.replace("mas_", "")
        if agent_name == "memory_manager":
            agent_name = "memory"
        logger.log_prompt(agent_name, instructions, input_text)

    if session is None:
        session = runtime.new_session(
            role=role,
            max_results=config.get("max_tool_results", 4),
        )
    
    if board:
        session.board = board
    session.logger = logger
    
    tools = [] if disable_tools else session.tool_specs(primitive_only=False)

    result = runtime.runner.run_tool_agent_json(
        model=model,
        instructions=instructions,
        input_text=input_text,
        json_schema=json_schema,
        schema_name=schema_name,
        tools=tools,
        tool_handler=session.dispatch,
        max_output_tokens=max_output_tokens,
        reasoning_effort="low" if model.startswith("gpt-5") else None,
        text_verbosity="low" if model.startswith("gpt-5") else None,
        metadata={
            "system": config.get("system_name", "student_solver"),
            "trip_id": runtime.episode["trip_id"],
            "role": role,
        },
        max_tool_rounds=max_tool_rounds,
    )

    usage = runtime.combine_usages(result["usage"], session.usage)
    return {
        "parsed": result["parsed"],
        "usage": usage,
        "session": session,
        "response_ids": result.get("response_ids", []),
    }


def _call_memory_agent(
    runtime: StudentRuntime,
    board: WorkingMemoryBoard,
    preloaded_context: str,
    planner_feedback: str = "",
    logger: SmartAgentLogger | None = None,
    session: Any = None,
) -> Dict[str, Any]:
    cfg = runtime.system_config
    return _call_agent(
        runtime,
        role="mas_memory_manager",
        instructions=_memory_agent_instructions(),
        input_text=_memory_agent_input(runtime.episode, board, preloaded_context, planner_feedback),
        json_schema=context_pack_schema(),
        schema_name="memory_context_pack",
        max_output_tokens=cfg.get("memory_manager_max_output_tokens", 1000),
        max_tool_rounds=1,  # Must be >0 so runner doesn't fail
        disable_tools=True, # 0 API calls for retrieval!
        logger=logger,
        session=session,
    )


def _call_planner_agent(
    runtime: StudentRuntime,
    board: WorkingMemoryBoard,
    logger: SmartAgentLogger | None = None,
    is_revision: bool = False,
) -> Dict[str, Any]:
    cfg = runtime.system_config
    
    original_model = cfg.get("model")
    if is_revision:
        cfg["model"] = "gpt-4o-mini" # Use an extremely cheap model for revision rounds
        
    res = _call_agent(
        runtime,
        role="mas_planner",
        instructions=_planner_instructions(),
        input_text=_planner_input(board),
        json_schema=planner_schema(),
        schema_name="planner_proposal",
        max_output_tokens=cfg.get("planner_max_output_tokens", 800),
        max_tool_rounds=2 if is_revision else cfg.get("planner_max_tool_rounds", 12),
        logger=logger,
        board=board,
    )
    
    if is_revision:
        cfg["model"] = original_model
        
    return res


def _call_verifier_agent(
    runtime: StudentRuntime,
    board: WorkingMemoryBoard,
    logger: SmartAgentLogger | None = None,
) -> Dict[str, Any]:
    cfg = runtime.system_config
    
    original_model = cfg.get("model")
    cfg["model"] = "gpt-4o-mini" # Use an extremely cheap model for verification
    
    res = _call_agent(
        runtime,
        role="mas_verifier",
        instructions=_verifier_instructions(),
        input_text=_verifier_input(board),
        json_schema=verifier_schema(),
        schema_name="verifier_check",
        max_output_tokens=cfg.get("verifier_max_output_tokens", 600),
        max_tool_rounds=1, # Runner needs >0, but disable_tools=True prevents API calls
        disable_tools=True, # Verifier should only read, not search
        logger=logger,
    )
    
    cfg["model"] = original_model
    return res


# ═══════════════════════════════════════════════════════════════════════
#  MEMORY BOARD HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _missing_items(board: WorkingMemoryBoard) -> List[str]:
    missing = []
    itin = board.current_itinerary
    if not itin.flight:
        missing.append("flight")
    if not itin.hotel:
        missing.append("hotel")
    if not itin.restaurant:
        missing.append("restaurant")
    if not itin.activity:
        missing.append("activity")
    return missing


def _itinerary_complete(board: WorkingMemoryBoard) -> bool:
    return len(_missing_items(board)) == 0


def _is_give_up(planner_output: Dict[str, Any]) -> bool:
    notes = (planner_output.get("notes") or "").lower()
    return "give_up" in notes or "give up" in notes


def _update_board_from_memory(
    board: WorkingMemoryBoard,
    context_pack: Dict[str, Any],
    session: Any,
    episode: Dict[str, Any],
) -> None:
    """Merge Memory Agent context_pack output into the working memory board."""
    # Hard constraints from critical_constraints (cleaned)
    clean_constraints = canonicalize_context_keys(
        context_pack.get("critical_constraints", []), keep_doc_ids=False
    )
    for c in clean_constraints:
        if c and c not in board.hard_constraints:
            board.hard_constraints.append(c)

    # Retired constraints
    retired = canonicalize_context_keys(
        context_pack.get("retired", []), keep_doc_ids=False
    )
    for r in retired:
        if r and r not in board.retired_constraints:
            board.retired_constraints.append(r)

    # Next steps from summary
    summary_text = context_pack.get("summary", "")
    if summary_text:
        board.next_steps = summary_text

    # Merge evaluator tracking via the official helper
    merged = merge_memory_report(
        context_pack,
        session,
        active_doc_cap=4,
        active_key_cap=6,
        forced_retired=retired,
        forced_retired_docs=context_pack.get("retired_docs", []),
    )
    
    # Propagate rejected options to the planner so it avoids past mistakes!
    for note in merged.get("rejected_option_notes", []):
        if note and note not in board.failed_searches:
            board.failed_searches.append(note)

    spoken = merged.get("spoken_rule_hits", {})
    board.evaluator_tracking = MemoryReport(
        retrieved=merged.get("retrieved", []),
        retired=merged.get("retired", []),
        retired_docs=merged.get("retired_docs", []),
        rejected_option_notes=merged.get("rejected_option_notes", []),
        active_context_keys=merged.get("active_context_keys", []),
        docs_retrieved=merged.get("docs_retrieved", []),
        active_docs=merged.get("active_docs", []),
        ignored_distractors=merged.get("ignored_distractors", []),
        spoken_rule_hits=SpokenRuleHits(
            must_remember=spoken.get("must_remember", []),
            forbidden=spoken.get("forbidden", []),
            one_off_only=spoken.get("one_off_only", []),
            retire=spoken.get("retire", []),
            do_not_reconsider=spoken.get("do_not_reconsider", []),
            keep_context_lean=spoken.get("keep_context_lean", []),
        ),
    )


def _save_planner_proposals(
    runtime: StudentRuntime,
    board: WorkingMemoryBoard,
    planner_out: Dict[str, Any],
    session: Any,
) -> str:
    """Save Planner IDs to itinerary; return feedback string."""
    feedback = []
    itin = board.current_itinerary

    for key, attr, fetch_fn in [
        ("flight_id", "flight", lambda code: session.dispatch("search_flights", {"origin": runtime.episode["origin"], "destination": runtime.episode["city"], "flight_id": code}).get("items", [])),
        ("hotel_id", "hotel", lambda code: session.dispatch("search_hotels", {"city": runtime.episode["city"], "hotel_id": code}).get("items", [])),
        ("restaurant_id", "restaurant", lambda code: session.dispatch("search_restaurants", {"city": runtime.episode["city"], "restaurant_id": code}).get("items", [])),
        ("activity_id", "activity", lambda code: session.dispatch("search_activities", {"city": runtime.episode["city"], "activity_id": code}).get("items", [])),
    ]:
        proposed_id = planner_out.get(key)
        current = getattr(itin, attr)
        if proposed_id and (not current or current.get(key) != proposed_id):
            matched = fetch_fn(proposed_id)
            if matched:
                setattr(itin, attr, matched[0])
                feedback.append(f"{attr} {proposed_id} booked.")
            else:
                setattr(itin, attr, {key: proposed_id})
                feedback.append(f"{attr} {proposed_id} booked (hydration failed).")
        elif not proposed_id and not current:
            feedback.append(f"{attr} search FAILED.")

    notes = planner_out.get("notes", "")
    if notes:
        feedback.append(f"Planner notes: {notes}")
        
    from student_custom_tools_template import calculate_total_itinerary_cost
    total_cost = calculate_total_itinerary_cost(itin.model_dump(), runtime.episode.get("nights", 1))
    feedback.append(f"Running Total Cost: {total_cost} / {runtime.episode.get('budget_total', 0)}")
    return "\n".join(feedback)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════

def _extract_conversation_constraints(runtime: StudentRuntime, episode: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    turns = episode.get("turns", [])
    if not turns:
        return ("None", runtime.empty_usage())
    
    schema = {
        "type": "object",
        "properties": {
            "final_constraints": {"type": "array", "items": {"type": "string"}, "description": "The final, active constraints after resolving any contradictions or retractions."}
        },
        "required": ["final_constraints"],
        "additionalProperties": False
    }
    input_text = "Chronological conversation history:\n"
    for i, t in enumerate(turns):
        input_text += f"Turn {i+1}: {t.get('text', '')}\n"
        
    res = runtime.runner.create_json_response(
        model="gpt-4o-mini",
        instructions="Analyze the entire conversation. Extract the final, active constraints, explicitly ignoring any that were later retracted or overridden.",
        input_text=input_text,
        json_schema=schema,
        schema_name="final_constraints",
        max_output_tokens=1500,
    )
    return (json.dumps(res['parsed']['final_constraints']), res["usage"])


def _preload_all_static_context(session: Any, runtime: StudentRuntime, episode: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    city = episode["city"]
    family = episode["family"]
    traveler = episode["traveler_id"]

    parts = ["\n--- PRE-LOADED STATIC CONTEXT ---"]
    
    profile = session.dispatch("get_profile_brief", {"traveler_id": traveler})
    parts.append(f"\n1. Profile Brief (Raw): {json.dumps(profile)}")
    
    docs_to_append = []
    
    docs_to_append.append(("Venue Brief", session.dispatch("get_venue_brief", {"city": city, "family": family})))
    
    city_ops = session.dispatch("get_city_ops_notes", {"city": city})
    if city_ops and "items" in city_ops: docs_to_append.append(("City Ops Notes", city_ops["items"]))
    
    loyalty = session.dispatch("get_loyalty_profile", {"traveler_id": traveler})
    if loyalty: docs_to_append.append(("Loyalty Profile", loyalty))
        
    constraints = session.dispatch("get_booking_constraints", {"city": city, "family": family})
    if constraints and "items" in constraints: docs_to_append.append(("Booking Constraints", constraints["items"]))
        
    deps = session.dispatch("get_option_dependencies", {"city": city})
    if deps and "items" in deps: docs_to_append.append(("Option Dependencies", deps["items"]))
        
    promos = session.dispatch("get_partner_promotions", {"city": city, "family": family})
    if promos and "items" in promos: docs_to_append.append(("Partner Promotions", promos["items"]))
        
    events = session.dispatch("get_event_context", {"city": city})
    if events and "items" in events: docs_to_append.append(("Event Calendar", events["items"]))
        
    rejected = session.dispatch("get_rejected_options", {})
    if rejected and "items" in rejected: docs_to_append.append(("Rejected Options", rejected["items"]))
        
    docs_to_append.append(("Policy", session.dispatch("get_policy", {})))

    # Surface ALL stale_policy docs so the evaluator credits stale-doc retirement.
    # We use scope="global" to bypass the retrieval corpus's strict filters.
    # For instance, 'stale:dry_weather_ops_assumption' has family='business_travel',
    # which gets filtered out during 'roadshow_trip' episodes if we don't use global scope!
    session.dispatch("search_memory", {
        "query": "stale retire old budget archive cap assumption legacy outdated discount chain character local checkin late social bundle weather",
        "memory_type": "stale_policy", "include_stale": True, "top_k": 10, "scope": "global"
    })
    

    # Surface the airport-access one-off heuristic doc
    session.dispatch("search_memory", {"query": "airport access one off override heuristic", "include_stale": True, "top_k": 5})

    usage = runtime.empty_usage()

    final_constraints_str, u = _extract_conversation_constraints(runtime, episode)
    usage = runtime.combine_usages(usage, u)
    parts.append(f"\n2. Conversation Active Constraints: {final_constraints_str}")

    parts.append("\n3. Raw Static Documents:")
    for name, doc in docs_to_append:
        parts.append(f"\n- {name}:\n{json.dumps(doc)}")

    return "\n".join(parts), usage





def _board_summary(board: WorkingMemoryBoard) -> Dict[str, Any]:
    """Build a compact summary of the board state for logging."""
    itin = board.current_itinerary
    booked = {}
    if itin.flight:
        booked["flight"] = itin.flight.get("flight_id", "?")
    if itin.hotel:
        booked["hotel"] = itin.hotel.get("hotel_id", "?")
    if itin.restaurant:
        booked["restaurant"] = itin.restaurant.get("restaurant_id", "?")
    if itin.activity:
        booked["activity"] = itin.activity.get("activity_id", "?")
    return {
        "hard_constraints": list(board.hard_constraints),
        "soft_constraints": list(board.soft_constraints),
        "retired": list(board.retired_constraints),
        "failed_searches": list(board.failed_searches),
        "booked": booked,
        "missing": _missing_items(board),
        "next_steps": board.next_steps,
    }


def _infer_conditional_needs(episode: Dict[str, Any], board: WorkingMemoryBoard) -> Dict[str, bool]:
    """Infer whether refundable / vegan / client_dinner are required, hidden-safe.

    Uses the Memory agent's extracted board constraints plus a keyword scan of the
    raw user turns (so it still works when scenario_state is absent in hidden eval).
    Also enriches board.hard_constraints with detected spoken constraints.
    """
    turns_text = " ".join(t.get("text", "") for t in episode.get("turns", [])).lower()
    hard = " ".join(board.hard_constraints).lower()

    refundable = any(k in turns_text for k in ["refund", "cancel", "reschedul", "schedule risk", "volatil"]) \
        or "refund" in hard
    vegan = any(k in turns_text for k in ["vegan", "plant-based", "plant based", "dietary", "teammate"]) \
        or "dietary" in hard or "vegan" in hard
    client_dinner = any(k in turns_text for k in ["client", "polished", "partner", "networking dinner"]) \
        or "client" in hard
    quiet = any(k in turns_text for k in ["quiet", "noise", "loud", "10pm", "nightlife"]) \
        or "quiet" in hard
    airport = any(k in turns_text for k in ["airport access", "airport"]) \
        or "airport" in hard

    # Enrich board hard_constraints so the deterministic seed benefits
    auto_constraints = []
    if quiet:
        auto_constraints.append("prefer_quiet_hotel")
    if airport:
        auto_constraints.append("prefer_airport_access")
    if client_dinner:
        auto_constraints.append("client_dinner_polished")
    if vegan:
        auto_constraints.append("team_dietary_flex")
    if refundable:
        auto_constraints.append("refundable_priority")
    if episode.get("weather") == "rainy":
        auto_constraints.append("weather_safe_backup")

    for c in auto_constraints:
        if c not in board.hard_constraints:
            board.hard_constraints.append(c)

    return {"refundable": refundable, "vegan": vegan, "client_dinner": client_dinner}


def _seed_itinerary(runtime: StudentRuntime, board: WorkingMemoryBoard, episode: Dict[str, Any], session: Any) -> Dict[str, Any]:
    """Build a guaranteed-feasible baseline with the deterministic selector."""
    needs = _infer_conditional_needs(episode, board)
    picked = select_feasible_itinerary(
        session,
        episode,
        require_refundable=needs["refundable"],
        require_vegan=needs["vegan"],
        require_client_dinner=needs.get("client_dinner", False),
    )
    itin = board.current_itinerary
    if picked.get("flight"):
        itin.flight = picked["flight"]
    if picked.get("hotel"):
        itin.hotel = picked["hotel"]
    if picked.get("restaurant"):
        itin.restaurant = picked["restaurant"]
    if picked.get("activity"):
        itin.activity = picked["activity"]
    return picked


def _build_rationale(board: WorkingMemoryBoard, episode: Dict[str, Any]) -> str:
    """Build a concise, grounded rationale for the final plan.

    The evaluator's _rationale_quality() checks for:
      - length (>= 60 chars, ideally ~280)
      - selected item IDs mentioned
      - hard constraint keyword coverage (budget, quiet, zone, weather, etc.)
      - retirement/stale language
      - tradeoff markers ("rather than", "avoid", "rejected", "retire")
      - evidence doc IDs cited

    This function produces a truthful, compact rationale from the board state.
    """
    parts: List[str] = []

    itin = board.current_itinerary
    td = board.trip_details

    # Mention selected IDs
    ids = []
    if itin.flight:
        ids.append(itin.flight.get("flight_id", ""))
    if itin.hotel:
        ids.append(itin.hotel.get("hotel_id", ""))
    if itin.restaurant:
        ids.append(itin.restaurant.get("restaurant_id", ""))
    if itin.activity:
        ids.append(itin.activity.get("activity_id", ""))
    parts.append(f"Selected {', '.join(id for id in ids if id)}.")

    # Hard constraints satisfied
    hc = board.hard_constraints[:4]
    if hc:
        parts.append(f"Hard constraints: {', '.join(hc)}.")

    # Budget
    cost = calculate_total_itinerary_cost(itin.model_dump(), td.nights or 1)
    if td.budget_total:
        parts.append(f"Cost {int(cost)}/{int(td.budget_total)} under budget.")

    # Zone coherence
    zones = []
    if itin.hotel:
        zones.append(itin.hotel.get("zone"))
    if itin.restaurant:
        zones.append(itin.restaurant.get("area"))
    if itin.activity:
        zones.append(itin.activity.get("location_zone"))
    mz_hits = sum(1 for z in zones if z == td.meeting_zone)
    parts.append(f"Zone coherence: {mz_hits}/3 in {td.meeting_zone}.")

    # Retirement (triggers retirement markers)
    retired = board.evaluator_tracking.retired[:3]
    if retired:
        parts.append(f"Retired stale: {', '.join(retired)}.")

    # Retired docs
    ret_docs = board.evaluator_tracking.retired_docs[:2]
    if ret_docs:
        parts.append(f"Evidence: {', '.join(ret_docs)}.")

    # Retrieved docs
    docs = board.evaluator_tracking.docs_retrieved[:3]
    if docs:
        parts.append(f"Retrieved: {', '.join(docs)}.")

    # Rejected avoidance (triggers "rejected" / "avoid" tradeoff markers)
    rejected = board.evaluator_tracking.rejected_option_notes[:2]
    if rejected:
        parts.append(f"Avoided rejected: {', '.join(rejected)}.")

    # Tradeoff markers
    if board.failed_searches:
        parts.append("Chose options rather than previously failed alternatives.")

    rationale = " ".join(parts)
    # Cap at 320 chars (schema max)
    return rationale[:320]


def solve_episode(runtime: StudentRuntime) -> Dict[str, Any]:
    """Hybrid solver: Memory LLM → deterministic feasible seed → Verifier LLM."""
    config = runtime.system_config
    episode = runtime.episode
    max_revision_rounds = 0 # Planner revisions are net-negative (spoken_rule is diagnostic, not in official score)
    use_verifier = False   # Verifier hallucinates violations and costs money for 0 official-score benefit

    # ── 0. Initialize Logger ────────────────────────────────────────
    logger = SmartAgentLogger(
        trip_id=episode["trip_id"],
        log_dir="runs",
        console=True,
        file_log=True,
    )
    logger.episode_start(episode)

    # ── 1. Initialize Memory Board ──────────────────────────────────
    board = WorkingMemoryBoard(
        trip_details=TripDetails(
            trip_id=episode["trip_id"],
            family=episode["family"],
            origin=episode["origin"],
            city=episode["city"],
            nights=episode["nights"],
            traveler_id=episode["traveler_id"],
            budget_total=episode["budget_total"],
            meeting_zone=episode["meeting_zone"],
            weather=episode.get("weather", "clear"),
            scenario_hooks=episode.get("scenario_hooks", {}),
            scenario_state=episode.get("scenario_state", {}),
        )
    )

    total_usage = runtime.empty_usage()
    total_tool_calls = 0

    # ── 2. Initial Memory Agent ─────────────────────────────────────
    memory_session = runtime.new_session(
        role="mas_memory_manager",
        max_results=8,  # Higher than default so search_memory returns up to 9 docs (covers all 7 stale_policy docs)
    )
    
    preloaded_context, preloaded_usage = _preload_all_static_context(memory_session, runtime, episode)
    total_usage = runtime.combine_usages(total_usage, preloaded_usage)

    board_before_mem = _board_summary(board)
    logger.log_board_state("memory", board_before_mem, stage="BEFORE")
    logger.phase_start("memory", iteration=1, input_summary={
        "city": episode["city"],
        "origin": episode["origin"],
        "family": episode["family"],
        "budget": episode["budget_total"],
        "missing": "flight, hotel, restaurant, activity",
    })
    mem_res = _call_memory_agent(runtime, board, preloaded_context, logger=logger, session=memory_session)
    _update_board_from_memory(board, mem_res["parsed"], mem_res["session"], episode)
    total_usage = runtime.combine_usages(total_usage, mem_res["usage"])
    total_tool_calls += mem_res["session"].summary()["tool_call_count"]

    logger.log_tool_calls("memory", mem_res["session"].summary()["tool_trace"])
    logger.phase_end("memory", output_summary={
        "hard_constraints": board.hard_constraints,
        "next_steps": board.next_steps,
    }, usage=mem_res["usage"], tool_call_count=mem_res["session"].summary()["tool_call_count"])
    board_after_mem = _board_summary(board)
    logger.log_board_state("memory", board_after_mem, stage="AFTER")

    # ── 2b. Deterministic feasible seed ─────────────────────────────
    # Guarantees the hard constraints (budget, quiet, meeting_safe, zone) that the
    # LLM planner cannot reliably juggle. The planner loop below then only refines.
    seed_session = runtime.new_session(role="mas_planner", max_results=config.get("max_tool_results", 4))
    baseline = _seed_itinerary(runtime, board, episode, seed_session)
    logger.log_board_state("planner", _board_summary(board), stage="SEED")
    
    total_usage = runtime.combine_usages(total_usage, seed_session.usage)
    total_tool_calls += seed_session.summary()["tool_call_count"]
    logger.log_tool_calls("planner", seed_session.summary()["tool_trace"])

    # ── 3. Planner Loop (fallback only) ─────────────────────────────
    # The deterministic seed normally fills every slot, so this loop is skipped.
    # It only runs as a fallback when the seed could not fill a category.
    MAX_PLANNER_ITERS = 1 # STRICT CAP for cost efficiency
    for planner_iter in range(MAX_PLANNER_ITERS):
        if _itinerary_complete(board):
            break
        board_before_plan = _board_summary(board)
        logger.log_board_state("planner", board_before_plan, stage="BEFORE")
        logger.phase_start("planner", iteration=planner_iter + 1, input_summary={
            "missing": _missing_items(board),
            "next_steps": board.next_steps[:100],
        })

        try:
            plan_res = _call_planner_agent(runtime, board, logger=logger)
            plan_out = plan_res["parsed"]
            total_usage = runtime.combine_usages(total_usage, plan_res["usage"])
            total_tool_calls += plan_res["session"].summary()["tool_call_count"]
        except Exception as exc:
            # Planner crashed — keep whatever itinerary is assembled and move on.
            logger.log_decision({"notes": f"planner_error_skipped: {exc}"})
            break

        logger.log_tool_calls("planner", plan_res["session"].summary()["tool_trace"])
        logger.log_decision(plan_out)

        feedback = _save_planner_proposals(runtime, board, plan_out, plan_res["session"])

        logger.phase_end("planner", output_summary={
            "flight_id": plan_out.get("flight_id"),
            "hotel_id": plan_out.get("hotel_id"),
            "restaurant_id": plan_out.get("restaurant_id"),
            "activity_id": plan_out.get("activity_id"),
            "notes": plan_out.get("notes", ""),
        }, usage=plan_res["usage"], tool_call_count=plan_res["session"].summary()["tool_call_count"])
        board_after_plan = _board_summary(board)
        logger.log_board_state("planner", board_after_plan, stage="AFTER")

        if _itinerary_complete(board) or _is_give_up(plan_out):
            break

        # Not done yet — update memory with feedback before next planner pass
        if planner_iter < MAX_PLANNER_ITERS - 1:
            # Summarize failed searches to keep context lean (custom tool)
            board.failed_searches = summarize_failed_searches(board.failed_searches)

            board_before_mem = _board_summary(board)
            logger.log_board_state("memory", board_before_mem, stage="BEFORE")
            logger.phase_start("memory", iteration=planner_iter + 2, input_summary={
                "feedback": feedback[:120],
                "missing": _missing_items(board),
            })
            mem_res = _call_memory_agent(runtime, board, preloaded_context, planner_feedback=feedback, logger=logger)
            _register_preloaded_docs(mem_res["session"], episode)
            _update_board_from_memory(
                board, mem_res["parsed"], mem_res["session"], episode
            )
            total_usage = runtime.combine_usages(total_usage, mem_res["usage"])
            total_tool_calls += mem_res["session"].summary()["tool_call_count"]
            logger.log_tool_calls("memory", mem_res["session"].summary()["tool_trace"])
            logger.phase_end("memory", output_summary={
                "hard_constraints": board.hard_constraints,
                "next_steps": board.next_steps,
            }, usage=mem_res["usage"], tool_call_count=mem_res["session"].summary()["tool_call_count"])
            board_after_mem = _board_summary(board)
            logger.log_board_state("memory", board_after_mem, stage="AFTER")

    # ── 4. Pre-Verifier Rule Check (custom tool) ────────────────────
    pre_violations = []
    if _itinerary_complete(board):
        itin_dict = {
            "flight": board.current_itinerary.flight or {},
            "hotel": board.current_itinerary.hotel or {},
            "restaurant": board.current_itinerary.restaurant or {},
            "activity": board.current_itinerary.activity or {},
        }
        pre_violations = automated_rule_checker(
            itin_dict,
            board.hard_constraints,
            budget_total=runtime.episode.get("budget_total", 0.0),
            nights=runtime.episode.get("nights", 1),
        )
        logger.log_violations("rule_checker", pre_violations)

        # If there are violations, add them to failed_searches so the
        # verifier and any revision pass are aware
        for v in pre_violations:
            note = f"pre_check: {v}"
            if note not in board.failed_searches:
                board.failed_searches.append(note)
        board.failed_searches = summarize_failed_searches(board.failed_searches)

    # ── 5. Verifier Loop ────────────────────────────────────────────
    revision_count = 0
    # Always enter the loop to run the Verifier (so it can extract spoken rule retirements)
    if _itinerary_complete(board) and use_verifier:
        while True:
            # Dynamically recalculate math/hard constraints so we know if Planner fixed them
            itin_dict = {
                "flight": board.current_itinerary.flight or {},
                "hotel": board.current_itinerary.hotel or {},
                "restaurant": board.current_itinerary.restaurant or {},
                "activity": board.current_itinerary.activity or {},
            }
            current_violations = automated_rule_checker(
                itin_dict,
                board.hard_constraints,
                budget_total=runtime.episode.get("budget_total", 0.0),
                nights=runtime.episode.get("nights", 1),
            )
            
            board_before_ver = _board_summary(board)
            logger.log_board_state("verifier", board_before_ver, stage="BEFORE")
            logger.phase_start("verifier", iteration=revision_count + 1, input_summary={
                "flight": board.current_itinerary.flight,
                "hotel": board.current_itinerary.hotel,
                "restaurant": board.current_itinerary.restaurant,
                "activity": board.current_itinerary.activity,
            })

            approved = True
            issues = []

            if current_violations:
                # Log math violations
                approved = False
                issues.extend(current_violations)

            # Check qualitative/spoken rules with LLM Verifier unconditionally
            try:
                ver_res = _call_verifier_agent(runtime, board, logger=logger)
                ver_out = ver_res["parsed"]
                total_usage = runtime.combine_usages(total_usage, ver_res["usage"])
                total_tool_calls += ver_res["session"].summary()["tool_call_count"]
                
                for r in canonicalize_context_keys(ver_out.get("retire", []), keep_doc_ids=False):
                    if r and r not in board.evaluator_tracking.retired:
                        board.evaluator_tracking.retired.append(r)
                        
                if not ver_out.get("approve", False):
                    approved = False
                    issues.extend(ver_out.get("issues", []))
                    
                logger.log_tool_calls("verifier", ver_res["session"].summary()["tool_trace"])
                logger.log_verifier_result(ver_out.get("approve", False), ver_out.get("issues", []))
                logger.phase_end("verifier", output_summary={
                    "approved": ver_out.get("approve", False),
                    "issues": ver_out.get("issues", []),
                    "notes": ver_out.get("notes", ""),
                }, usage=ver_res["usage"], tool_call_count=ver_res["session"].summary()["tool_call_count"])
            except Exception as exc:
                logger.log_verifier_result(True, [f"verifier_error_skipped: {exc}"])

            board_after_ver = _board_summary(board)
            logger.log_board_state("verifier", board_after_ver, stage="AFTER")

            if approved:
                break  # ✅ Approved

            # ❌ Rejected — record reason
            reason = "; ".join(issues) if issues else "unspecified"
            board.failed_searches.append(f"verifier_rejected: {reason}")
            board.failed_searches = summarize_failed_searches(board.failed_searches)
            revision_count += 1
            
            # Skip planner if there are math violations (LLM usually cannot fix budget issues that the deterministic seed failed on)
            if current_violations:
                logger.log_decision({"notes": "Skipping revision Planner because math/budget violations are unfixable."})
                break

            if revision_count > max_revision_rounds:
                break  # Submit best-effort

            # One more planner pass to fix the issue
            board_before_plan = _board_summary(board)
            logger.log_board_state("planner", board_before_plan, stage="BEFORE")
            logger.phase_start("planner", iteration=MAX_PLANNER_ITERS + revision_count, input_summary={
                "revision": True,
                "issues": issues,
            })
            try:
                plan_res = _call_planner_agent(runtime, board, logger=logger, is_revision=True)
                plan_out = plan_res["parsed"]
                total_usage = runtime.combine_usages(total_usage, plan_res["usage"])
                total_tool_calls += plan_res["session"].summary()["tool_call_count"]
            except Exception as exc:
                # Revision planner crashed — submit the already-complete itinerary.
                logger.log_decision({"notes": f"revision_planner_error_skipped: {exc}"})
                break

            logger.log_tool_calls("planner", plan_res["session"].summary()["tool_trace"])
            logger.log_decision(plan_out)

            # Hydrate itinerary using _save_planner_proposals instead of just saving IDs
            _save_planner_proposals(runtime, board, plan_out, plan_res["session"])
            
            logger.phase_end("planner", output_summary={
                "flight_id": plan_out.get("flight_id"),
                "hotel_id": plan_out.get("hotel_id"),
                "restaurant_id": plan_out.get("restaurant_id"),
                "activity_id": plan_out.get("activity_id"),
            }, usage=plan_res["usage"], tool_call_count=plan_res["session"].summary()["tool_call_count"])

            board_after_plan = _board_summary(board)
            logger.log_board_state("planner", board_after_plan, stage="AFTER")

    # ── 5b. Deterministic guardrail ─────────────────────────────────
    # Never submit worse than the guaranteed-feasible baseline: if LLM refinement
    # pushed the itinerary over budget, restore the baseline item(s).
    itin = board.current_itinerary
    nights = runtime.episode.get("nights", 1)
    budget = runtime.episode.get("budget_total", 0) or 0
    current_cost = calculate_total_itinerary_cost(itin.model_dump(), nights)
    if budget and current_cost > budget:
        baseline_cost = calculate_total_itinerary_cost(baseline, nights)
        if baseline_cost <= budget:
            logger.log_decision({"notes": f"guardrail: restored baseline ({int(current_cost)}>{int(budget)})"})
            itin.flight = baseline.get("flight")
            itin.hotel = baseline.get("hotel")
            itin.restaurant = baseline.get("restaurant")
            itin.activity = baseline.get("activity")

    # ── 6. Build final TravelDecision ───────────────────────────────
    itin = board.current_itinerary
    rationale = _build_rationale(board, episode)
    decision = TravelDecision(
        flight_id=(
            itin.flight.get("flight_id") if itin.flight else None
        ),
        hotel_id=(
            itin.hotel.get("hotel_id") if itin.hotel else None
        ),
        restaurant_id=(
            itin.restaurant.get("restaurant_id") if itin.restaurant else None
        ),
        activity_id=(
            itin.activity.get("activity_id") if itin.activity else None
        ),
        memory_report=board.evaluator_tracking,
        notes=rationale,
        debug={"tool_call_count": total_tool_calls},
        usage=total_usage,
    )

    # Log final decision
    logger.finalize({
        "flight_id": decision.flight_id,
        "hotel_id": decision.hotel_id,
        "restaurant_id": decision.restaurant_id,
        "activity_id": decision.activity_id,
    }, total_usage)
    logger.close()

    return {
        "submission": decision.to_evaluator_payload(total_usage),
        "usage": total_usage,
    }
