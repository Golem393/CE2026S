from __future__ import annotations

import json
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
    planner_feedback: str = "",
) -> str:
    parts = [episode_prompt(episode)]
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
        "Prefer focused searches over broad browsing. "
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
    if td.scenario_hooks:
        parts.append(f"scenario_hooks: {json.dumps(td.scenario_hooks)}")
    if td.scenario_state:
        parts.append(f"scenario_state: {json.dumps(td.scenario_state)}")

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
        "Cross-reference the proposed itinerary against ALL constraints: "
        "budget, zone, quiet/noise, dietary, red-eye, refundability, weather, spoken rules. "
        "Use tools to look up specific items if needed. "
        "Set approve=true if compliant. If ANY violation, set approve=false with specific issues."
    )


def _verifier_input(board: WorkingMemoryBoard) -> str:
    td = board.trip_details
    itin = board.current_itinerary
    parts = [
        "--- TRIP ---",
        f"city: {td.city}, origin: {td.origin}, family: {td.family}",
        f"budget: {td.budget_total}, nights: {td.nights}",
        f"meeting_zone: {td.meeting_zone}, weather: {td.weather}",
    ]
    if td.scenario_state:
        parts.append(f"scenario_state: {json.dumps(td.scenario_state)}")
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
) -> Dict[str, Any]:
    """Generic wrapper: create session, run tool agent, return parsed + usage + session."""
    config = runtime.system_config
    model = config["model"]

    session = runtime.new_session(
        role=role,
        max_results=config.get("max_tool_results", 4),
    )

    result = runtime.runner.run_tool_agent_json(
        model=model,
        instructions=instructions,
        input_text=input_text,
        json_schema=json_schema,
        schema_name=schema_name,
        tools=session.tool_specs(primitive_only=False),
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
    planner_feedback: str = "",
) -> Dict[str, Any]:
    cfg = runtime.system_config
    return _call_agent(
        runtime,
        role="mas_memory_manager",
        instructions=_memory_agent_instructions(),
        input_text=_memory_agent_input(runtime.episode, board, planner_feedback),
        json_schema=context_pack_schema(),
        schema_name="memory_context_pack",
        max_output_tokens=cfg.get("memory_manager_max_output_tokens", 1000),
        max_tool_rounds=cfg.get("memory_manager_max_tool_rounds", 8),
    )


def _call_planner_agent(
    runtime: StudentRuntime,
    board: WorkingMemoryBoard,
) -> Dict[str, Any]:
    cfg = runtime.system_config
    return _call_agent(
        runtime,
        role="mas_planner",
        instructions=_planner_instructions(),
        input_text=_planner_input(board),
        json_schema=planner_schema(),
        schema_name="planner_proposal",
        max_output_tokens=cfg.get("planner_max_output_tokens", 800),
        max_tool_rounds=cfg.get("planner_max_tool_rounds", 12),
    )


def _call_verifier_agent(
    runtime: StudentRuntime,
    board: WorkingMemoryBoard,
) -> Dict[str, Any]:
    cfg = runtime.system_config
    return _call_agent(
        runtime,
        role="mas_verifier",
        instructions=_verifier_instructions(),
        input_text=_verifier_input(board),
        json_schema=verifier_schema(),
        schema_name="verifier_check",
        max_output_tokens=cfg.get("verifier_max_output_tokens", 600),
        max_tool_rounds=cfg.get("verifier_max_tool_rounds", 6),
    )


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
    # Hard constraints from critical_constraints
    for c in context_pack.get("critical_constraints", []):
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
    board: WorkingMemoryBoard,
    planner_out: Dict[str, Any],
) -> str:
    """Save Planner IDs to itinerary; return feedback string."""
    feedback = []
    itin = board.current_itinerary

    for key, attr in [
        ("flight_id", "flight"),
        ("hotel_id", "hotel"),
        ("restaurant_id", "restaurant"),
        ("activity_id", "activity"),
    ]:
        proposed_id = planner_out.get(key)
        current = getattr(itin, attr)
        if proposed_id and not current:
            setattr(itin, attr, {key: proposed_id})
            feedback.append(f"{attr} {proposed_id} booked.")
        elif not proposed_id and not current:
            feedback.append(f"{attr} search FAILED.")

    notes = planner_out.get("notes", "")
    if notes:
        feedback.append(f"Planner notes: {notes}")
    return "\n".join(feedback)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════

def solve_episode(runtime: StudentRuntime) -> Dict[str, Any]:
    """3-agent solver: Memory → Planner → Verifier loop."""
    config = runtime.system_config
    episode = runtime.episode
    max_revision_rounds = config.get("max_revision_rounds", 1)

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
    mem_res = _call_memory_agent(runtime, board)
    _update_board_from_memory(board, mem_res["parsed"], mem_res["session"], episode)
    total_usage = runtime.combine_usages(total_usage, mem_res["usage"])
    total_tool_calls += mem_res["session"].summary()["tool_call_count"]

    # ── 3. Planner Loop (max 4 iterations) ──────────────────────────
    MAX_PLANNER_ITERS = 4
    for planner_iter in range(MAX_PLANNER_ITERS):
        plan_res = _call_planner_agent(runtime, board)
        plan_out = plan_res["parsed"]
        total_usage = runtime.combine_usages(total_usage, plan_res["usage"])
        total_tool_calls += plan_res["session"].summary()["tool_call_count"]

        feedback = _save_planner_proposals(board, plan_out)

        if _itinerary_complete(board) or _is_give_up(plan_out):
            break

        # Not done yet — update memory with feedback before next planner pass
        if planner_iter < MAX_PLANNER_ITERS - 1:
            mem_res = _call_memory_agent(runtime, board, planner_feedback=feedback)
            _update_board_from_memory(
                board, mem_res["parsed"], mem_res["session"], episode
            )
            total_usage = runtime.combine_usages(total_usage, mem_res["usage"])
            total_tool_calls += mem_res["session"].summary()["tool_call_count"]

    # ── 4. Verifier Loop ────────────────────────────────────────────
    revision_count = 0
    if _itinerary_complete(board):
        while True:
            ver_res = _call_verifier_agent(runtime, board)
            ver_out = ver_res["parsed"]
            total_usage = runtime.combine_usages(total_usage, ver_res["usage"])
            total_tool_calls += ver_res["session"].summary()["tool_call_count"]

            # Merge verifier retire into evaluator tracking
            for r in canonicalize_context_keys(
                ver_out.get("retire", []), keep_doc_ids=False
            ):
                if r and r not in board.evaluator_tracking.retired:
                    board.evaluator_tracking.retired.append(r)

            if ver_out.get("approve", False):
                break  # ✅ Approved

            # ❌ Rejected — record reason
            issues = ver_out.get("issues", [])
            reason = "; ".join(issues) if issues else "unspecified"
            board.failed_searches.append(f"verifier_rejected: {reason}")
            revision_count += 1

            if revision_count >= max_revision_rounds:
                break  # Submit best-effort

            # One more planner pass to fix the issue
            plan_res = _call_planner_agent(runtime, board)
            plan_out = plan_res["parsed"]
            total_usage = runtime.combine_usages(total_usage, plan_res["usage"])
            total_tool_calls += plan_res["session"].summary()["tool_call_count"]

            # Overwrite items if planner proposes replacements
            itin = board.current_itinerary
            if plan_out.get("flight_id"):
                itin.flight = {"flight_id": plan_out["flight_id"]}
            if plan_out.get("hotel_id"):
                itin.hotel = {"hotel_id": plan_out["hotel_id"]}
            if plan_out.get("restaurant_id"):
                itin.restaurant = {"restaurant_id": plan_out["restaurant_id"]}
            if plan_out.get("activity_id"):
                itin.activity = {"activity_id": plan_out["activity_id"]}

    # ── 5. Build final TravelDecision ───────────────────────────────
    itin = board.current_itinerary
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
        debug={"tool_call_count": total_tool_calls},
        usage=total_usage,
    )

    return {
        "submission": decision.to_evaluator_payload(total_usage),
        "usage": total_usage,
    }
