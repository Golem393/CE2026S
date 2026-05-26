import json
import sys
from pathlib import Path
from typing import Any, Dict

# ANSI COLOR CODES
class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    MEMORY = "\033[38;5;141m"      # Purple
    PLANNER = "\033[38;5;75m"      # Blue
    VERIFIER = "\033[38;5;215m"    # Orange
    ORCHESTRATOR = "\033[38;5;114m" # Green
    SUCCESS = "\033[38;5;114m"     # Green
    WARNING = "\033[38;5;220m"     # Yellow
    ERROR = "\033[38;5;203m"       # Red
    INFO = "\033[38;5;247m"        # Gray
    HEADER = "\033[38;5;255m"      # White
    TOOL = "\033[38;5;180m"        # Tan

_AGENT_EMOJI = {
    "memory": "🧠",
    "planner": "📋",
    "verifier": "✅",
    "orchestrator": "🎯",
    "rule_checker": "🔍",
}

_AGENT_COLOR = {
    "memory": _C.MEMORY,
    "planner": _C.PLANNER,
    "verifier": _C.VERIFIER,
    "orchestrator": _C.ORCHESTRATOR,
    "rule_checker": _C.WARNING,
}

_ROLE_TO_AGENT = {
    "mas_memory_manager": "memory",
    "mas_planner": "planner",
    "mas_verifier": "verifier",
}

def _truncate(text: str, max_len: int = 120) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."

def _format_dict_compact(d: Dict[str, Any], max_items: int = 8) -> str:
    if not d:
        return "{}"
    items = list(d.items())[:max_items]
    parts = [f"{k}={_truncate(str(v), 40)}" for k, v in items]
    suffix = f" +{len(d) - max_items} more" if len(d) > max_items else ""
    return ", ".join(parts) + suffix

def find_submission_in_json(log_file_path: str, trip_id: str) -> Dict[str, Any] | None:
    if not log_file_path:
        return None
    log_path = Path(log_file_path)
    
    # Candidate paths for result JSONs
    candidates = [
        log_path.parent / "llm_results_public_v2.json",
        Path("api_student5/llm_results_public_v2.json"),
    ]
    # Also look at any json files in log_path's grandparents or siblings
    try:
        for sibling in log_path.parent.glob("*.json"):
            candidates.append(sibling)
    except Exception:
        pass
    try:
        if Path("api_student5").exists():
            for sibling in Path("api_student5").glob("*.json"):
                candidates.append(sibling)
    except Exception:
        pass
    try:
        for child in Path(".").glob("**/llm_results*.json"):
            candidates.append(child)
    except Exception:
        pass

    # Deduplicate candidates while keeping order
    seen = set()
    deduped = []
    for c in candidates:
        abs_p = c.resolve() if c.exists() else None
        if abs_p and abs_p not in seen:
            seen.add(abs_p)
            deduped.append(c)

    for path in deduped:
        try:
            data = json.loads(path.read_text())
            systems = data.get("systems", {})
            if systems:
                for system_data in systems.values():
                    for res in system_data.get("results", []):
                        if res.get("trip_id") == trip_id:
                            return res.get("submission")
            else:
                # Maybe it's a list or directly contains results
                results = data.get("results", [])
                for res in results:
                    if res.get("trip_id") == trip_id:
                        return res.get("submission")
        except Exception:
            pass
    return None

def print_log_line(entry: Dict[str, Any], log_file_path: str = ""):
    event = entry.get("event")
    
    if event == "episode_start":
        trip_id = entry.get("trip_id", "?")
        print("")
        print(f"{_C.ORCHESTRATOR}{_C.BOLD}╔═══════════════════════════════════════════════════════╗{_C.RESET}")
        print(f"{_C.ORCHESTRATOR}{_C.BOLD}║  🎯 EPISODE START: {trip_id:<36s}║{_C.RESET}")
        print(f"{_C.ORCHESTRATOR}{_C.BOLD}╚═══════════════════════════════════════════════════════╝{_C.RESET}")
        
        # In trace.jsonl, episode details are in the json public file, but we can print what we have
        if "city" in entry:
            print(f"   {_C.DIM}city={_C.RESET}{entry.get('city', '?')}  {_C.DIM}origin={_C.RESET}{entry.get('origin', '?')}  {_C.DIM}family={_C.RESET}{entry.get('family', '?')}")
            print(f"   {_C.DIM}budget={_C.RESET}{entry.get('budget', '?')}  {_C.DIM}nights={_C.RESET}{entry.get('nights', '?')}  {_C.DIM}zone={_C.RESET}{entry.get('meeting_zone', '?')}  {_C.DIM}weather={_C.RESET}{entry.get('weather', '?')}")

    elif event == "phase_start" or event == "tool_agent_start":
        role = entry.get("role") or entry.get("agent", "?")
        agent = _ROLE_TO_AGENT.get(role, role)
        iteration = entry.get("iteration", 1)
        trip_id = entry.get("trip_id", "?")
        emoji = _AGENT_EMOJI.get(agent, "🔄")
        color = _AGENT_COLOR.get(agent, _C.INFO)
        agent_upper = agent.upper().replace("_", " ")

        print("")
        print(f"{_C.HEADER}{_C.BOLD}═══════════════════════════════════════════════════════{_C.RESET}")
        print(f"{color}{_C.BOLD}{emoji} {agent_upper} AGENT — Iteration {iteration}  {_C.DIM}[trip: {trip_id}]{_C.RESET}")
        print(f"{_C.HEADER}{_C.BOLD}═══════════════════════════════════════════════════════{_C.RESET}")

        input_summary = entry.get("input_summary")
        if input_summary:
            print(f"{_C.INFO}📥 Input:{_C.RESET}")
            for key, value in input_summary.items():
                val_str = _truncate(str(value), 100)
                print(f"   {_C.DIM}{key}={_C.RESET}{val_str}")

    elif event == "tool_agent_tool_call":
        tool = entry.get("tool", "?")
        args = entry.get("arguments", {})
        args_str = ", ".join(f"{k}={_truncate(str(v), 50)}" for k, v in args.items() if v is not None)
        print(f"   {_C.TOOL}🔧 Tool Call: {tool}({args_str}){_C.RESET}")

    elif event == "tool_result":
        preview = entry.get("preview", "")
        print(f"   {_C.INFO}   📥 Preview: {preview}{_C.RESET}")

    elif event == "phase_end" or event == "tool_agent_finish":
        role = entry.get("role") or entry.get("agent", "?")
        agent = _ROLE_TO_AGENT.get(role, role)
        color = _AGENT_COLOR.get(agent, _C.INFO)
        output_summary = entry.get("output_summary")
        
        if output_summary:
            print(f"{color}📤 Output:{_C.RESET}")
            for key, value in output_summary.items():
                if isinstance(value, list):
                    val_str = str(value[:6])
                    if len(value) > 6:
                        val_str += f" +{len(value)-6} more"
                else:
                    val_str = _truncate(str(value), 100)
                print(f"   {_C.DIM}{key}:{_C.RESET} {val_str}")

        usage = entry.get("usage", {})
        duration = entry.get("duration_s", 0) or entry.get("elapsed_s", 0)
        tool_call_count = entry.get("tool_call_count", 0)
        tokens = usage.get("total_tokens", 0) or entry.get("tokens", 0)
        cost = usage.get("estimated_cost_usd", 0.0) or entry.get("cost_usd", 0.0)
        
        print(f"{_C.DIM}⏱  Duration: {duration:.1f}s | Tools: {tool_call_count} | Tokens: {tokens} | Cost: ${cost:.6f}{_C.RESET}")
        print(f"{_C.DIM}───────────────────────────────────────────────────────{_C.RESET}")

    elif event == "board_state":
        print(f"{_C.INFO}📊 Board State:{_C.RESET}")
        board_summary = entry.get("board", {})
        for key, value in board_summary.items():
            if isinstance(value, list):
                if value:
                    val_str = str(value[:4])
                    if len(value) > 4:
                        val_str += f" +{len(value)-4} more"
                else:
                    val_str = "[]"
            else:
                val_str = _truncate(str(value), 80)
            print(f"   {_C.DIM}{key}:{_C.RESET} {val_str}")

    elif event == "violations":
        agent = entry.get("agent", "?")
        violations = entry.get("violations", [])
        if not violations:
            print(f"{_C.SUCCESS}✓ No violations found by {agent}{_C.RESET}")
        else:
            print(f"{_C.ERROR}{_C.BOLD}⚠ {len(violations)} violation(s) found by {agent}:{_C.RESET}")
            for v in violations:
                color = _C.ERROR if "VIOLATION" in v else _C.WARNING
                print(f"   {color}• {v}{_C.RESET}")

    elif event == "verifier_result":
        approved = entry.get("approved", False)
        issues = entry.get("issues", [])
        if approved:
            print(f"{_C.SUCCESS}{_C.BOLD}✅ VERIFIER: APPROVED{_C.RESET}")
        else:
            print(f"{_C.ERROR}{_C.BOLD}❌ VERIFIER: REJECTED{_C.RESET}")
            for issue in issues:
                print(f"   {_C.ERROR}• {issue}{_C.RESET}")

    elif event == "episode_finalize" or event == "episode_success":
        total_elapsed = entry.get("total_elapsed_s", 0) or entry.get("duration_s", 0)
        tokens = entry.get("total_tokens", 0) or entry.get("tokens", 0)
        cost = entry.get("total_cost_usd", 0.0) or entry.get("cost_usd", 0.0)
        trip_id = entry.get("trip_id", "?")
        
        print("")
        print(f"{_C.ORCHESTRATOR}{_C.BOLD}╔═══════════════════════════════════════════════════════╗{_C.RESET}")
        print(f"{_C.ORCHESTRATOR}{_C.BOLD}║  🏁 FINAL DECISION                                  ║{_C.RESET}")
        print(f"{_C.ORCHESTRATOR}{_C.BOLD}╚═══════════════════════════════════════════════════════╝{_C.RESET}")

        submission = find_submission_in_json(log_file_path, trip_id)
        if submission:
            for key in ["flight_id", "hotel_id", "restaurant_id", "activity_id"]:
                val = submission.get(key, "null")
                status = _C.SUCCESS if val and val != "null" else _C.ERROR
                symbol = "✓" if val and val != "null" else "✗"
                print(f"   {status}{_C.BOLD}{symbol} {key}: {val}{_C.RESET}")
        else:
            for key in ["flight_id", "hotel_id", "restaurant_id", "activity_id"]:
                val = entry.get(key, "null")
                status = _C.SUCCESS if val and val != "null" else _C.ERROR
                symbol = "✓" if val and val != "null" else "✗"
                print(f"   {status}{_C.BOLD}{symbol} {key}: {val}{_C.RESET}")

        dq = entry.get("decision_quality")
        dq_str = f" | Decision Quality: {dq}" if dq is not None else ""
        print(f"\n   {_C.DIM}Total: {total_elapsed:.1f}s | {tokens} tokens | ${cost:.6f}{dq_str}{_C.RESET}\n")

def main():
    log_file = "runs/agent_debug_log.jsonl"
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
        
    path = Path(log_file)
    if not path.exists():
        print(f"Log file {log_file} not found.")
        sys.exit(1)
        
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    print_log_line(entry, log_file)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON: {line}")
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
