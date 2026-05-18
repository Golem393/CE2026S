from __future__ import annotations

"""Smart Agent Logger for multi-agent travel planner.

Provides step-by-step, color-coded console output and structured JSON logs
so you can see exactly which agent is running, what it's doing, what went
in and what came out.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════
#  ANSI COLOR CODES
# ═══════════════════════════════════════════════════════════════════════

class _C:
    """ANSI color constants for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Agent colors
    MEMORY = "\033[38;5;141m"      # Purple
    PLANNER = "\033[38;5;75m"      # Blue
    VERIFIER = "\033[38;5;215m"    # Orange
    ORCHESTRATOR = "\033[38;5;114m" # Green

    # Status colors
    SUCCESS = "\033[38;5;114m"     # Green
    WARNING = "\033[38;5;220m"     # Yellow
    ERROR = "\033[38;5;203m"       # Red
    INFO = "\033[38;5;247m"        # Gray

    # Decoration
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


def _truncate(text: str, max_len: int = 120) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_dict_compact(d: Dict[str, Any], max_items: int = 8) -> str:
    """Format a dict compactly for one-line display."""
    if not d:
        return "{}"
    items = list(d.items())[:max_items]
    parts = [f"{k}={_truncate(str(v), 40)}" for k, v in items]
    suffix = f" +{len(d) - max_items} more" if len(d) > max_items else ""
    return ", ".join(parts) + suffix


# ═══════════════════════════════════════════════════════════════════════
#  SMART AGENT LOGGER
# ═══════════════════════════════════════════════════════════════════════

class SmartAgentLogger:
    """Step-by-step logger for multi-agent debugging.

    Usage:
        logger = SmartAgentLogger(trip_id="rtl7_public_easy_001")

        logger.phase_start("memory", iteration=1, input_summary={...})
        # ... run agent ...
        logger.phase_end("memory", output_summary={...}, usage={...})

        logger.log_tool_calls("memory", tool_trace=[...])
        logger.log_board_delta("memory", before={...}, after={...})
        logger.log_violations("rule_checker", violations=[...])

        logger.finalize(decision={...})
    """

    def __init__(
        self,
        trip_id: str = "",
        log_dir: str = "runs",
        console: bool = True,
        file_log: bool = True,
    ) -> None:
        self.trip_id = trip_id
        self.console = console
        self._log_entries: List[Dict[str, Any]] = []
        self._phase_start_time: float = 0.0
        self._episode_start_time: float = time.perf_counter()

        # File logging
        self._file_handle = None
        if file_log:
            log_path = Path(log_dir) / "agent_debug_log.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = log_path.open("a", encoding="utf-8")

    # ── Console Output ───────────────────────────────────────────────

    def _print(self, text: str) -> None:
        if self.console:
            print(text, flush=True)

    def _write_json(self, entry: Dict[str, Any]) -> None:
        self._log_entries.append(entry)
        if self._file_handle:
            self._file_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._file_handle.flush()

    # ── Phase Logging ────────────────────────────────────────────────

    def phase_start(
        self,
        agent: str,
        *,
        iteration: int = 1,
        input_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log the start of an agent phase."""
        self._phase_start_time = time.perf_counter()
        emoji = _AGENT_EMOJI.get(agent, "🔄")
        color = _AGENT_COLOR.get(agent, _C.INFO)
        agent_upper = agent.upper().replace("_", " ")

        self._print("")
        self._print(
            f"{_C.HEADER}{_C.BOLD}"
            f"═══════════════════════════════════════════════════════"
            f"{_C.RESET}"
        )
        self._print(
            f"{color}{_C.BOLD}"
            f"{emoji} {agent_upper} AGENT — Iteration {iteration}"
            f"  {_C.DIM}[trip: {self.trip_id}]"
            f"{_C.RESET}"
        )
        self._print(
            f"{_C.HEADER}{_C.BOLD}"
            f"═══════════════════════════════════════════════════════"
            f"{_C.RESET}"
        )

        if input_summary:
            self._print(f"{_C.INFO}📥 Input:{_C.RESET}")
            for key, value in input_summary.items():
                val_str = _truncate(str(value), 100)
                self._print(f"   {_C.DIM}{key}={_C.RESET}{val_str}")

        self._write_json({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "phase_start",
            "trip_id": self.trip_id,
            "agent": agent,
            "iteration": iteration,
            "input_summary": _safe_serialize(input_summary),
        })

    def phase_end(
        self,
        agent: str,
        *,
        output_summary: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, Any]] = None,
        tool_call_count: int = 0,
    ) -> None:
        """Log the end of an agent phase."""
        duration = time.perf_counter() - self._phase_start_time
        color = _AGENT_COLOR.get(agent, _C.INFO)

        if output_summary:
            self._print(f"{color}📤 Output:{_C.RESET}")
            for key, value in output_summary.items():
                if isinstance(value, list):
                    val_str = str(value[:6])
                    if len(value) > 6:
                        val_str += f" +{len(value)-6} more"
                else:
                    val_str = _truncate(str(value), 100)
                self._print(f"   {_C.DIM}{key}:{_C.RESET} {val_str}")

        # Timing and cost line
        tokens = usage.get("total_tokens", 0) if usage else 0
        cost = usage.get("estimated_cost_usd", 0.0) if usage else 0.0
        self._print(
            f"{_C.DIM}⏱  Duration: {duration:.1f}s | "
            f"Tools: {tool_call_count} | "
            f"Tokens: {tokens} | "
            f"Cost: ${cost:.6f}{_C.RESET}"
        )
        self._print(
            f"{_C.DIM}"
            f"───────────────────────────────────────────────────────"
            f"{_C.RESET}"
        )

        self._write_json({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "phase_end",
            "trip_id": self.trip_id,
            "agent": agent,
            "duration_s": round(duration, 3),
            "tool_call_count": tool_call_count,
            "tokens": tokens,
            "cost_usd": cost,
            "output_summary": _safe_serialize(output_summary),
        })

    # ── Tool Call Logging ────────────────────────────────────────────

    def log_tool_calls(
        self,
        agent: str,
        tool_trace: List[Dict[str, Any]],
    ) -> None:
        """Log a summary of tool calls made by an agent."""
        if not tool_trace:
            return

        self._print(f"{_C.TOOL}🔧 Tools called: {len(tool_trace)}{_C.RESET}")
        for t in tool_trace[:10]:
            name = t.get("tool", "?")
            args = t.get("arguments", {})
            preview = t.get("preview", "")

            # Compact arg summary
            arg_parts = []
            for k, v in list(args.items())[:3]:
                if v is not None and v != [] and v != "":
                    arg_parts.append(f"{v}")
            arg_str = ", ".join(arg_parts) if arg_parts else ""

            preview_str = _truncate(str(preview), 60)
            self._print(
                f"   {_C.DIM}→{_C.RESET} {_C.TOOL}{name}{_C.RESET}"
                f"({arg_str}) → {_C.DIM}{preview_str}{_C.RESET}"
            )

        if len(tool_trace) > 10:
            self._print(
                f"   {_C.DIM}... and {len(tool_trace) - 10} more{_C.RESET}"
            )

    # ── Board State Logging ──────────────────────────────────────────

    def log_board_state(
        self,
        agent: str,
        board_summary: Dict[str, Any],
    ) -> None:
        """Log the current state of the working memory board."""
        self._print(f"{_C.INFO}📊 Board State:{_C.RESET}")
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
            self._print(f"   {_C.DIM}{key}:{_C.RESET} {val_str}")

        self._write_json({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "board_state",
            "trip_id": self.trip_id,
            "agent": agent,
            "board": _safe_serialize(board_summary),
        })

    # ── Violation / Rule Check Logging ───────────────────────────────

    def log_violations(
        self,
        agent: str,
        violations: List[str],
    ) -> None:
        """Log constraint violations found by the automated rule checker."""
        if not violations:
            self._print(
                f"{_C.SUCCESS}✓ No violations found by {agent}{_C.RESET}"
            )
            return

        self._print(
            f"{_C.ERROR}{_C.BOLD}⚠ {len(violations)} violation(s) "
            f"found by {agent}:{_C.RESET}"
        )
        for v in violations:
            color = _C.ERROR if "VIOLATION" in v else _C.WARNING
            self._print(f"   {color}• {v}{_C.RESET}")

        self._write_json({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "violations",
            "trip_id": self.trip_id,
            "agent": agent,
            "count": len(violations),
            "violations": violations,
        })

    # ── Decision Logging ─────────────────────────────────────────────

    def log_decision(
        self,
        decision: Dict[str, Any],
    ) -> None:
        """Log the planner's current proposal."""
        self._print(f"{_C.PLANNER}📝 Planner Proposal:{_C.RESET}")
        for key in ["flight_id", "hotel_id", "restaurant_id", "activity_id"]:
            val = decision.get(key, "null")
            status = _C.SUCCESS if val else _C.WARNING
            symbol = "✓" if val else "✗"
            self._print(f"   {status}{symbol} {key}: {val}{_C.RESET}")

        notes = decision.get("notes", "")
        if notes:
            self._print(f"   {_C.DIM}notes: {_truncate(notes, 80)}{_C.RESET}")

    def log_verifier_result(
        self,
        approved: bool,
        issues: List[str],
    ) -> None:
        """Log the verifier's verdict."""
        if approved:
            self._print(
                f"{_C.SUCCESS}{_C.BOLD}✅ VERIFIER: APPROVED{_C.RESET}"
            )
        else:
            self._print(
                f"{_C.ERROR}{_C.BOLD}❌ VERIFIER: REJECTED{_C.RESET}"
            )
            for issue in issues:
                self._print(f"   {_C.ERROR}• {issue}{_C.RESET}")

        self._write_json({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "verifier_result",
            "trip_id": self.trip_id,
            "approved": approved,
            "issues": issues,
        })

    # ── Episode-Level Logging ────────────────────────────────────────

    def episode_start(self, episode: Dict[str, Any]) -> None:
        """Log the start of an episode."""
        total_elapsed = time.perf_counter() - self._episode_start_time

        self._print("")
        self._print(
            f"{_C.ORCHESTRATOR}{_C.BOLD}"
            f"╔═══════════════════════════════════════════════════════╗"
            f"{_C.RESET}"
        )
        self._print(
            f"{_C.ORCHESTRATOR}{_C.BOLD}"
            f"║  🎯 EPISODE START: {self.trip_id:<36s}║"
            f"{_C.RESET}"
        )
        self._print(
            f"{_C.ORCHESTRATOR}{_C.BOLD}"
            f"╚═══════════════════════════════════════════════════════╝"
            f"{_C.RESET}"
        )

        # Key episode facts
        self._print(f"   {_C.DIM}city={_C.RESET}{episode.get('city', '?')}"
                     f"  {_C.DIM}origin={_C.RESET}{episode.get('origin', '?')}"
                     f"  {_C.DIM}family={_C.RESET}{episode.get('family', '?')}")
        self._print(f"   {_C.DIM}budget={_C.RESET}{episode.get('budget_total', '?')}"
                     f"  {_C.DIM}nights={_C.RESET}{episode.get('nights', '?')}"
                     f"  {_C.DIM}zone={_C.RESET}{episode.get('meeting_zone', '?')}"
                     f"  {_C.DIM}weather={_C.RESET}{episode.get('weather', '?')}")

        hooks = episode.get("scenario_hooks", {})
        if hooks:
            self._print(
                f"   {_C.DIM}hooks={_C.RESET}"
                f"{_format_dict_compact(hooks, 4)}"
            )

        self._write_json({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "episode_start",
            "trip_id": self.trip_id,
            "city": episode.get("city"),
            "origin": episode.get("origin"),
            "family": episode.get("family"),
            "budget": episode.get("budget_total"),
            "nights": episode.get("nights"),
            "meeting_zone": episode.get("meeting_zone"),
            "weather": episode.get("weather"),
        })

    def finalize(
        self,
        decision: Dict[str, Any],
        total_usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log the final decision at episode end."""
        total_elapsed = time.perf_counter() - self._episode_start_time

        self._print("")
        self._print(
            f"{_C.ORCHESTRATOR}{_C.BOLD}"
            f"╔═══════════════════════════════════════════════════════╗"
            f"{_C.RESET}"
        )
        self._print(
            f"{_C.ORCHESTRATOR}{_C.BOLD}"
            f"║  🏁 FINAL DECISION                                  ║"
            f"{_C.RESET}"
        )
        self._print(
            f"{_C.ORCHESTRATOR}{_C.BOLD}"
            f"╚═══════════════════════════════════════════════════════╝"
            f"{_C.RESET}"
        )

        for key in ["flight_id", "hotel_id", "restaurant_id", "activity_id"]:
            val = decision.get(key, "null")
            status = _C.SUCCESS if val else _C.ERROR
            symbol = "✓" if val else "✗"
            self._print(f"   {status}{_C.BOLD}{symbol} {key}: {val}{_C.RESET}")

        if total_usage:
            tokens = total_usage.get("total_tokens", 0)
            cost = total_usage.get("estimated_cost_usd", 0.0)
            self._print(
                f"\n   {_C.DIM}Total: {total_elapsed:.1f}s | "
                f"{tokens} tokens | ${cost:.6f}{_C.RESET}"
            )

        self._print("")

        self._write_json({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "episode_finalize",
            "trip_id": self.trip_id,
            "flight_id": decision.get("flight_id"),
            "hotel_id": decision.get("hotel_id"),
            "restaurant_id": decision.get("restaurant_id"),
            "activity_id": decision.get("activity_id"),
            "total_elapsed_s": round(total_elapsed, 3),
            "total_tokens": total_usage.get("total_tokens", 0) if total_usage else 0,
            "total_cost_usd": total_usage.get("estimated_cost_usd", 0.0) if total_usage else 0.0,
        })

        # Close file handle
        if self._file_handle:
            self._file_handle.flush()

    def close(self) -> None:
        """Close the log file handle."""
        if self._file_handle:
            self._file_handle.flush()
            self._file_handle.close()
            self._file_handle = None


def _safe_serialize(obj: Any) -> Any:
    """Make an object JSON-safe by truncating/converting."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {
            str(k): _safe_serialize(v)
            for k, v in list(obj.items())[:20]
        }
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in list(obj)[:10]]
    return str(obj)[:200]
