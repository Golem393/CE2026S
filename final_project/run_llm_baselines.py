from __future__ import annotations

import argparse
import importlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_csv_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]

from budget_knobs import apply_budget_overrides, validate_system_budget_caps
from dynamic_travel_replanning.evaluator import evaluate_episode, summarize_rows
from llm_agents import run_memory_single, run_single_baseline
from llm_runner import LLMRunner
from llm_tools import TravelToolbox
from runtime_api import StudentRuntime
from llm_tools import set_active_student_runtime, reset_active_student_runtime
from trace_logger import TraceLogger


SYSTEM_RUNNERS = {
    "llm_single_baseline": run_single_baseline,
    "llm_memory_single": run_memory_single,
}


def _has_module(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


DYNAMIC_SOLVER_DEFAULTS = {
    "student_solver_example": {
        "solver_module": "student_solver_example",
        "system_name": "student_solver_example",
    },
    "student_solver": {
        "solver_module": "student_solver",
        "system_name": "student_solver",
    },
}

if _has_module("ta_only.student_solver2"):
    DYNAMIC_SOLVER_DEFAULTS["student_solver2"] = {
        "solver_module": "ta_only.student_solver2",
        "system_name": "student_solver2",
    }


_SOLVER_HIDDEN_DROP_KEYS = {
    "gold",
    "scenario_state",
    "scenario_hooks",
    "required_hard",
    "soft_tags",
    "acceptable_flights",
    "acceptable_hotels",
    "acceptable_restaurants",
    "acceptable_activities",
    "required_docs",
    "distractor_docs",
    "distractor_docs_to_avoid",
    "stale_docs_to_retire",
    "benchmark_family",
    "private_blueprint_note",
    "travel_analogue_family",
    # Hidden-only: avoid exposing evaluator-only state flags while keeping
    # released observable trip context fields (`budget_total`, `meeting_zone`,
    # and `weather`) solver-visible for backward compatibility. The same
    # context is also expressed below as natural language.
}


def _hidden_context_turn(episode: Dict[str, Any]) -> Dict[str, str]:
    budget = episode.get("budget_total")
    zone = episode.get("meeting_zone") or episode.get("preferred_zone")
    weather = str(episode.get("weather") or "").lower()
    parts = []
    if budget:
        parts.append(f"keep the full package within roughly {int(budget):,} KRW")
    if zone:
        parts.append(f"treat {zone} as the main work-anchor area")
    if weather in {"rainy", "storm", "wet"}:
        parts.append("assume the forecast now favors covered or indoor fallbacks")
    elif weather in {"sunny", "clear", "dry"}:
        parts.append("assume the forecast does not require avoiding outdoor options")
    elif weather:
        parts.append(f"account for the current forecast described as {weather}")
    text = "For this hidden run, the latest user-provided context is to " + "; ".join(parts) + "." if parts else "For this hidden run, rely on the latest user turns and tool evidence."
    return {"speaker": "user", "text": text}


def sanitize_episode_for_solver(episode: Dict[str, Any], *, hidden: bool = False) -> Dict[str, Any]:
    """Return the episode object exposed to student solvers.

    Public episodes intentionally remain transparent so students can inspect the
    benchmark design and debug against the public evaluator. Hidden evaluation,
    however, should test whether solvers infer constraints from user turns and
    tool-accessible evidence rather than reading evaluator-only state flags or
    structured hard-clue fields. The full hidden episode is still passed to the
    evaluator for scoring.
    """
    clean = deepcopy(episode)
    if not hidden:
        return clean
    context_turn = _hidden_context_turn(episode)
    for key in _SOLVER_HIDDEN_DROP_KEYS:
        clean.pop(key, None)
    turns = list(clean.get("turns") or [])
    clean["turns"] = [context_turn] + turns
    clean["solver_visible_note"] = (
        "Hidden evaluation exposes user turns and tool-accessible trip data. "
        "Evaluator-only state/hooks/gold metadata are withheld from the solver; observable trip context remains available for compatibility."
    )
    return clean


def _load_solver_callable(module_name: str):
    module = importlib.import_module(module_name)
    solve = getattr(module, "solve_episode", None)
    if solve is None:
        raise ValueError(f"Module '{module_name}' must define solve_episode(runtime).")
    return solve


def _run_dynamic_solver(
    runner: LLMRunner,
    toolbox: TravelToolbox,
    episode: Dict[str, Any],
    system_config: Dict[str, Any],
) -> Dict[str, Any]:
    module_name = system_config["solver_module"]
    runtime = StudentRuntime(
        runner=runner,
        toolbox=toolbox,
        episode=episode,
        system_config=system_config,
        role=system_config.get("system_name", module_name),
    )
    token = set_active_student_runtime(runtime)
    try:
        result = _load_solver_callable(module_name)(runtime)
    finally:
        reset_active_student_runtime(token)
    if not isinstance(result, dict) or "submission" not in result or "usage" not in result:
        raise ValueError(
            f"{module_name}.solve_episode(runtime) must return a dict with at least 'submission' and 'usage'."
        )
    result.setdefault("response_ids", [])
    runtime_trace = runtime.trace_summary()
    # Official scoring uses harness-observed tool traces; solver-provided traces
    # are kept only for debugging/backward-compatible result dumps.
    result.setdefault("tool_trace", runtime_trace.get("tool_trace", []))
    result.setdefault("retrieval", {})
    result["official_trace"] = runtime_trace
    result.setdefault("api_status", {"success": True})
    return result


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def sample_hidden_episodes(episodes: List[Dict[str, Any]], size: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    return sorted(rng.sample(episodes, size), key=lambda row: row["trip_id"])


def resolve_gold(episode: Dict[str, Any], gold_map: Dict[str, Dict[str, Any]] | None) -> Dict[str, Any]:
    if "gold" in episode:
        return episode["gold"]
    if gold_map is None:
        raise ValueError(f"No gold available for {episode['trip_id']}")
    return gold_map[episode["trip_id"]]


def merge_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        merged[key] = value
    return merged



def attach_debug(submission: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(submission)
    payload["debug"] = {
        "tool_call_count": run_result.get("retrieval", {}).get("tool_call_count", len(run_result.get("tool_trace", []))),
        "tool_trace": run_result.get("tool_trace", []),
    }
    return payload


def build_failed_row(episode: Dict[str, Any], error: str) -> Dict[str, Any]:
    zero_metrics = {
        "decision_quality": 0.0,
        "hard_constraint_rate": 0.0,
        "semantic_fit_rate": 0.0,
        "bundle_coherence_rate": 0.0,
        "update_handling_rate": 0.0,
        "memory_retrieval_rate": 0.0,
        "memory_retirement_rate": 0.0,
        "distributed_context_rate": 0.0,
        "stale_doc_retirement_rate": 0.0,
        "distractor_avoidance_rate": 0.0,
        "rejected_option_memory_rate": 0.0,
        "active_context_hygiene_rate": 0.0,
        "spoken_rule_compliance_rate": 0.0,
        "policy_ok": 0.0,
    }
    return {
        "trip_id": episode["trip_id"],
        "difficulty_tier": episode.get("difficulty_tier", "unknown"),
        **zero_metrics,
        "tool_calls": 0,
        "tokens": 0,
        "estimated_cost_usd": 0.0,
        "episode_failed": True,
        "execution_error": error,
    }


def run_eval_set(
    *,
    runner: LLMRunner,
    toolbox: TravelToolbox,
    systems: Dict[str, Dict[str, Any]],
    episodes: List[Dict[str, Any]],
    gold_map: Dict[str, Dict[str, Any]] | None = None,
    max_concurrency: int = 1,
    sanitize_solver_episodes: bool = False,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"systems": {}}
    max_workers = max(1, int(max_concurrency or 1))

    def run_one_episode(
        *,
        system_name: str,
        system_config: Dict[str, Any],
        episode: Dict[str, Any],
        episode_index: int,
        episode_count: int,
        gold: Dict[str, Any],
    ) -> Tuple[int, Dict[str, Any], Dict[str, Any], Dict[str, Any] | None, Dict[str, Any] | None]:
        episode_started_at = datetime.now(timezone.utc)
        worker_runner = LLMRunner(runner.config, trace_logger=runner.trace_logger)
        worker_runner.trace(
            "episode_start",
            system=system_name,
            trip_id=episode["trip_id"],
            episode_index=episode_index,
            episode_count=episode_count,
        )
        try:
            runner_key = system_config.get("runner_name", system_name)
            solver_episode = sanitize_episode_for_solver(episode, hidden=sanitize_solver_episodes)
            if "solver_module" in system_config:
                run_result = _run_dynamic_solver(worker_runner, toolbox, solver_episode, system_config)
            else:
                run_result = SYSTEM_RUNNERS[runner_key](worker_runner, toolbox, solver_episode, system_config)
            payload = attach_debug(run_result["submission"], run_result)
            observed_usage = worker_runner.usage_summary()
            # For dynamic submissions, ignore solver self-reported usage in
            # official runner output. Built-in agents also have identical
            # observed usage via the same ledger.
            payload["usage"] = observed_usage
            trace = run_result.get("official_trace")
            if trace is None:
                trace = run_result.get("retrieval", {})
                trace.setdefault("tool_trace", run_result.get("tool_trace", []))
                trace.setdefault("tool_call_count", len(run_result.get("tool_trace", [])))
            row = evaluate_episode(toolbox.env, episode, payload, gold=gold, trace=trace)
            result = {
                "trip_id": episode["trip_id"],
                "observed_episode": {key: value for key, value in solver_episode.items() if key != "gold"},
                "submission": payload,
                "row": row,
                "response_ids": run_result.get("response_ids", []),
                "retrieval": run_result.get("retrieval", {}),
                "api_status": run_result.get("api_status", {"success": True}),
            }
            duration_s = (datetime.now(timezone.utc) - episode_started_at).total_seconds()
            worker_runner.trace(
                "episode_success",
                system=system_name,
                trip_id=episode["trip_id"],
                episode_index=episode_index,
                duration_s=round(duration_s, 3),
                decision_quality=row.get("decision_quality"),
                cost_usd=row.get("estimated_cost_usd"),
                tool_calls=row.get("tool_calls"),
            )
            return episode_index - 1, result, row, None, None
        except Exception as exc:
            error_text = str(exc)
            failed_row = build_failed_row(episode, error_text)
            result = {
                "trip_id": episode["trip_id"],
                "observed_episode": {key: value for key, value in sanitize_episode_for_solver(episode, hidden=sanitize_solver_episodes).items() if key != "gold"},
                "submission": None,
                "row": failed_row,
                "response_ids": [],
                "retrieval": {},
                "api_status": {"success": False, "error": error_text},
            }
            duration_s = (datetime.now(timezone.utc) - episode_started_at).total_seconds()
            worker_runner.trace(
                "episode_error",
                system=system_name,
                trip_id=episode["trip_id"],
                episode_index=episode_index,
                duration_s=round(duration_s, 3),
                error=error_text,
                status="error",
            )
            return episode_index - 1, result, failed_row, {"trip_id": episode["trip_id"], "error": error_text}, None

    for system_name, system_config in systems.items():
        runner.trace(
            "system_start",
            system=system_name,
            episode_count=len(episodes),
            model=system_config.get("model") or system_config.get("planner_model"),
            max_concurrency=max_workers,
        )

        if max_workers > 1 and system_config.get("retrieval_strategy") == "embedding" and system_config.get("embedding_model"):
            warm = toolbox.memory_corpus.ensure_doc_embeddings(runner, system_config["embedding_model"])
            runner.trace(
                "embedding_prewarm",
                system=system_name,
                model=system_config["embedding_model"],
                cache_hit=warm.get("cache_hit", False),
                usage=warm.get("usage", {}),
            )

        rows: List[Dict[str, Any]] = [None] * len(episodes)  # type: ignore[list-item]
        results: List[Dict[str, Any]] = [None] * len(episodes)  # type: ignore[list-item]
        api_success = True
        errors: List[Dict[str, str]] = []

        if max_workers == 1:
            for episode_index, episode in enumerate(episodes, start=1):
                gold = resolve_gold(episode, gold_map)
                slot, result, row, error_entry, _ = run_one_episode(
                    system_name=system_name,
                    system_config=system_config,
                    episode=episode,
                    episode_index=episode_index,
                    episode_count=len(episodes),
                    gold=gold,
                )
                rows[slot] = row
                results[slot] = result
                if error_entry is not None:
                    api_success = False
                    errors.append(error_entry)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {}
                for episode_index, episode in enumerate(episodes, start=1):
                    gold = resolve_gold(episode, gold_map)
                    fut = executor.submit(
                        run_one_episode,
                        system_name=system_name,
                        system_config=system_config,
                        episode=episode,
                        episode_index=episode_index,
                        episode_count=len(episodes),
                        gold=gold,
                    )
                    future_map[fut] = episode["trip_id"]
                for future in as_completed(future_map):
                    slot, result, row, error_entry, _ = future.result()
                    rows[slot] = row
                    results[slot] = result
                    if error_entry is not None:
                        api_success = False
                        errors.append(error_entry)

        rows = [row for row in rows if row is not None]
        results = [result for result in results if result is not None]
        out["systems"][system_name] = {
            "config": system_config,
            "results": results,
            "rows": rows,
            "summary": summarize_rows(rows),
            "api_calls_all_succeeded": api_success and not errors,
            "errors": errors,
        }
        runner.trace(
            "system_finish",
            system=system_name,
            episode_count=len(episodes),
            success=api_success and not errors,
            summary=out["systems"][system_name]["summary"],
            error_count=len(errors),
        )
    return out

def build_ablation_systems(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    systems: Dict[str, Dict[str, Any]] = {}
    ablation_config = config.get("ablations") or {}
    for ablation_name, ablation in (ablation_config.get("systems") or {}).items():
        base_system = ablation["base_system"]
        if base_system not in config["default_systems"]:
            raise ValueError(f"Ablation {ablation_name!r} references unknown base_system {base_system!r}.")
        merged = merge_config(config["default_systems"][base_system], ablation["overrides"])
        merged.setdefault("runner_name", base_system)
        merged.setdefault("base_system", base_system)
        systems[ablation_name] = merged
    return systems


def find_episodes_by_trip_ids(episodes: List[Dict[str, Any]], trip_ids: List[str]) -> List[Dict[str, Any]]:
    wanted = set(trip_ids)
    selected = [episode for episode in episodes if episode["trip_id"] in wanted]
    missing = sorted(wanted - {episode["trip_id"] for episode in selected})
    if missing:
        raise ValueError(f"Unknown trip_ids: {missing}")
    return selected


def _extract_failed_trip_ids(results_payload: Dict[str, Any]) -> List[str]:
    failed: List[str] = []
    for system_payload in results_payload.get("systems", {}).values():
        for err in system_payload.get("errors", []):
            trip_id = err.get("trip_id")
            if trip_id:
                failed.append(trip_id)
        for result in system_payload.get("results", []):
            row = result.get("row") or {}
            if row.get("episode_failed"):
                trip_id = result.get("trip_id") or row.get("trip_id")
                if trip_id:
                    failed.append(trip_id)
    return sorted(dict.fromkeys(failed))


def resolve_rerun_failed_trip_ids(root: Path, rerun_source: str | None) -> Tuple[List[str], List[str]]:
    if not rerun_source:
        return [], []
    source = Path(rerun_source)
    if not source.is_absolute():
        source = root / source
    public_ids: List[str] = []
    hidden_ids: List[str] = []
    if source.is_dir():
        public_path = source / "llm_results_public_v2.json"
        hidden_path = source / "llm_results_hidden_sample_v2.json"
        if public_path.exists():
            public_ids = _extract_failed_trip_ids(load_json(public_path))
        if hidden_path.exists():
            hidden_ids = _extract_failed_trip_ids(load_json(hidden_path))
        return public_ids, hidden_ids
    payload = load_json(source)
    ids = _extract_failed_trip_ids(payload)
    name = source.name.lower()
    if "hidden" in name:
        return [], ids
    if "public" in name:
        return ids, []
    return ids, ids


def best_system_name(systems_payload: Dict[str, Any]) -> str:
    if not systems_payload:
        return ""
    scored = [(name, payload["summary"].get("raw_score_for_ranking", float("-inf"))) for name, payload in systems_payload.items()]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[0][0]


def get_summary(systems_payload: Dict[str, Any], system_name: str) -> Dict[str, Any]:
    return systems_payload.get(system_name, {}).get("summary", {})


def format_delta(left: Dict[str, Any], right: Dict[str, Any], key: str) -> str:
    return f"{left.get(key, 0.0):.4f} -> {right.get(key, 0.0):.4f}"


def render_summary_md(public_payload: Dict[str, Any], hidden_payload: Dict[str, Any], ablation_payload: Dict[str, Any]) -> str:
    public_systems = public_payload["systems"]
    best_public = best_system_name(public_systems)
    baseline = get_summary(public_systems, "llm_single_baseline")
    memory = get_summary(public_systems, "llm_memory_single")
    mas = get_summary(public_systems, "llm_anchor_mas")
    all_success = all(
        payload["api_calls_all_succeeded"]
        for group in [public_systems, hidden_payload["systems"], ablation_payload["systems"]]
        for payload in group.values()
    )

    lines = [
        "# LLM Eval Summary V4.7",
        "",
        "## Setup",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Public episodes: {public_payload['episode_count']}",
        f"- Hidden sample episodes: {hidden_payload['episode_count']}",
        f"- Hidden sample seed: {hidden_payload['sample_seed']}",
        f"- Hidden sample trip_ids: {', '.join(hidden_payload['sample_trip_ids'])}",
        f"- Hidden solver visibility: hidden evaluation passes a sanitized episode to the solver (user turns + observable trip context) while the evaluator uses separate full metadata/gold; public episodes remain transparent for debugging.",
        f"- Student budget caps present: {'Yes' if public_payload.get('budget_policy_present') else 'No'}",
        "",
        "## Main Results",
        "| system | official_score_100 | decision_quality | distributed_context | stale_doc_retirement | distractor_avoidance | spoken_rule | cost_usd | tool_calls |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system_name, payload in public_systems.items():
        summary = payload["summary"]
        lines.append(
            f"| {system_name} | {summary.get('raw_score_for_ranking', 0.0):.4f} | {summary.get('mean_decision_quality', 0.0):.4f} | "
            f"{summary.get('mean_distributed_context_rate', 0.0):.4f} | {summary.get('mean_stale_doc_retirement_rate', 0.0):.4f} | "
            f"{summary.get('mean_distractor_avoidance_rate', 0.0):.4f} | {summary.get('mean_spoken_rule_compliance_rate', 0.0):.4f} | "
            f"{summary.get('total_cost_usd', 0.0):.6f} | {summary.get('mean_tool_calls', 0.0):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Ablations",
            f"- Ablation sample trip_ids: {', '.join(ablation_payload['sample_trip_ids'])}",
            "| ablation | official_score_100 | decision_quality | stale_doc_retirement | spoken_rule | cost_usd |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for system_name, payload in ablation_payload["systems"].items():
        summary = payload["summary"]
        lines.append(
            f"| {system_name} | {summary.get('raw_score_for_ranking', 0.0):.4f} | {summary.get('mean_decision_quality', 0.0):.4f} | "
            f"{summary.get('mean_stale_doc_retirement_rate', 0.0):.4f} | {summary.get('mean_spoken_rule_compliance_rate', 0.0):.4f} | {summary.get('total_cost_usd', 0.0):.6f} |"
        )

    mas_analysis = (
        f"On public episodes, compare the official /100 scores directly: `llm_memory_single` {memory.get('raw_score_for_ranking', 0.0):.4f} vs `llm_anchor_mas` {mas.get('raw_score_for_ranking', 0.0):.4f}. "
        f"These summaries now include failed episodes as zero-score rows, so the comparison reflects both quality and execution reliability."
    )
    if memory and mas:
        if mas.get('raw_score_for_ranking', 0.0) > memory.get('raw_score_for_ranking', 0.0):
            mas_analysis = (
                f"On public episodes, yes: `llm_anchor_mas` edged out `llm_memory_single` {memory.get('raw_score_for_ranking', 0.0):.4f} -> {mas.get('raw_score_for_ranking', 0.0):.4f}. "
                f"The gain came from stronger adaptation-style metrics without a collapse in decision quality."
            )
        elif mas.get('raw_score_for_ranking', 0.0) < memory.get('raw_score_for_ranking', 0.0):
            mas_analysis = (
                f"On public episodes, no: `llm_memory_single` remained ahead at {memory.get('raw_score_for_ranking', 0.0):.4f} official /100 score, while `llm_anchor_mas` reached {mas.get('raw_score_for_ranking', 0.0):.4f}. "
                f"MAS may still show advantages in adaptation metrics, but they did not outweigh its reliability/cost tradeoff in this run."
            )
        else:
            mas_analysis = (
                f"On public episodes, the two systems tied at {mas.get('raw_score_for_ranking', 0.0):.4f} official /100 score under the corrected failure-aware summary."
            )
    memory_in_run = "llm_memory_single" in public_systems
    mas_in_run = "llm_anchor_mas" in public_systems
    if not memory_in_run and not mas_in_run:
        mas_analysis = "Not evaluated in this run because neither `llm_memory_single` nor `llm_anchor_mas` was included in `--systems`."
    elif not mas_in_run:
        mas_analysis = "Not evaluated in this run because `llm_anchor_mas` was not included in `--systems`."
    elif not memory_in_run:
        mas_analysis = (
            f"Partially evaluated. `llm_anchor_mas` ran and reached {mas.get('raw_score_for_ranking', 0.0):.4f} official /100 score, but `llm_memory_single` was not included, so the main MAS-vs-memory comparison is unavailable for this run."
        )
    elif not memory and not mas:
        mas_analysis = "The relevant systems were included, but neither completed a successful public episode in this run, so the MAS-vs-memory comparison is unavailable."
    elif not mas:
        mas_analysis = "`llm_anchor_mas` was included, but it produced no successful public episodes in this run, so the MAS-vs-memory comparison is unavailable."
    elif not memory:
        mas_analysis = "`llm_memory_single` was included, but it produced no successful public episodes in this run, so the MAS-vs-memory comparison is unavailable."

    separation_analysis = "Not enough systems were included to compare retrieval / retirement / spoken-rule separation in this run."
    if baseline and memory:
        separation_analysis = (
            f"Yes. Public `llm_single_baseline` vs `llm_memory_single`: memory retirement {format_delta(baseline, memory, 'mean_memory_retirement_rate')}, "
            f"stale-doc retirement {format_delta(baseline, memory, 'mean_stale_doc_retirement_rate')}, "
            f"active-context hygiene {format_delta(baseline, memory, 'mean_active_context_hygiene_rate')}, "
            f"spoken-rule compliance {format_delta(baseline, memory, 'mean_spoken_rule_compliance_rate')}."
        )

    cost_analysis = "Cost comparison across the canonical baseline/memory/MAS trio is unavailable for this run."
    if baseline and memory and mas:
        cost_analysis = (
            f"Cost increased from `${baseline.get('total_cost_usd', 0.0):.6f}` for `llm_single_baseline` to `${memory.get('total_cost_usd', 0.0):.6f}` for `llm_memory_single`, "
            f"then to `${mas.get('total_cost_usd', 0.0):.6f}` for `llm_anchor_mas`. Whether the extra spend was worth it should be judged against the corrected failure-aware official /100 scores above."
        )
    elif len(public_systems) >= 2:
        ordered = [
            f"`{name}` `${payload['summary'].get('total_cost_usd', 0.0):.6f}`"
            for name, payload in public_systems.items()
        ]
        cost_analysis = "Run-specific cost snapshot: " + ", ".join(ordered) + "."

    lines.extend(
        [
            "",
            "## Required Analysis",
            "1. Hidden set leakage fix",
            f"Hidden evaluation no longer reads `gold` from the executable hidden episode file. `episodes_hidden_input.json` contains only observable fields, while `episodes_hidden_gold.json` stores evaluator-only labels keyed by trip id. The runtime joins them only inside the evaluator path.",
            "2. Tool-use structure",
            "The harness now runs actual Responses API function-calling loops. Agents start with only episode context, then selectively call filtered tools such as `search_flights`, `search_hotels`, `search_restaurants`, `search_activities`, `search_memory`, and `get_rejected_options`. Inventory is no longer prepacked into the prompt.",
            "3. Retrieval corpus difficulty",
            "The benchmark now includes canonical docs, stale notes, distractor traveler notes, alternative venue playbooks, and explicit heuristics in `memory_corpus.json`, plus `rejected_options_memory.json`. This makes retrieval choice, stale retirement, and distractor avoidance matter instead of always pulling the same 3 docs.",
            "4. Heuristic pre-processing reduction",
            "Strong shortlist construction, bundle ranking, forced ID repair, and Python-side spoken-rule parsing were removed from the execution path. Python now provides environment search, retrieval indexing, tool dispatch, and result validation only.",
            "5. Did MAS beat semantic single-agent?",
            mas_analysis,
            "6. Did retrieval / retirement / spoken-rule metrics separate?",
            separation_analysis,
            "7. Cost vs performance",
            cost_analysis,
            "8. Remaining limits",
            "The benchmark is now stricter and more LLM-native, but the inventory is still relatively small, hidden gold still exists locally for TA-side grading, and the harness still relies on model-reported memory summaries rather than a fully independently derived semantic memory state.",
            "",
            "## Hidden Sample Results",
            "| system | official_score_100 | decision_quality | stale_doc_retirement | distractor_avoidance | cost_usd |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for system_name, payload in hidden_payload["systems"].items():
        summary = payload["summary"]
        lines.append(
            f"| {system_name} | {summary.get('raw_score_for_ranking', 0.0):.4f} | {summary.get('mean_decision_quality', 0.0):.4f} | "
            f"{summary.get('mean_stale_doc_retirement_rate', 0.0):.4f} | {summary.get('mean_distractor_avoidance_rate', 0.0):.4f} | {summary.get('total_cost_usd', 0.0):.6f} |"
        )

    best_public_line = "No public systems were executed."
    if best_public:
        best_public_line = f"`{best_public}` with official /100 score {public_systems[best_public]['summary'].get('raw_score_for_ranking', 0.0):.4f}."

    lines.extend(
        [
            "",
            "## Best Public System",
            best_public_line,
            "",
            "## API Status",
            f"All API calls succeeded: {'Yes' if all_success else 'No'}",
        ]
    )
    error_lines: List[str] = []
    for section_name, payload in [("Public", public_payload), ("Hidden", hidden_payload), ("Ablation", ablation_payload)]:
        for system_name, system_payload in payload.get("systems", {}).items():
            for err in system_payload.get("errors", []):
                error_lines.append(f"- {section_name} / {system_name} / {err.get('trip_id')}: {err.get('error')}")
    if error_lines:
        lines.extend(["", "## Errors", *error_lines])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="llm_eval_config.json")
    parser.add_argument("--systems", type=parse_csv_arg, default=[], help="Comma-separated default systems to run.")
    parser.add_argument("--skip-hidden", action="store_true")
    parser.add_argument("--limit-public", type=int, default=0)
    parser.add_argument("--public-trip-ids", type=parse_csv_arg, default=[], help="Comma-separated public trip_ids to run.")
    parser.add_argument("--hidden-sample-size", type=int, default=0)
    parser.add_argument("--hidden-trip-ids", type=parse_csv_arg, default=[], help="Comma-separated hidden trip_ids to run instead of sampling.")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--ablation-systems", type=parse_csv_arg, default=[], help="Comma-separated ablation system names to run.")
    parser.add_argument("--trace-path", default="llm_run_trace.jsonl")
    parser.add_argument("--output-dir", default=".", help="Directory to write run artifacts into.")
    parser.add_argument("--max-concurrency", type=int, default=1, help="Maximum number of episodes to execute concurrently per system.")
    parser.add_argument("--rerun-failed-from", default="", help="Path to a previous results directory or results JSON file. Failed trip_ids will be rerun instead of the default public/hidden selection.")
    parser.add_argument("--set", dest="config_overrides", action="append", default=[], help="Override a student-tunable system budget knob. Format: SYSTEM.FIELD=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_json(root / args.config)

    default_systems = dict(config["default_systems"])
    memory_base = dict(default_systems.get("llm_memory_single", {}))
    mas_base = dict(default_systems.get("llm_anchor_mas", {}))
    for system_name, defaults in DYNAMIC_SOLVER_DEFAULTS.items():
        if system_name not in default_systems:
            seed = memory_base
            if system_name == "student_solver2" and "llm_anchor_mas" in default_systems:
                seed = mas_base
            merged = dict(seed)
            merged.update(defaults)
            default_systems[system_name] = merged
    if args.systems:
        default_systems = {name: default_systems[name] for name in args.systems}
    applied_overrides = apply_budget_overrides(default_systems, config, args.config_overrides)
    validate_system_budget_caps(default_systems, config)

    ablation_systems = build_ablation_systems(config)
    if args.ablation_systems:
        missing_ablations = [name for name in args.ablation_systems if name not in ablation_systems]
        if missing_ablations:
            raise ValueError(
                f"Unknown ablation system(s): {missing_ablations}. "
                "This config may be student-facing and may not define an 'ablations' block."
            )
        ablation_systems = {name: ablation_systems[name] for name in args.ablation_systems}

    trace_path = Path(args.trace_path)
    if not trace_path.is_absolute():
        trace_path = output_dir / trace_path
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_logger = TraceLogger(trace_path, console=True)
    trace_logger.log(
        "run_start",
        config=args.config,
        systems=list(default_systems),
        public_trip_ids=args.public_trip_ids,
        hidden_trip_ids=args.hidden_trip_ids,
        ablation_systems=list(ablation_systems),
        skip_hidden=args.skip_hidden,
        limit_public=args.limit_public,
        hidden_sample_size=args.hidden_sample_size or config["hidden_eval_sample_size"],
        skip_ablations=args.skip_ablations,
        max_concurrency=args.max_concurrency,
        budget_overrides=applied_overrides,
        rerun_failed_from=args.rerun_failed_from,
    )
    try:
        runner = LLMRunner(config, dotenv_path=root / ".env", trace_logger=trace_logger)
        toolbox = TravelToolbox(
            root / "dynamic_travel_replanning",
            max_results=max(system.get("max_tool_results", 4) for system in config["default_systems"].values()),
        )

        rerun_public_ids, rerun_hidden_ids = resolve_rerun_failed_trip_ids(root, args.rerun_failed_from)

        public_eps_all = load_json(root / config["public_eval_file"])
        public_eps = public_eps_all
        if rerun_public_ids:
            public_eps = find_episodes_by_trip_ids(public_eps_all, rerun_public_ids)
        elif args.public_trip_ids:
            public_eps = find_episodes_by_trip_ids(public_eps_all, args.public_trip_ids)
        elif args.limit_public:
            public_eps = public_eps_all[: args.limit_public]
        public_payload = run_eval_set(runner=runner, toolbox=toolbox, systems=default_systems, episodes=public_eps, max_concurrency=args.max_concurrency)
        public_payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        public_payload["budget_policy_present"] = bool(config.get("student_tunable_budgets"))
        public_payload["budget_overrides"] = applied_overrides
        public_payload["episode_count"] = len(public_eps)
        trace_logger.log("public_eval_complete", episode_count=len(public_eps), trip_ids=[episode["trip_id"] for episode in public_eps])

        hidden_input_file = config.get("hidden_eval_input_file")
        hidden_gold_file = config.get("hidden_eval_gold_file")
        hidden_input_path = root / hidden_input_file if hidden_input_file else None
        hidden_gold_path = root / hidden_gold_file if hidden_gold_file else None
        hidden_available = bool(hidden_input_path and hidden_gold_path and hidden_input_path.exists() and hidden_gold_path.exists())
        hidden_payload = {"systems": {}, "episode_count": 0, "sample_seed": config.get("hidden_eval_seed", 0), "sample_trip_ids": []}
        if hidden_available:
            hidden_eps = load_json(hidden_input_path)
            hidden_gold = load_json(hidden_gold_path)
            if rerun_hidden_ids:
                hidden_sample = find_episodes_by_trip_ids(hidden_eps, rerun_hidden_ids)
            elif args.hidden_trip_ids:
                hidden_sample = find_episodes_by_trip_ids(hidden_eps, args.hidden_trip_ids)
            else:
                hidden_sample_size = args.hidden_sample_size or config["hidden_eval_sample_size"]
                hidden_sample = sample_hidden_episodes(hidden_eps, hidden_sample_size, config["hidden_eval_seed"])
            if not args.skip_hidden:
                hidden_payload = run_eval_set(
                    runner=runner,
                    toolbox=toolbox,
                    systems=default_systems,
                    episodes=hidden_sample,
                    gold_map=hidden_gold,
                    max_concurrency=args.max_concurrency,
                    sanitize_solver_episodes=True,
                )
                hidden_payload["generated_at"] = datetime.now(timezone.utc).isoformat()
                hidden_payload["budget_policy_present"] = bool(config.get("student_tunable_budgets"))
                hidden_payload["budget_overrides"] = applied_overrides
                hidden_payload["episode_count"] = len(hidden_sample)
                hidden_payload["sample_seed"] = config["hidden_eval_seed"]
                hidden_payload["sample_trip_ids"] = [episode["trip_id"] for episode in hidden_sample]
                trace_logger.log("hidden_eval_complete", episode_count=len(hidden_sample), trip_ids=hidden_payload["sample_trip_ids"])
        else:
            args.skip_hidden = True
            trace_logger.log("hidden_eval_skipped", reason="hidden assets unavailable")

        ablation_payload = {"systems": {}, "sample_trip_ids": []}
        if not args.skip_ablations and ablation_systems:
            ablation_trip_ids = args.public_trip_ids or (config.get("ablations") or {}).get("public_sample_trip_ids", [])
            if not ablation_trip_ids:
                trace_logger.log("ablation_eval_skipped", reason="no ablation public_sample_trip_ids configured")
            else:
                ablation_eps = find_episodes_by_trip_ids(public_eps_all, ablation_trip_ids)
                ablation_payload = run_eval_set(
                    runner=runner,
                    toolbox=toolbox,
                    systems=ablation_systems,
                    episodes=ablation_eps,
                    max_concurrency=args.max_concurrency,
                )
                ablation_payload["sample_trip_ids"] = ablation_trip_ids
                ablation_payload["budget_overrides"] = applied_overrides
                ablation_payload["generated_at"] = datetime.now(timezone.utc).isoformat()
                trace_logger.log("ablation_eval_complete", episode_count=len(ablation_eps), trip_ids=ablation_trip_ids, systems=list(ablation_systems))
        elif not args.skip_ablations:
            trace_logger.log("ablation_eval_skipped", reason="no ablations configured")

        public_payload["ablations"] = ablation_payload
        public_results_path = output_dir / "llm_results_public_v2.json"
        hidden_results_path = output_dir / "llm_results_hidden_sample_v2.json"
        summary_path = output_dir / "llm_eval_summary_v2.md"
        write_json(public_results_path, public_payload)
        write_json(hidden_results_path, hidden_payload)
        summary_md = render_summary_md(public_payload, hidden_payload, ablation_payload)
        summary_path.write_text(summary_md)
        trace_logger.log(
            "artifacts_written",
            output_dir=str(output_dir),
            public_results=str(public_results_path),
            hidden_results=str(hidden_results_path),
            summary=str(summary_path),
        )
        print(summary_md)
    finally:
        trace_logger.log("run_finish", trace_path=str(trace_path), output_dir=str(output_dir))
        trace_logger.close()


if __name__ == "__main__":
    main()
