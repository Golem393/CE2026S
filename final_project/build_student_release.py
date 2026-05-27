from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "_build" / "student_release"

INCLUDE = [
    ".env.sample",
    ".gitignore",
    "allowed_models.json",
    "README_STARTER.md",
    "requirements.txt",
    "run_student.py",
    "run_llm_baselines.py",
    "llm_runner.py",
    "runtime_api.py",
    "budget_knobs.py",
    "llm_agents.py",
    "llm_tools.py",
    "retrieval.py",
    "trace_logger.py",
    "schemas.py",
    "dynamic_travel_replanning/evaluator.py",
    "dynamic_travel_replanning/episode_generator.py",
    "dynamic_travel_replanning/evaluation_tracks.json",
    "dynamic_travel_replanning/task_catalog.json",
    "dynamic_travel_replanning/task_bank_validation.json",
    "dynamic_travel_replanning/tier_manifest.json",
    "dynamic_travel_replanning/rtl_semantic_env.py",
    "dynamic_travel_replanning/episodes_public_example.json",
    "dynamic_travel_replanning/inventory_flights.json",
    "dynamic_travel_replanning/inventory_hotels.json",
    "dynamic_travel_replanning/inventory_restaurants.json",
    "dynamic_travel_replanning/inventory_activities.json",
    "dynamic_travel_replanning/profile_briefs.json",
    "dynamic_travel_replanning/venue_briefs.json",
    "dynamic_travel_replanning/city_ops_notes.json",
    "dynamic_travel_replanning/memory_corpus.json",
    "dynamic_travel_replanning/rejected_options_memory.json",
    "dynamic_travel_replanning/partner_promotions.json",
    "dynamic_travel_replanning/event_calendar.json",
    "dynamic_travel_replanning/loyalty_programs.json",
    "dynamic_travel_replanning/stakeholder_briefs.json",
    "dynamic_travel_replanning/booking_constraints.json",
    "dynamic_travel_replanning/option_dependencies.json",
    "dynamic_travel_replanning/policy_rules.json",
    "dynamic_travel_replanning/scenario_grammar.json",
    "dynamic_travel_replanning/transit_matrix.json",
    "dynamic_travel_replanning/weather_buckets.json",
    "STUDENT_EVALUATION.md",
    "STUDENT_TOOLING.md",
    "student_custom_tools_template.py",
    "student_solver.py",
    "student_solver_example.py",
    "dynamic_wellness_replanning",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean student-facing release.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory for the student release.")
    parser.add_argument("--zip", action="store_true", help="Also write a zip archive next to the output directory.")
    return parser.parse_args()


def _load_json(rel: str):
    return json.loads((ROOT / rel).read_text())


def build_student_config() -> dict:
    config = json.loads((ROOT / "llm_eval_config_template.json").read_text())
    memory_base = dict(config["default_systems"].get("llm_memory_single", {}))
    for system_cfg in config.get("default_systems", {}).values():
        system_cfg["primitive_tools_only"] = True
    config["default_systems"] = {
        "student_solver": {
            **memory_base,
            "solver_module": "student_solver",
            "system_name": "student_solver",
            "primitive_tools_only": True,
        },
        "student_solver_example": {
            **memory_base,
            "solver_module": "student_solver_example",
            "system_name": "student_solver_example",
            "primitive_tools_only": True,
        },
    }
    config["public_eval_file"] = "dynamic_travel_replanning/episodes_public_example.json"
    config.pop("hidden_eval_file", None)
    config.pop("hidden_eval_input_file", None)
    config.pop("hidden_eval_gold_file", None)
    config.pop("hidden_eval_sample_size", None)
    budgets = config.get("student_tunable_budgets", {})
    config["student_tunable_budgets"] = {
        key: budgets[key]
        for key in ["student_solver", "student_solver_example"]
        if key in budgets
    }
    return config


def build_student_evaluation_tracks() -> dict:
    tracks = _load_json("dynamic_travel_replanning/evaluation_tracks.json")
    if "tracks" in tracks:
        return {
            "tracks": {k: v for k, v in tracks.get("tracks", {}).items() if k.startswith("public_")},
            "hidden_summary": tracks.get("hidden_summary", {}),
        }
    allowed = ["starter_public", "core_public", "full_public", "balanced_public_by_metadata", "note"]
    return {k: tracks[k] for k in allowed if k in tracks}


def build_student_task_catalog() -> list[dict]:
    catalog = _load_json("dynamic_travel_replanning/task_catalog.json")
    return [item for item in catalog if item.get("split") == "public"]


def build_student_task_bank_validation() -> dict:
    payload = _load_json("dynamic_travel_replanning/task_bank_validation.json")
    return {
        "public_count": payload.get("public_count", 0),
        "public_family_counts": payload.get("public_family_counts", {}),
        "public_exact_duplicate_prompt_groups": payload.get("public_exact_duplicate_prompt_groups", []),
        "public_tier_counts": payload.get("public_tier_counts", {}),
        "note": "Student release contains public-only validation metadata.",
    }


def build_student_tier_manifest() -> dict:
    manifest = _load_json("dynamic_travel_replanning/tier_manifest.json")
    return {
        "public_easy": manifest.get("public_easy", []),
        "public_medium": manifest.get("public_medium", []),
        "public_hard": manifest.get("public_hard", []),
        "public_all": manifest.get("public_all", []),
        "public_count": manifest.get("public_count", 0),
        "public_tier_counts": manifest.get("public_tier_counts", {}),
        "hidden_count": manifest.get("hidden_count", 30),
        "hidden_tier_counts": manifest.get("hidden_tier_counts", {}),
        "hidden_note": manifest.get("hidden_note", "Hidden specs and gold labels are TA-only."),
        "retiered_trip_ids": [trip_id for trip_id in manifest.get("retiered_trip_ids", []) if trip_id.startswith("rtl7_public_")],
    }


def build_student_runner_source() -> str:
    source = (ROOT / "run_llm_baselines.py").read_text()
    source = source.replace('from llm_agents import run_memory_single, run_single_baseline\n', '')
    source = source.replace(
        'SYSTEM_RUNNERS = {\n    "llm_single_baseline": run_single_baseline,\n    "llm_memory_single": run_memory_single,\n}\n\n',
        'SYSTEM_RUNNERS = {}\n\n',
    )
    ta_hook = '''if _has_module("ta_only.student_solver2"):\n    DYNAMIC_SOLVER_DEFAULTS["student_solver2"] = {\n        "solver_module": "ta_only.student_solver2",\n        "system_name": "student_solver2",\n    }\n'''
    source = source.replace(ta_hook, "")
    source = source.replace(
        '    memory_base = dict(default_systems.get("llm_memory_single", {}))\n    mas_base = dict(default_systems.get("llm_anchor_mas", {}))\n    for system_name, defaults in DYNAMIC_SOLVER_DEFAULTS.items():\n        if system_name not in default_systems:\n            seed = memory_base\n            if system_name == "student_solver2" and "llm_anchor_mas" in default_systems:\n                seed = mas_base\n            merged = dict(seed)\n            merged.update(defaults)\n            default_systems[system_name] = merged\n',
        '    default_seed = dict(next(iter(default_systems.values()), {}))\n    for system_name, defaults in DYNAMIC_SOLVER_DEFAULTS.items():\n        if system_name not in default_systems:\n            merged = dict(default_seed)\n            merged.update(defaults)\n            default_systems[system_name] = merged\n',
    )
    generic_summary = r'''def render_summary_md(public_payload: Dict[str, Any], hidden_payload: Dict[str, Any], ablation_payload: Dict[str, Any]) -> str:
    public_systems = public_payload["systems"]
    lines = [
        "# Student Run Summary",
        "",
        "## Setup",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Public episodes: {public_payload['episode_count']}",
        f"- Hidden episodes executed: {hidden_payload['episode_count']}",
        f"- Student budget caps present: {'Yes' if public_payload.get('budget_policy_present') else 'No'}",
        "",
        "## Public Results",
        "| system | raw_score | decision_quality | update_handling | spoken_rule | cost_usd | tool_calls |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system_name, payload in public_systems.items():
        summary = payload["summary"]
        lines.append(
            f"| {system_name} | {summary.get('raw_score_for_ranking', 0.0):.4f} | {summary.get('mean_decision_quality', 0.0):.4f} | "
            f"{summary.get('mean_update_handling_rate', 0.0):.4f} | {summary.get('mean_spoken_rule_compliance_rate', 0.0):.4f} | "
            f"{summary.get('total_cost_usd', 0.0):.6f} | {summary.get('mean_tool_calls', 0.0):.2f} |"
        )
    if public_systems:
        best = best_system_name(public_systems)
        best_score = public_systems[best]["summary"].get("raw_score_for_ranking", 0.0)
        lines.extend([
            "",
            "## Notes",
            f"- Best public system in this run: `{best}` ({best_score:.4f} raw score).",
            "- This student-facing summary only compares the systems included in your run.",
        ])
    if ablation_payload.get("systems"):
        lines.extend(["", "## Ablations", "| ablation | raw_score | decision_quality | cost_usd |", "| --- | ---: | ---: | ---: |"])
        for system_name, payload in ablation_payload["systems"].items():
            summary = payload["summary"]
            lines.append(
                f"| {system_name} | {summary.get('raw_score_for_ranking', 0.0):.4f} | {summary.get('mean_decision_quality', 0.0):.4f} | {summary.get('total_cost_usd', 0.0):.6f} |"
            )
    error_lines: List[str] = []
    for section_name, payload in [("Public", public_payload), ("Hidden", hidden_payload), ("Ablation", ablation_payload)]:
        for system_name, system_payload in payload.get("systems", {}).items():
            for err in system_payload.get("errors", []):
                error_lines.append(f"- {section_name} / {system_name} / {err.get('trip_id')}: {err.get('error')}")
    if error_lines:
        lines.extend(["", "## Errors", *error_lines])
    return "\\n".join(lines) + "\\n"
'''
    source = re.sub(
        r'def render_summary_md\(public_payload: Dict\[str, Any\], hidden_payload: Dict\[str, Any\], ablation_payload: Dict\[str, Any\]\) -> str:\n(?:.|\n)*?\n\ndef parse_args\(\) -> argparse.Namespace:',
        generic_summary + '\n\ndef parse_args() -> argparse.Namespace:',
        source,
        count=1,
    )
    return source


def main() -> None:
    args = parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True)
    for rel in INCLUDE:
        src = ROOT / rel
        if not src.exists():
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".cache"))
        else:
            shutil.copy2(src, dst)

    # Replace the student-facing runner with a slimmed copy that omits TA-only hooks.
    (out / "run_llm_baselines.py").write_text(build_student_runner_source())

    student_config = build_student_config()
    (out / "llm_eval_config_student.json").write_text(json.dumps(student_config, ensure_ascii=False, indent=2))

    sanitized_json = {
        "dynamic_travel_replanning/evaluation_tracks.json": build_student_evaluation_tracks(),
        "dynamic_travel_replanning/task_catalog.json": build_student_task_catalog(),
        "dynamic_travel_replanning/task_bank_validation.json": build_student_task_bank_validation(),
        "dynamic_travel_replanning/tier_manifest.json": build_student_tier_manifest(),
    }
    for rel, payload in sanitized_json.items():
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.zip:
        zip_base = out.with_suffix("") if out.suffix else out
        shutil.make_archive(str(zip_base), "zip", root_dir=out.parent, base_dir=out.name)

    print(f"wrote student release to {out}")


if __name__ == "__main__":
    main()
