from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "llm_eval_config_student.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple student-facing entrypoint for public evaluation.")
    p.add_argument("--solver", default="student_solver", choices=["student_solver", "student_solver_example"], help="Which student solver module to run.")
    p.add_argument("--limit-public", type=int, default=None, help="Optional number of public episodes to run for a quick smoke test.")
    p.add_argument("--output-dir", default="runs/student_run", help="Directory for results.")
    p.add_argument("--trace-path", default="trace.jsonl", help="Trace file path relative to output dir.")
    p.add_argument("--max-concurrency", type=int, default=1, help="How many public episodes to run concurrently.")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="Budget override, e.g. --set student_solver.max_tool_rounds=12")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "run_llm_baselines.py"),
        "--config",
        str(DEFAULT_CONFIG),
        "--systems",
        args.solver,
        "--skip-hidden",
        "--skip-ablations",
        "--max-concurrency",
        str(args.max_concurrency),
        "--output-dir",
        args.output_dir,
        "--trace-path",
        args.trace_path,
    ]
    if args.limit_public is not None:
        cmd.extend(["--limit-public", str(args.limit_public)])
    for override in args.overrides:
        cmd.extend(["--set", override])
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
