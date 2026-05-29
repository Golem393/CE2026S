from __future__ import annotations

from typing import Any, Dict

from llm_agents import run_single_baseline
from runtime_api import StudentRuntime


def solve_episode(runtime: StudentRuntime) -> Dict[str, Any]:
    """Working baseline example shipped to students.

    Important: create tool sessions through runtime.new_session, not
    runtime.toolbox.new_session, so the official evaluator can observe the
    retrieval/tool trace used for memory and context-grounding credit.
    """
    return run_single_baseline(
        runtime.runner,
        runtime.toolbox,
        runtime.episode,
        runtime.system_config,
        session_factory=runtime.new_session,
    )
