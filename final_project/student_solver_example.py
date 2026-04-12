from __future__ import annotations

from typing import Any, Dict

from llm_agents import run_single_baseline
from runtime_api import StudentRuntime


def solve_episode(runtime: StudentRuntime) -> Dict[str, Any]:
    """Working baseline example shipped to students."""
    return run_single_baseline(runtime.runner, runtime.toolbox, runtime.episode, runtime.system_config)
