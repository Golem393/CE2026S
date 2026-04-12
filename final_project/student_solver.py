from __future__ import annotations

from typing import Any, Dict

from runtime_api import StudentRuntime


def solve_episode(runtime: StudentRuntime) -> Dict[str, Any]:
    """Implement your own solver here.

    Requirements:
    - Route every model call through runtime.runner so usage/cost is measured by
      the official wrapper.
    - Return a dict with at least:
        {
          "submission": {...},
          "usage": {...}
        }
      and optionally `response_ids`, `tool_trace`, `retrieval`, `api_status`.

    See student_solver_example.py for a working baseline implementation.
    """
    raise NotImplementedError("Fill in solve_episode() in student_solver.py")
