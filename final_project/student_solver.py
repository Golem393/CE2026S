from __future__ import annotations

from typing import Any, Dict

from runtime_api import StudentRuntime


def solve_episode(runtime: StudentRuntime) -> Dict[str, Any]:
    """Implement your own solver here.

    Requirements:
    - Route every model call through runtime.runner so usage/cost is measured by
      the official wrapper.
    - Route every tool session through runtime.new_session(...), not
      runtime.toolbox.new_session(...), so official retrieval/tool traces are
      captured for memory and context-grounding credit.
    - Return a dict with at least:
        {
          "submission": {...},
          "usage": {...}
        }
      and optionally `response_ids`, `tool_trace`, `retrieval`, `api_status`.

    See student_solver_example.py for a working baseline implementation.
    """
    raise NotImplementedError("Fill in solve_episode() in student_solver.py")
