
# Student Tooling Policy

## Primitive tools you can rely on
The public API intentionally starts from primitive tools such as:
- `search_flights`
- `search_hotels`
- `search_restaurants`
- `search_activities`
- `search_memory`
- `get_rejected_options`
- `get_profile_brief`
- `get_venue_brief`
- `get_city_ops_notes`
- `get_policy`

These tools are deterministic and transparent.

## What students are encouraged to add
You are encouraged to build your own wrappers and helper tools on top of the primitive API, for example:
- semantic rerankers for hotels/restaurants
- hybrid lexical+dense memory retrieval
- bundle-aware search helpers
- fallback search when strict filters return no results
- risk-aware scoring / verifier helpers

The project is intentionally designed so that improving tool ergonomics is part of the assignment.


## Budget knobs
The starter harness includes student-tunable budget knobs for tool rounds, output tokens, and tool result counts. You can change them from the config file or from the command line with `--set SYSTEM.FIELD=VALUE`.

Examples:
- `--set student_solver.max_tool_rounds=12`
- `--set student_solver.max_tool_results=5`
- `--set student_solver.max_output_tokens=1200`

The course release should keep hard caps enabled. Those caps are part of the benchmark design: you are meant to find a good budget, not remove the budget entirely.


## Student solver interface

Students implement `solve_episode(runtime)` inside `student_solver.py`. `student_solver_example.py` is the baseline example included in the student release. Official grading reruns submitted solver code with the staff-owned runtime wrapper and evaluator.


## Quick start

For the simplest local workflow, use `python run_student.py --solver student_solver_example --limit-public 2`.
