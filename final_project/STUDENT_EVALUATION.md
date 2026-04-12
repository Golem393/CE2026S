
# Student-Facing Evaluation Guide

The full evaluation code is public in `dynamic_travel_replanning/evaluator.py`.

## Public-facing score buckets
Students should focus on four understandable buckets.

1. **Feasibility / constraints (40%)**
   - Does the plan satisfy hard constraints?
   - Is the plan policy-safe?
   - If a bundle is claimed, is it actually valid?

2. **Preference fit / plan quality (30%)**
   - Is the chosen bundle high quality overall?
   - Does it match quiet / polished / low-friction preferences?
   - Does it follow the spoken rules?

3. **Adaptation / memory discipline (20%)**
   - Did the system retrieve relevant memory?
   - Did it retire stale assumptions?
   - Did it avoid rediscovering rejected options?
   - Did it keep active context lean?

4. **Efficiency bonus (10%)**
   - Lower cost and fewer unnecessary calls help, but efficiency never overrides clearly bad planning.

## Why the detailed metrics still exist
The evaluator also keeps detailed diagnostics such as stale-doc retirement, rejected-option memory, and active-context hygiene. These are primarily for debugging and benchmark maintenance. Students do not need to optimize to each sub-metric separately.


## Quick start

For the simplest local workflow, use `python run_student.py --solver student_solver_example --limit-public 2`.
