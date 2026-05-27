
# Student-Facing Evaluation Guide

The full evaluation code is public in `dynamic_travel_replanning/evaluator.py`.

## Public-facing official score
The official project score is a single `/100` score.  Leaderboard ranking uses the same `/100` score shown to students; there is no separate `/45` ranking scale.

Students should focus on eight explainable components:

| Component | Points | What it means |
| --- | ---: | --- |
| Hard constraints | 45 | The selected plan satisfies active hard constraints such as budget, timing, safety, dietary, weather, and zone requirements. |
| Bundle coherence | 5 | Linked-perk / bundle choices are actually valid when the plan relies on them. |
| Semantic soft fit | 10 | The plan matches soft preferences such as sleep-friendly lodging, covered fallback quality, plant-based fit, convenience, and style. |
| Exact-ish option fit | 5 | The selected options are close to the evaluator's acceptable answer set. |
| Update handling | 15 | The final plan follows the latest user request rather than stale earlier assumptions. |
| Stale retirement | 5 | The solver correctly retires superseded or contradictory old conditions. |
| Rejected-option memory | 5 | The solver avoids options the user previously rejected. |
| Cost efficiency | 10 | Lower cost/tokens help, but cost should not override clearly bad planning. |

Formula:

```text
Official Score /100 =
45 * hard_constraint_rate
+ 5 * bundle_coherence_rate
+ 10 * semantic_fit_rate
+ 5 * exactish_rate
+ 15 * update_handling_rate
+ 5 * stale_doc_retirement_rate
+ 5 * rejected_option_memory_rate
+ 10 * efficiency_score
```


## Why the detailed metrics still exist
The evaluator also keeps detailed diagnostics such as stale-doc retirement, rejected-option memory, and active-context hygiene. These are primarily for debugging and benchmark maintenance. Students do not need to optimize to each sub-metric separately.


## Quick start

For the simplest local workflow, use `python run_student.py --solver student_solver_example --limit-public 2`.

## Final-answer rationale

Official hidden evaluation may check whether your final `notes` / `plan_rationale` is grounded in the current user turns and the evidence your solver actually retrieved. This is not a direct penalty for using or not using an LLM. A deterministic system can receive credit if it produces a concise, truthful rationale. However, claims such as "I retrieved this document" or "I retired this stale condition" should be supported by the runtime/tool trace.

A good rationale briefly states:

- the current hard constraints it satisfied,
- which stale or contradictory earlier assumptions were retired,
- why the selected options are better than plausible alternatives,
- which memory or document evidence was actually used.

## Tool trace note

For official evaluation, tool/retrieval credit is based on the harness-observed trace.
Create tool sessions through `runtime.new_session(...)` inside `solve_episode(runtime)`.
Do not create sessions directly with `runtime.toolbox.new_session(...)`, because those calls
may run locally but will not be visible to the official trace-based memory/context scoring.


## Hidden evaluation visibility note

Public episodes intentionally remain transparent for debugging and may expose metadata such as
`scenario_state`, `scenario_hooks`, or benchmark-family labels. You may use the public evaluator
and public metadata to understand the rubric.

Hidden evaluation is different: the solver-facing runtime receives user turns and observable
trip/wellness context only. Evaluator-only metadata such as `gold`, `scenario_state`,
`scenario_hooks`, `required_docs`, hidden blueprint notes, and acceptable-answer labels are withheld
from the solver and used only by the TA-side scorer. Robust solutions should infer constraints from
natural-language user requests, available tools, retrieved memory/documents, and inventory data,
rather than relying on public-only metadata fields being present at hidden time.

### Hidden metadata contract update (v4.6)

Public episodes may expose additional metadata for transparency and debugging. Hidden evaluation does not guarantee those metadata fields to solvers. In hidden runs, solvers should rely on user turns, available tools, retrieved memory/documents, and inventory data. Evaluator-only fields such as `scenario_state`, `scenario_hooks`, hidden gold/rubric metadata, and acceptable-answer labels are withheld. Core released trip context such as `budget_total`, `meeting_zone`, `weather`, and `family` remains available for compatibility.


## v4.8 scoring simplification

The official score is still one `/100` score, but v4.8 uses the simple component formula above.  Very small 1--3 point terms were removed from the official score and kept as diagnostics only.

Memory/tool retrieval metrics such as `memory_retrieval_rate`, `distributed_context_rate`, `spoken_rule_compliance_rate`, `active_context_hygiene_rate`, and `rationale_quality_rate` remain visible as diagnostics.  They are useful for debugging agent behavior, but they are not directly averaged into the official score.  Retrieval matters insofar as it helps the solver satisfy constraints, handle updates, retire stale conditions, avoid rejected options, and choose better soft-fit plans.

Soft constraints do exist.  They are preferences that should be optimized after hard feasibility is satisfied, such as quiet/sleep-friendly lodging, airport or meeting-zone convenience, plant-based dining fit, covered/indoor fallback quality, linked-perk usefulness, and other style or comfort preferences surfaced in user turns or retrieved context.
