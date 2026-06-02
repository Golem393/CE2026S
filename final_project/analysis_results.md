# Analysis of the Multi-Agent Pipeline

Based on the logs and the codebase, here is a detailed breakdown of what goes well, what goes wrong, and the proposed solutions for each phase.

## Phase 1: Memory Agent

**What goes well:**
*   The Memory Agent successfully extracts the relevant `hard_constraints` (`avoid_red_eye`, `prefer_quiet_hotel`, `loud_after_10pm`, `team_dietary_flex`, `protect_prep_evening`).
*   It effectively purges stale constraints into the `retired` list.
*   It captures the `rejected_options` (failed searches) so they won't be repeated.
*   It generates an excellent, concise `next_steps` summary string.

**What goes wrong:**
*   It extracts `meeting_zone` as a constraint key, but since `meeting_zone` is a string value (`namba`), the rule checker doesn't process it cleanly as a boolean toggle. However, this isn't fatal because the Planner receives `meeting_zone: namba` explicitly in its prompt block.

**Proposed Solution:**
*   This phase is largely functioning exactly as designed. The slight abstraction over `meeting_zone` can be safely ignored since the Planner already gets the value.

---

## Phase 2: Planner Agent

**What goes well:**
*   The Planner recognizes it needs missing components and calls the corresponding search tools (`search_flights`, `search_hotels`, etc.).
*   It uses the correct origin, destination, and city arguments for its searches.

**What goes wrong:**
*   **Lazy Tool Usage:** The Planner calls broad searches like `search_flights(GMP, OSA)` without using the available filter arguments (e.g., `quiet_min`, `dietary`).
*   **Ignoring Strict Constraints:** Even though the tools return `compact_item` dictionaries containing details, the LLM hallucinates or ignores the data. For example, it blindly selects `HT207` (which has a `quiet_score: 0.0`) despite the strict `prefer_quiet_hotel` constraint. It also picks `RS3004` (which only supports `vegan_preorder`) instead of a true vegan option, failing the `team_dietary_flex` constraint. Unlike the budget, these are critical functional constraints that the Planner *must* respect. By ignoring them, it misses the optimal gold bundle entirely (`HT206` + `RS3001`).
*   **Budget Blindness:** While we now know the budget can be slightly exceeded if a powerful bundle offsets the penalty, the Planner wasn't making a calculated trade-off. It just blindly picked `FL106` (445k) and `HT207` (436k), massively overshooting the 610k budget for no strategic gain.

**Proposed Solution:**
*   **Prompt Engineering:** Explicitly instruct the Planner to treat the budget as a *flexible* target (where going slightly over is acceptable if it secures a powerful bundle promotion), but to treat constraints like `quiet_score` and `dietary` as *absolute strict requirements*.
*   **Forced Filtering:** Instruct the Planner to *always* use filter arguments (`quiet_min=8.0`, `dietary="vegan"`) when calling search tools instead of relying on post-search reading.

---

## Phase 3: Python Pre-Check (`automated_rule_checker`)

**What goes well:**
*   The architecture correctly triggers the rule checker before the Verifier to save LLM tokens.

**What goes wrong:**
*   **Data Loss Bug:** In `student_solver.py`, the `_save_planner_proposals` function stores *only* the IDs back into the working memory board (e.g., `itin.hotel = {"hotel_id": "HT207"}`).
*   When the Python `automated_rule_checker` runs, it calls `hotel.get("quiet_score", 0)`. Since the dictionary only contains the ID, this defaults to `0`, throwing false-positive violation alerts (e.g., `Hotel HT207 quiet_score=0.0 is below threshold (7.0)`). 

**Proposed Solution:**
*   **Hydrate the Itinerary:** Modify `_save_planner_proposals` to fetch the full dictionary from `runtime.toolbox.env` (e.g., `next(item for item in env.hotels if item["hotel_id"] == proposed_id)`) and save that to the `current_itinerary` instead of just the ID. **(This seems to have been fixed or partially addressed as the Verifier now sees full objects like `{"flight_id": "FL101", "fare_total": 420000, ...}`)**.
*   **Custom Tools Integration:** The custom tools from `student_custom_tools_template.py` (like `automated_rule_checker`) are successfully being called by the Python orchestrator. This correctly injects `pre_check: WARNING...` into the memory board before the Verifier runs. However, the LLM agents themselves are *not* calling any custom tools—they only have access to the basic search tools.

---

## Phase 4: Verifier Agent & Revision Loop

**What goes well:**
*   It successfully identified violations based on the constraints provided to it (e.g., catching the budget violation and the dietary mismatch for RS3003).
*   It outputs clear reasons for rejection.

**What goes wrong:**
*   **The Verifier is Too Strict on Budget:** The Verifier rejects itineraries for exceeding the budget, even though the optimal scoring function might allow it if offset by a bundle.
*   **Revision Loop Bug:** When the Verifier rejects the itinerary, the system is supposed to do another Planner pass. However, there is a bug in `student_solver.py` in the `max_revision_rounds` logic:
    ```python
    revision_count += 1
    if revision_count >= max_revision_rounds:
        break  # Submit best-effort
    ```
    If `max_revision_rounds` defaults to 1, then after the first rejection, `revision_count` becomes 1, which triggers the `break` immediately. The Planner *never* gets a chance to revise the itinerary! It just submits the rejected plan as the final decision.
*   **Unverifiable Constraints:** The Verifier complains that it cannot verify constraints like `bundle_expires_after_late_checkin` because the Planner didn't provide enough detail.

**Proposed Solution:**
*   **Fix the Revision Loop Bug:** Change the condition to `if revision_count > max_revision_rounds:` or change `max_revision_rounds` in the configuration to at least 2.
*   **Relax Verifier Budget Rules:** Treat budget as a soft constraint or give the Verifier explicit rules on when it's acceptable to go over budget.
*   **Fix Planner Filtering (again):** Since the Planner still isn't using search filters, it will keep proposing invalid options. The prompt needs to strongly enforce using filters, or the custom tools in `student_custom_tools_template.py` (like `rerank_hotels` and `find_cheapest_compliant_combo`) should be exposed directly to the Planner agent as tools, rather than just being Python helpers.
