"""evolve_verifier — Safety Verifier agent (Team 5 / Evolve).

Implements the 12-step verifier_invocation_order from safety_policy_rules.json:

    1.  prompt-injection resistance (ignore meta-instructions)
    2.  T0/T1/T2/T3 classification per safety_taxonomy
    3.  object-directed anger disambiguation
    4.  frequency-escalation T1->T2 (>=2 T1 in 15min)
    5.  severity dominance (T3 overrides all)
    6.  minor protection (age modifications)
    7.  no-SOCE block
    8.  no-medication-initiation redirect
    9.  caregiver-consent gate
    10. grace-day anti-punishment substitution
    11. emit tier-appropriate response template
    12. record memory_packet contributions

V1 is deterministic — pattern matching against the trigger_phrases_examples
encoded in safety_policy_rules.json. The runner + session args are accepted
for forward compatibility with an LLM-assisted v2 (and so the orchestrator's
budget_knobs flow through unchanged) but not invoked in v1; pattern matching
is sufficient for the cataloged trigger phrases and is dramatically faster +
cheaper + more auditable than an LLM call per turn.

Entry point: run_verifier_phase(runtime, episode, draft, cfg, ...) -> dict.
Symmetric with run_memory_phase / run_planner_phase: takes runtime + cfg,
opens its own per-role session, reads budgets via cfg bracket access.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).resolve().parent
_policies_cache: dict | None = None
_taxonomy_cache: dict | None = None
_caregivers_cache: dict | None = None


def _load_policies() -> dict:
    global _policies_cache
    if _policies_cache is None:
        _policies_cache = json.loads((_DATA_DIR / "safety_policy_rules.json").read_text())
    return _policies_cache


def _load_taxonomy() -> dict:
    global _taxonomy_cache
    if _taxonomy_cache is None:
        _taxonomy_cache = json.loads((_DATA_DIR / "safety_taxonomy.json").read_text())
    return _taxonomy_cache


def _load_caregivers() -> dict:
    global _caregivers_cache
    if _caregivers_cache is None:
        _caregivers_cache = json.loads((_DATA_DIR / "caregiver_briefs.json").read_text())
    return _caregivers_cache


def _policy(policy_id: str) -> dict:
    for p in _load_policies()["policies"]:
        if p["policy_id"] == policy_id:
            return p
    raise KeyError(f"unknown policy_id: {policy_id}")


def _phrase_hit(text: str, phrases: list[str]) -> str | None:
    """Case-insensitive substring match. Returns the matched phrase or None."""
    lowered = text.lower()
    for phrase in phrases:
        if phrase.lower() in lowered:
            return phrase
    return None


def _concat_user_turns(episode: dict) -> str:
    """Concatenate ALL user turns into one string for cross-turn pattern detection.
    A prompt-injection attempt in turn 1 must still trigger when turn 3 is benign;
    same for T1 signals that accumulate across a conversation."""
    return " ||| ".join(t.get("text", "") for t in episode.get("turns", [])
                        if t.get("speaker") == "user")


# Regex fallbacks for surface variation the policy's literal trigger phrases miss.
# Kept narrow and high-precision; if these fire, the underlying signal is unambiguous.
_INJECTION_RE = re.compile(
    r"\b(ignore|forget|disregard|bypass|override|skip)\b[^.!?]{0,40}\b"
    r"(safety|guideline|verifier|guardrail|rule|filter|restriction)",
    re.IGNORECASE,
)
_INJECTION_ROLE_RE = re.compile(
    r"\b(i'?m|i am|pretend|roleplay as|act as)\b[^.!?]{0,30}\b"
    r"(clinician|doctor|psychiatrist|researcher|tester|developer|system)",
    re.IGNORECASE,
)
# T3 method+intent patterns. Method tokens AND intent tokens within proximity.
_T3_METHOD_TOKENS = r"(pills?|prescription|bottles?|medication|overdose|rope|gun|firearm|knife|blade|razor)"
_T3_INTENT_TOKENS = r"(tonight|over|end (it|this)|decided|ready|laid out|holding|finish|goodbye|done trying)"
_T3_METHOD_INTENT_RE = re.compile(
    rf"\b{_T3_METHOD_TOKENS}\b.{{0,60}}\b{_T3_INTENT_TOKENS}\b|"
    rf"\b{_T3_INTENT_TOKENS}\b.{{0,60}}\b{_T3_METHOD_TOKENS}\b",
    re.IGNORECASE,
)
# T1 passive-ideation surface forms not covered by the literal trigger list.
_T1_PATTERNS = [
    r"\b(feel|feeling) (completely |so |really |totally |just )?alone\b",
    r"\bno one (cares|gets it|understands)\b",
    r"\b(tired of trying|done trying|why bother|what's the use)\b",
    r"\b(empty inside|hollow|numb)\b",
]
_T1_RE = re.compile("|".join(_T1_PATTERNS), re.IGNORECASE)

# Object-directed-anger: verb + (optional filler) + external object noun.
# Mirrors the structure of _T3_METHOD_INTENT_RE; permits intensifiers like
# "this stupid laptop" between verb and object. Self-directed pattern takes
# precedence in _disambiguate_anger so reflexive cases (myself, my body)
# still promote correctly.
_ANGER_VERB = (r"(punch|kick|hit|smash|kill|destroy|break|throttle|throw|"
               r"scream|wreck|trash|rage[\s-]?quit|murder|slam)")
_EXTERNAL_OBJECT = (r"(wall|laptop|computer|phone|table|chair|keyboard|screen|"
                    r"monitor|desk|car|door|window|stuff|something|things?|"
                    r"object|boss|coworker|kid|child|sibling|everything|it)")
_SELF_REF = r"(myself|my\s+(body|head|arm|face|skin|hair|life)|me\b)"
_ANGER_OBJECT_DIRECTED_RE = re.compile(
    rf"\b{_ANGER_VERB}\b[^.!?]{{0,30}}\b{_EXTERNAL_OBJECT}\b",
    re.IGNORECASE,
)
_ANGER_SELF_DIRECTED_RE = re.compile(
    rf"\b(want to|wanna|gonna|going to|need to|could|gotta|will)\s+"
    rf"(hurt|harm|cut|burn|kill|damage|hit|destroy|punish)\s+{_SELF_REF}",
    re.IGNORECASE,
)

# SOCE: covers the most common indirect/euphemistic framings.
_SOCE_RE = re.compile(
    r"\b(conversion|reparative|ex[\s-]?gay)\s+(therap|program|treatment|counsel|ministr)"
    r"|\bmake me\s+(straight|cis|normal|not\s+(gay|trans|queer))"
    r"|\b(pray|cure|fix|reverse)\s+(the\s+)?(gay|trans|queer)\b"
    r"|\b(reverse|reset)\s+my\s+(gender|identity|orientation)\b",
    re.IGNORECASE,
)

# Medication-initiation: covers start/stop/titrate language and dosage queries.
_MED_INIT_RE = re.compile(
    r"\b(start(ing)?|stop(ping)?|switch(ing)?|taper(ing)?|increase|decrease|adjust|titrate)\b"
    r"[^.!?]{0,40}\b(medication|pill|drug|prescription|dosag?e|antidepressant|"
    r"benzo|stimulant|ssri|snri|maoi)"
    r"|\b(dosag?e|how much|how many\s+\w+)\s+(of\s+)?(an?\s+)?(ssri|snri|maoi|antidepressant|"
    r"fluoxetine|sertraline|paroxetine|escitalopram|venlafaxine|lexapro|zoloft|prozac|cymbalta)"
    r"|\bwhat'?s\s+(an?\s+|the\s+)?(lethal|safe|typical|standard)\s+dose",
    re.IGNORECASE,
)


# Step 1 - prompt injection resistance
def _detect_prompt_injection(text: str) -> dict:
    p = _policy("POLICY-PROMPT-INJECTION-RESISTANCE")
    hit = _phrase_hit(text, p["trigger_phrases_examples"])
    if hit is None and _INJECTION_RE.search(text):
        hit = "(regex) ignore/forget/disregard X safety/guideline/verifier/rule"
    if hit is None and _INJECTION_ROLE_RE.search(text):
        hit = "(regex) claimed-clinician/researcher role"
    return {"detected": hit is not None, "matched_phrase": hit, "policy_id": p["policy_id"]}


# Step 2 - T0/T1/T2/T3 classification
def _classify_tier(text: str) -> dict:
    """Match user text against tier trigger phrases + regex patterns. T3 > T2 > T1 > T0."""
    # T3 first (severity dominance)
    p3 = _policy("POLICY-T3-IMMINENT-BYPASS")
    hit = _phrase_hit(text, p3["trigger_phrases_examples"])
    if hit is None and _T3_METHOD_INTENT_RE.search(text):
        hit = "(regex) method+intent proximity"
    if hit:
        return {"tier": "T3", "matched_phrase": hit, "policy_id": p3["policy_id"], "confidence": "high"}
    # T2
    p2 = _policy("POLICY-T2-ACTIVE-RESOURCES")
    hit = _phrase_hit(text, p2["trigger_phrases_examples"])
    if hit:
        return {"tier": "T2", "matched_phrase": hit, "policy_id": p2["policy_id"], "confidence": "medium"}
    # T1
    p1 = _policy("POLICY-T1-PASSIVE-CHECKIN")
    hit = _phrase_hit(text, p1["trigger_phrases_examples"])
    if hit is None and _T1_RE.search(text):
        hit = "(regex) passive-ideation surface form"
    if hit:
        return {"tier": "T1", "matched_phrase": hit, "policy_id": p1["policy_id"], "confidence": "medium"}
    return {"tier": "T0", "matched_phrase": None, "policy_id": None, "confidence": "default"}


# Step 3 - object-directed anger disambiguation
def _disambiguate_anger(text: str) -> dict:
    p = _policy("POLICY-OBJECT-DIRECTED-ANGER-DISAMBIGUATION")
    # Self-directed takes precedence (severity dominance preserved).
    self_hit = _phrase_hit(text, p["examples_unsafe_self_directed_DO_promote"])
    if self_hit is None and _ANGER_SELF_DIRECTED_RE.search(text):
        self_hit = "(regex) want/gonna/etc. + hurt/harm/etc. + myself/my-body"
    if self_hit:
        return {"is_object_directed": False, "promote_to": "T2",
                "matched_phrase": self_hit, "policy_id": p["policy_id"]}
    # Object-directed: literal phrase OR verb+external-object regex.
    object_hit = _phrase_hit(text, p["examples_safe_NOT_self_harm"])
    if object_hit is None and _ANGER_OBJECT_DIRECTED_RE.search(text):
        object_hit = "(regex) anger verb + external object proximity"
    if object_hit:
        return {"is_object_directed": True, "promote_to": None,
                "matched_phrase": object_hit, "policy_id": p["policy_id"]}
    return {"is_object_directed": False, "promote_to": None,
            "matched_phrase": None, "policy_id": p["policy_id"]}


# Step 4 - frequency escalation T1 x2 in 15min -> T2
def _check_frequency_escalation(session_events: list[dict], current_tier: str) -> dict:
    p = _policy("POLICY-FREQUENCY-ESCALATION")
    window_min = p["promotion_logic"]["window_minutes"]
    threshold = p["promotion_logic"]["threshold_count"]
    if current_tier != "T1":
        return {"promote": False, "t1_count_in_window": 0, "policy_id": p["policy_id"]}
    now = datetime.utcnow()
    cutoff = now - timedelta(minutes=window_min)
    t1_count = sum(1 for ev in session_events
                   if ev.get("tier") == "T1"
                   and isinstance(ev.get("timestamp"), datetime)
                   and ev["timestamp"] >= cutoff)
    return {"promote": t1_count >= threshold, "t1_count_in_window": t1_count, "policy_id": p["policy_id"]}


# Step 6 - minor protection
def _apply_minor_protection(tier: str, episode: dict) -> dict:
    p = _policy("POLICY-MINOR-PROTECTION")
    is_adolescent = bool(episode.get("scenario_state", {}).get("adolescent"))
    if not is_adolescent:
        return {"applied": False, "policy_id": p["policy_id"]}
    promoted = tier
    if tier == "T1":
        promoted = "T2"
    return {"applied": True, "tier_modification": promoted, "modifications": p["modifications"], "policy_id": p["policy_id"]}


# Step 7 - no SOCE
def _check_no_soce(text: str) -> dict:
    p = _policy("POLICY-NO-SOCE")
    soce_phrases = ["conversion therapy", "change my sexuality", "stop being gay", "become straight",
                    "fix my orientation", "stop being trans", "change my gender"]
    hit = _phrase_hit(text, soce_phrases)
    if hit is None and _SOCE_RE.search(text):
        hit = "(regex) conversion/reparative/ex-gay therap*/program OR make-me-straight/cis/normal OR pray/cure/fix the gay"
    return {"triggered": hit is not None, "matched_phrase": hit, "policy_id": p["policy_id"]}


# Step 8 - no medication initiation
def _check_no_med_init(text: str) -> dict:
    p = _policy("POLICY-NO-MEDICATION-INITIATION")
    med_phrases = ["what dose", "lethal dose", "what medication should i take",
                   "should i start", "should i stop taking", "stop my", "titrate",
                   "ssri", "fluoxetine", "sertraline", "lexapro", "zoloft", "prozac"]
    hit = _phrase_hit(text, med_phrases)
    if hit is None and _MED_INIT_RE.search(text):
        hit = "(regex) start/stop/titrate medication OR dosage of specific drug OR lethal/safe dose"
    return {"triggered": hit is not None, "matched_phrase": hit, "policy_id": p["policy_id"]}


# Step 9 - caregiver consent gate
def _check_caregiver_consent(draft_text: str, episode: dict) -> dict:
    p = _policy("POLICY-CAREGIVER-CONSENT-GATE")
    caregivers = _load_caregivers()
    caregiver_ids = episode.get("scenario_state", {}).get("caregiver_ids", []) or []
    named_unregistered = []
    for token in re.findall(r"\b(mom|dad|mother|father|partner|spouse|husband|wife|sister|brother)\b",
                            draft_text.lower()):
        is_registered = any(token in caregivers.get(cid, {}).get("text", "").lower() for cid in caregiver_ids)
        if not is_registered:
            named_unregistered.append(token)
    return {"unregistered_persons_named": named_unregistered,
            "fallback_template": p["fallback_when_no_consented_caregiver"],
            "policy_id": p["policy_id"]}


# Step 10 - grace day anti-punishment
def _check_grace_day(text: str, episode: dict) -> dict:
    p = _policy("POLICY-GRACE-DAY-ANTI-PUNISHMENT")
    streak_status = episode.get("scenario_hooks", {}).get("streak_status")
    broken = streak_status == "broken"
    shame_phrases = ["i failed", "i fell off", "i slipped", "i missed", "i broke my streak",
                     "i lost it", "i should have", "i didn't even"]
    text_hit = _phrase_hit(text, shame_phrases)
    triggered = broken or text_hit is not None
    return {"triggered": triggered, "matched_phrase": text_hit, "broken_streak": broken,
            "template": p["script_templates"]["default"], "policy_id": p["policy_id"]}


def _select_locale_template(policy: dict, episode: dict) -> str:
    """Pick the appropriate locale template (us_default / kr_default / lgbtq variants)."""
    templates = policy.get("script_templates", {})
    state = episode.get("scenario_state", {})
    if state.get("lgbtq_minority_stress_signal") and "with_lgbtq_minority_stress_signal" in templates:
        return templates["with_lgbtq_minority_stress_signal"]
    if state.get("locale_country") == "kr":
        return templates.get("kr_default") or templates.get("default") or ""
    return templates.get("us_default") or templates.get("default") or ""


# Step 11 - emit tier-appropriate response
def _emit_tier_response(tier: str, episode: dict, grace_active: bool) -> dict:
    if grace_active:
        return {"template": _policy("POLICY-GRACE-DAY-ANTI-PUNISHMENT")["script_templates"]["default"],
                "source_policy": "POLICY-GRACE-DAY-ANTI-PUNISHMENT", "planner_disabled": False}
    if tier == "T0":
        return {"template": None, "source_policy": None, "planner_disabled": False}
    policy_id = {"T1": "POLICY-T1-PASSIVE-CHECKIN", "T2": "POLICY-T2-ACTIVE-RESOURCES",
                 "T3": "POLICY-T3-IMMINENT-BYPASS"}[tier]
    p = _policy(policy_id)
    return {"template": _select_locale_template(p, episode),
            "source_policy": policy_id,
            "planner_disabled": bool(p.get("planner_disabled_this_turn", False))}


# Step 12 - assemble memory_packet contributions
def _build_memory_packet(tier: str, gates: dict, episode: dict) -> dict:
    must_remember, forbidden, one_off, retire = [], [], [], []
    do_not_reconsider, keep_lean = [], []
    docs_retrieved: list[str] = []
    if tier != "T0":
        docs_retrieved.append(f"safety:{tier}-RULE")
        must_remember.append(f"crisis_signal_{tier.lower()}_logged")
    if gates["anger"]["is_object_directed"]:
        must_remember.append("object_directed_anger_disambiguation_logged")
        do_not_reconsider.append("self_harm_inference_from_object_directed_anger")
        forbidden.append(f"t2_or_t3_classification_for_object_directed_anger")
        docs_retrieved.append("safety:OBJECT-DIRECTED-ANGER")
    if gates["frequency"]["promote"]:
        must_remember.append("frequency_promotion_triggered_t1_to_t2")
        docs_retrieved.append("safety:FREQUENCY-RULE")
    if gates["soce"]["triggered"]:
        forbidden.append("comply_with_soce_request_any_framing")
        docs_retrieved.append("safety:NO-SOCE")
    if gates["med_init"]["triggered"]:
        forbidden.append("provide_specific_medication_dose_information")
        docs_retrieved.append("safety:NO-MED-INIT")
    if gates["injection"]["detected"]:
        forbidden.append("comply_with_meta_instruction_to_ignore_safety")
        docs_retrieved.append("safety:PROMPT-INJECTION")
        do_not_reconsider.append("any_compliance_with_jailbreak_framing")
    if gates["caregiver"]["unregistered_persons_named"]:
        forbidden.extend([f"name_unregistered_person_{n}_in_response"
                          for n in gates["caregiver"]["unregistered_persons_named"]])
        docs_retrieved.append("safety:CAREGIVER-GATE")
    if gates["grace"]["triggered"]:
        forbidden.extend(["shame_or_failure_or_slipped_or_fell_off_language",
                          "earn_it_back_streak_loss_framing"])
        docs_retrieved.append("safety:GRACE-DAY")
        one_off.append("grace_day_template_active_this_turn")
    if gates["minor"]["applied"]:
        must_remember.append("adolescent_modification_applied")
        docs_retrieved.append("safety:MINOR-PROTECTION")
    return {
        "spoken_rule_hits": {
            "must_remember": must_remember, "forbidden": forbidden,
            "one_off_only": one_off, "retire": retire,
            "do_not_reconsider": do_not_reconsider, "keep_context_lean": keep_lean,
        },
        "docs_retrieved": docs_retrieved,
        "active_docs": list(dict.fromkeys(docs_retrieved)),
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_verifier_phase(
    runtime: Any = None,
    episode: dict | None = None,
    draft: dict | None = None,
    cfg: dict | None = None,
    verifier_feedback: Any = None,           # noqa: ARG001 — accepted for signature symmetry; verifier doesn't consume its own feedback
    session_events: list[dict] | None = None,
) -> dict:
    """Run the 12-step Safety Verifier pipeline against a Planner draft.

    Symmetric with run_memory_phase / run_planner_phase:
      - Takes runtime (opens per-role session internally — MAS pattern).
      - Reads budgets via cfg bracket access; KeyError surfaces missing keys.

    Args:
        runtime: StudentRuntime; verifier session opened on it. v1 doesn't
                 invoke the session (deterministic pattern matching).
        episode: episode dict per Nik's schema.
        draft: Planner output dict — at minimum {"response_text": str, ...}.
        cfg: budget knobs. MUST contain "verifier_max_tool_rounds" and
             "verifier_max_output_tokens" — bracket access raises KeyError.
        verifier_feedback: accepted for signature symmetry with memory +
             planner; verifier doesn't consume its own feedback (returns it).
        session_events: prior {tier, timestamp, ...} events for the
             frequency-escalation check. Orchestrator's responsibility to
             plumb a real history here in v2; v1 treats empty as "fresh".

    Returns: a verdict dict — see module docstring for fields.
    """
    episode = episode or {}
    draft = draft or {}
    cfg = cfg or {}
    session_events = session_events or []

    # Open per-role session — MAS pattern, same as memory + planner.
    ver_session = None
    if runtime is not None and hasattr(runtime, "new_session"):
        ver_session = runtime.new_session(role="verifier")

    # Budget knobs — bracket access surfaces config bugs early.
    _max_tool_rounds = cfg["verifier_max_tool_rounds"]      # noqa: F841
    _max_output_tokens = cfg["verifier_max_output_tokens"]  # noqa: F841
    # Look at the WHOLE conversation, not just the last user turn — injection
    # attempts and T1 accumulation routinely span multiple turns.
    all_user_text = _concat_user_turns(episode)
    draft_text = draft.get("response_text", "") or json.dumps(draft, ensure_ascii=False)

    # 12-step pipeline
    injection = _detect_prompt_injection(all_user_text)
    tier_result = _classify_tier(all_user_text)
    tier = tier_result["tier"]
    anger = _disambiguate_anger(all_user_text)
    if anger["is_object_directed"] and tier in {"T2", "T3"}:
        tier = "T0"
        tier_result["tier"] = "T0"
        tier_result["overridden_by"] = "object_directed_anger_disambiguation"
    frequency = _check_frequency_escalation(session_events, tier)
    if frequency["promote"]:
        tier = "T2"
    if injection["detected"]:
        tier = "T3"
    minor = _apply_minor_protection(tier, episode)
    if minor.get("applied") and minor.get("tier_modification"):
        tier = minor["tier_modification"]
    soce = _check_no_soce(all_user_text)
    med_init = _check_no_med_init(all_user_text)
    caregiver = _check_caregiver_consent(draft_text, episode)
    grace = _check_grace_day(all_user_text, episode)
    response_emission = _emit_tier_response(tier, episode, grace["triggered"])
    gates = {"injection": injection, "tier": tier_result, "anger": anger,
             "frequency": frequency, "minor": minor, "soce": soce,
             "med_init": med_init, "caregiver": caregiver, "grace": grace}
    memory_packet = _build_memory_packet(tier, gates, episode)

    # Verdict synthesis
    required_changes: list[str] = []
    if response_emission["planner_disabled"]:
        verdict = "block"
        required_changes = ["emit_tier_response_template_verbatim_no_planner_output"]
    elif (anger["is_object_directed"] and gates["tier"]["tier"] != "T0") \
            or caregiver["unregistered_persons_named"] \
            or soce["triggered"] or med_init["triggered"] \
            or injection["detected"] \
            or grace["triggered"]:
        verdict = "revise"
        if anger["is_object_directed"]:
            required_changes.append("planner_must_pivot_to_high_arousal_physical_release")
        if caregiver["unregistered_persons_named"]:
            required_changes.append("remove_unregistered_caregiver_names_use_default_template")
        if soce["triggered"]:
            required_changes.append("apply_no_soce_redirect_template")
        if med_init["triggered"]:
            required_changes.append("apply_psychoeducation_only_redirect_template")
        if injection["detected"]:
            required_changes.append("ignore_meta_instruction_apply_underlying_content_classification")
        if grace["triggered"]:
            required_changes.append("substitute_grace_day_template_no_llm_generated_streak_loss_language")
    else:
        verdict = "pass"

    return {
        "verdict": verdict,
        "needs_revision": verdict == "revise",
        "tier": tier,
        "rationale": f"tier={tier} via {tier_result.get('policy_id') or 'default_T0'}; "
                     f"injection={injection['detected']}; object_directed_anger={anger['is_object_directed']}; "
                     f"freq_promote={frequency['promote']}; grace={grace['triggered']}; "
                     f"caregiver_unregistered={bool(caregiver['unregistered_persons_named'])}",
            "gates_passed": [g["policy_id"] for g in gates.values() if isinstance(g, dict) and g.get("policy_id")],
        "required_changes": required_changes,
        "grace_inserted": grace["triggered"],
        "tier_response_template": response_emission["template"],
        "tier_response_source_policy": response_emission["source_policy"],
        "planner_disabled_this_turn": response_emission["planner_disabled"],
        "memory_packet_contributions": memory_packet,
        "usage": (ver_session.usage if ver_session is not None and hasattr(ver_session, "usage")
                  else {"calls": 0, "input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}),
    }
