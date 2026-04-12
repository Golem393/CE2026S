
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List, Tuple


def parse_override_value(raw: str) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_set_override(raw: str) -> Tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Invalid --set override '{raw}'. Expected KEY=VALUE.")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid --set override '{raw}'. Override key cannot be empty.")
    return key, parse_override_value(value)


def budget_knob_specs(config: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    return deepcopy(config.get("student_tunable_budgets", {}))


def budget_caps_for_system(config: Dict[str, Any], system_name: str) -> Dict[str, Dict[str, Any]]:
    return budget_knob_specs(config).get(system_name, {})


def normalize_system_override_path(raw_key: str) -> Tuple[str, str]:
    parts = [part.strip() for part in raw_key.split(".") if part.strip()]
    if len(parts) == 2:
        return parts[0], parts[1]
    if len(parts) == 3 and parts[0] == "default_systems":
        return parts[1], parts[2]
    raise ValueError(
        f"Unsupported override path '{raw_key}'. Use SYSTEM.FIELD or default_systems.SYSTEM.FIELD."
    )


def apply_budget_overrides(
    systems: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    assignments: List[str],
) -> List[Dict[str, Any]]:
    applied: List[Dict[str, Any]] = []
    if not assignments:
        return applied
    knob_specs = budget_knob_specs(config)
    for raw in assignments:
        raw_key, value = parse_set_override(raw)
        system_name, field_name = normalize_system_override_path(raw_key)
        if system_name not in systems:
            raise ValueError(
                f"Override target '{system_name}' is not in the selected systems. Selected systems: {', '.join(sorted(systems))}."
            )
        allowed_fields = knob_specs.get(system_name, {})
        if field_name not in allowed_fields:
            allowed = ", ".join(sorted(allowed_fields)) or "(none)"
            raise ValueError(
                f"Field '{field_name}' is not a student-tunable budget knob for '{system_name}'. Allowed knobs: {allowed}."
            )
        spec = allowed_fields[field_name]
        if spec.get("type") == "integer" and not isinstance(value, int):
            raise ValueError(f"Override '{raw_key}' must be an integer, got {value!r}.")
        min_value = spec.get("min")
        max_value = spec.get("max")
        if min_value is not None and value < min_value:
            raise ValueError(f"Override '{raw_key}={value}' is below the minimum allowed value {min_value}.")
        if max_value is not None and value > max_value:
            raise ValueError(f"Override '{raw_key}={value}' exceeds the maximum allowed value {max_value}.")
        systems[system_name][field_name] = value
        applied.append({"system": system_name, "field": field_name, "value": value})
    return applied


def validate_system_budget_caps(systems: Dict[str, Dict[str, Any]], config: Dict[str, Any]) -> None:
    knob_specs = budget_knob_specs(config)
    for system_name, system_config in systems.items():
        for field_name, spec in knob_specs.get(system_name, {}).items():
            if field_name not in system_config:
                continue
            value = system_config[field_name]
            if spec.get("type") == "integer" and not isinstance(value, int):
                raise ValueError(f"Configured value for {system_name}.{field_name} must be an integer, got {value!r}.")
            min_value = spec.get("min")
            max_value = spec.get("max")
            if min_value is not None and value < min_value:
                raise ValueError(
                    f"Configured value for {system_name}.{field_name}={value} is below the minimum allowed value {min_value}."
                )
            if max_value is not None and value > max_value:
                raise ValueError(
                    f"Configured value for {system_name}.{field_name}={value} exceeds the maximum allowed value {max_value}."
                )
