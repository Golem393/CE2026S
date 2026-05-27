from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI
from trace_logger import TraceLogger


def load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class ModelUsage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    calls: int = 0
    estimated_cost_usd: float = 0.0

    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.embedding_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "embedding_tokens": self.embedding_tokens,
            "total_tokens": self.total_tokens(),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


class LLMRunner:
    def __init__(
        self,
        config: Dict[str, Any],
        *,
        dotenv_path: str | Path | None = None,
        trace_logger: TraceLogger | None = None,
    ) -> None:
        if dotenv_path is not None:
            load_env_file(dotenv_path)
        self.client = OpenAI()
        self.config = config
        self.trace_logger = trace_logger
        self.pricing = config["pricing_usd_per_1m_tokens"]
        self.allowed_generation_models = set(config["allowed_generation_models"])
        self.allowed_embedding_models = set(config["allowed_embedding_models"])
        self.response_timeout_seconds = float(config.get("response_timeout_seconds", 120))
        self.embedding_timeout_seconds = float(config.get("embedding_timeout_seconds", 90))
        self.response_max_retries = int(config.get("response_max_retries", 5))
        self.embedding_max_retries = int(config.get("embedding_max_retries", 4))
        self.response_retry_backoff_base_seconds = float(config.get("response_retry_backoff_base_seconds", 2.0))
        self.response_retry_backoff_cap_seconds = float(config.get("response_retry_backoff_cap_seconds", 18.0))
        self.response_retry_jitter_seconds = float(config.get("response_retry_jitter_seconds", 1.25))
        self._usage_ledger = self.empty_usage()

    def trace(self, event: str, **fields: Any) -> None:
        if self.trace_logger is not None:
            self.trace_logger.log(event, **fields)

    def usage_summary(self) -> Dict[str, Any]:
        return self.combine_usages(self._usage_ledger)

    def _record_observed_usage(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        self._usage_ledger = self.combine_usages(self._usage_ledger, usage)
        return usage

    def estimate_generation_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price = self.pricing[model]
        return input_tokens / 1_000_000 * price["input"] + output_tokens / 1_000_000 * price["output"]

    def estimate_embedding_cost(self, model: str, input_tokens: int) -> float:
        return input_tokens / 1_000_000 * self.pricing[model]["input"]

    def empty_usage(self) -> Dict[str, Any]:
        return {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "embedding_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "by_model": {},
        }

    def combine_usages(self, *usages: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        total = self.empty_usage()
        by_model: Dict[str, Dict[str, Any]] = {}
        for usage in usages:
            if not usage:
                continue
            total["calls"] += int(usage.get("calls", 0))
            total["input_tokens"] += int(usage.get("input_tokens", 0))
            total["output_tokens"] += int(usage.get("output_tokens", 0))
            total["embedding_tokens"] += int(usage.get("embedding_tokens", 0))
            total["estimated_cost_usd"] += float(usage.get("estimated_cost_usd", 0.0))
            for model, model_usage in usage.get("by_model", {}).items():
                slot = by_model.setdefault(
                    model,
                    {
                        "model": model,
                        "calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "embedding_tokens": 0,
                        "total_tokens": 0,
                        "estimated_cost_usd": 0.0,
                    },
                )
                slot["calls"] += int(model_usage.get("calls", 0))
                slot["input_tokens"] += int(model_usage.get("input_tokens", 0))
                slot["output_tokens"] += int(model_usage.get("output_tokens", 0))
                slot["embedding_tokens"] += int(model_usage.get("embedding_tokens", 0))
                slot["estimated_cost_usd"] += float(model_usage.get("estimated_cost_usd", 0.0))
        for model_usage in by_model.values():
            model_usage["total_tokens"] = model_usage["input_tokens"] + model_usage["output_tokens"] + model_usage["embedding_tokens"]
            model_usage["estimated_cost_usd"] = round(model_usage["estimated_cost_usd"], 6)
        total["total_tokens"] = total["input_tokens"] + total["output_tokens"] + total["embedding_tokens"]
        total["estimated_cost_usd"] = round(total["estimated_cost_usd"], 6)
        total["by_model"] = dict(sorted(by_model.items()))
        return total

    def _assert_generation_model(self, model: str) -> None:
        if model not in self.allowed_generation_models:
            raise ValueError(f"Generation model '{model}' is not allowed.")

    def _assert_embedding_model(self, model: str) -> None:
        if model not in self.allowed_embedding_models:
            raise ValueError(f"Embedding model '{model}' is not allowed.")

    def _generation_usage(self, model: str, usage: Any) -> Dict[str, Any]:
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        model_usage = ModelUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            calls=1,
            estimated_cost_usd=self.estimate_generation_cost(model, input_tokens, output_tokens),
        )
        payload = self.empty_usage()
        payload.update(
            {
                "calls": 1,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "embedding_tokens": 0,
                "total_tokens": model_usage.total_tokens(),
                "estimated_cost_usd": round(model_usage.estimated_cost_usd, 6),
                "by_model": {model: model_usage.to_dict()},
            }
        )
        return self._record_observed_usage(payload)

    def _embedding_usage(self, model: str, prompt_tokens: int) -> Dict[str, Any]:
        model_usage = ModelUsage(
            model=model,
            embedding_tokens=prompt_tokens,
            calls=1,
            estimated_cost_usd=self.estimate_embedding_cost(model, prompt_tokens),
        )
        payload = self.empty_usage()
        payload.update(
            {
                "calls": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "embedding_tokens": prompt_tokens,
                "total_tokens": model_usage.total_tokens(),
                "estimated_cost_usd": round(model_usage.estimated_cost_usd, 6),
                "by_model": {model: model_usage.to_dict()},
            }
        )
        return self._record_observed_usage(payload)

    def _base_kwargs(
        self,
        *,
        model: str,
        instructions: str,
        max_output_tokens: int,
        reasoning_effort: Optional[str],
        text_verbosity: Optional[str],
        metadata: Optional[Dict[str, str]],
        json_schema: Dict[str, Any],
        schema_name: str,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "max_output_tokens": max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": json_schema,
                    "strict": True,
                }
            },
        }
        if tools:
            kwargs["tools"] = tools
        if metadata:
            kwargs["metadata"] = metadata
        if model.startswith("gpt-5"):
            if reasoning_effort:
                kwargs["reasoning"] = {"effort": reasoning_effort}
            if text_verbosity:
                kwargs["text"]["verbosity"] = text_verbosity
        return kwargs

    def _parse_output_text(self, output_text: str) -> Dict[str, Any]:
        try:
            return json.loads(output_text)
        except json.JSONDecodeError as exc:
            snippet = output_text[:400].replace("\n", "\\n")
            raise RuntimeError(f"{exc}. Raw output prefix: {snippet}") from exc

    def _is_retryable_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        retryable_markers = [
            "error code: 429",
            "error code: 500",
            "error code: 502",
            "error code: 503",
            "error code: 504",
            "rate limit",
            "timeout",
            "timed out",
            "temporarily unavailable",
            "server_error",
            "api connection",
            "connection reset",
        ]
        return any(marker in message for marker in retryable_markers)

    def _retry_sleep_seconds(self, attempt: int) -> float:
        base = self.response_retry_backoff_base_seconds
        cap = self.response_retry_backoff_cap_seconds
        jitter = self.response_retry_jitter_seconds
        wait = min(cap, base * (2 ** max(0, attempt - 1)))
        if jitter > 0:
            wait += random.uniform(0.0, jitter)
        return wait

    def _repair_malformed_json(
        self,
        *,
        model: str,
        repair_model: Optional[str] = None,
        malformed_text: str,
        json_schema: Dict[str, Any],
        schema_name: str,
        max_output_tokens: int,
        reasoning_effort: Optional[str],
        text_verbosity: Optional[str],
        metadata: Optional[Dict[str, str]],
    ) -> Optional[Dict[str, Any]]:
        repair_prompt = (
            "The JSON below is malformed or truncated. Rewrite it as VALID JSON that matches the schema exactly. "
            "Preserve intended short benchmark keys and IDs when possible. Do not put prose inside arrays. "
            "If an item is uncertain, omit it instead of inventing it.\n\n"
            "Malformed JSON:\n"
            f"{malformed_text[:4000]}"
        )
        repair_model = repair_model or model
        kwargs = self._base_kwargs(
            model=repair_model,
            instructions="You repair malformed structured benchmark JSON.",
            max_output_tokens=max(300, min(max_output_tokens, 800)),
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
            metadata=metadata,
            json_schema=json_schema,
            schema_name=f"{schema_name}_repair",
        )
        kwargs["input"] = repair_prompt
        kwargs["timeout"] = self.response_timeout_seconds
        try:
            response = self.client.responses.create(**kwargs)
            repaired_text = getattr(response, "output_text", None)
            if not repaired_text:
                return None
            parsed = self._parse_output_text(repaired_text)
            usage = self._generation_usage(repair_model, getattr(response, "usage", None))
            self.trace(
                "responses_json_repair_success",
                model=repair_model,
                schema_name=schema_name,
                response_id=getattr(response, "id", None),
                usage=usage,
                **(metadata or {}),
            )
            return {
                "parsed": parsed,
                "usage": usage,
                "response_id": getattr(response, "id", None),
                "raw_output_text": repaired_text,
            }
        except Exception as exc:
            self.trace(
                "responses_json_repair_error",
                model=repair_model,
                schema_name=schema_name,
                error=str(exc),
                **(metadata or {}),
            )
            return None

    def create_json_response(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        json_schema: Dict[str, Any],
        schema_name: str = "travel_decision",
        max_output_tokens: int = 500,
        reasoning_effort: Optional[str] = None,
        text_verbosity: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        max_retries: Optional[int] = None,
        repair_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._assert_generation_model(model)
        trace_context = dict(metadata or {})
        kwargs = self._base_kwargs(
            model=model,
            instructions=instructions,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
            metadata=metadata,
            json_schema=json_schema,
            schema_name=schema_name,
        )
        kwargs["input"] = input_text
        kwargs["timeout"] = self.response_timeout_seconds
        max_retries = int(max_retries or self.response_max_retries)
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            started_at = time.perf_counter()
            self.trace(
                "responses_json_start",
                model=model,
                schema_name=schema_name,
                attempt=attempt,
                timeout_seconds=self.response_timeout_seconds,
                prompt_chars=len(input_text),
                **trace_context,
            )
            try:
                response = self.client.responses.create(**kwargs)
                output_text = getattr(response, "output_text", None)
                if not output_text:
                    raise RuntimeError("No structured JSON output returned.")
                usage = self._generation_usage(model, getattr(response, "usage", None))
                self.trace(
                    "responses_json_success",
                    model=model,
                    schema_name=schema_name,
                    attempt=attempt,
                    response_id=getattr(response, "id", None),
                    duration_ms=round((time.perf_counter() - started_at) * 1000, 1),
                    usage=usage,
                    **trace_context,
                )
                try:
                    parsed = self._parse_output_text(output_text)
                    repair_result = None
                except RuntimeError:
                    repair_result = self._repair_malformed_json(
                        model=model,
                        malformed_text=output_text,
                        json_schema=json_schema,
                        schema_name=schema_name,
                        max_output_tokens=max_output_tokens,
                        reasoning_effort=reasoning_effort,
                        text_verbosity=text_verbosity,
                        metadata=metadata,
                        repair_model=repair_model,
                    )
                    if not repair_result:
                        raise
                    parsed = repair_result["parsed"]
                    usage = self.combine_usages(usage, repair_result["usage"])
                return {
                    "parsed": parsed,
                    "usage": usage,
                    "response_id": getattr(response, "id", None),
                    "raw_output_text": output_text,
                    "repaired_output_text": None if not repair_result else repair_result.get("raw_output_text"),
                }
            except Exception as exc:
                last_error = exc
                self.trace(
                    "responses_json_error",
                    model=model,
                    schema_name=schema_name,
                    attempt=attempt,
                    duration_ms=round((time.perf_counter() - started_at) * 1000, 1),
                    error=str(exc),
                    **trace_context,
                )
                if attempt >= max_retries or not self._is_retryable_error(exc):
                    break
                time.sleep(self._retry_sleep_seconds(attempt))
        raise RuntimeError(f"Responses API call failed after {max_retries} attempts: {last_error}") from last_error

    def run_tool_agent_json(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        json_schema: Dict[str, Any],
        schema_name: str,
        tools: List[Dict[str, Any]],
        tool_handler: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        max_output_tokens: int,
        reasoning_effort: Optional[str] = None,
        text_verbosity: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        max_tool_rounds: int = 8,
        max_retries: Optional[int] = None,
        fallback_models: Optional[List[str]] = None,
        repair_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._assert_generation_model(model)
        trace_context = dict(metadata or {})
        tool_trace: List[Dict[str, Any]] = []
        response_ids: List[str] = []
        usage = self.empty_usage()
        max_retries = int(max_retries or self.response_max_retries)
        fallback_models = [candidate for candidate in (fallback_models or []) if candidate and candidate != model]

        def build_base_kwargs(active_model: str) -> Dict[str, Any]:
            return self._base_kwargs(
                model=active_model,
                instructions=instructions,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                text_verbosity=text_verbosity,
                metadata=metadata,
                json_schema=json_schema,
                schema_name=schema_name,
                tools=tools,
            )

        active_model = model
        base_kwargs = build_base_kwargs(active_model)

        def create_with_retry(**kwargs: Any):
            nonlocal active_model, base_kwargs
            candidate_models = [active_model] + [candidate for candidate in fallback_models if candidate != active_model]
            last_error: Exception | None = None
            for model_index, candidate_model in enumerate(candidate_models):
                active_model = candidate_model
                base_kwargs = build_base_kwargs(active_model)
                if model_index > 0:
                    self.trace(
                        "responses_tool_request_fallback_model",
                        schema_name=schema_name,
                        previous_model=candidate_models[model_index - 1],
                        fallback_model=active_model,
                        **trace_context,
                    )
                active_kwargs = dict(kwargs)
                for attempt in range(1, max_retries + 1):
                    started_at = time.perf_counter()
                    self.trace(
                        "responses_tool_request_start",
                        model=active_model,
                        schema_name=schema_name,
                        attempt=attempt,
                        timeout_seconds=self.response_timeout_seconds,
                        has_previous_response_id=bool(active_kwargs.get("previous_response_id")),
                        **trace_context,
                    )
                    try:
                        response = self.client.responses.create(**active_kwargs)
                        self.trace(
                            "responses_tool_request_success",
                            model=active_model,
                            schema_name=schema_name,
                            attempt=attempt,
                            response_id=getattr(response, "id", None),
                            duration_ms=round((time.perf_counter() - started_at) * 1000, 1),
                            **trace_context,
                        )
                        return response
                    except Exception as exc:
                        last_error = exc
                        self.trace(
                            "responses_tool_request_error",
                            model=active_model,
                            schema_name=schema_name,
                            attempt=attempt,
                            duration_ms=round((time.perf_counter() - started_at) * 1000, 1),
                            error=str(exc),
                            **trace_context,
                        )
                        if attempt >= max_retries or not self._is_retryable_error(exc):
                            break
                        time.sleep(self._retry_sleep_seconds(attempt))
            raise RuntimeError(f"Responses API call failed after {max_retries} attempts: {last_error}") from last_error

        self.trace(
            "tool_agent_start",
            model=active_model,
            schema_name=schema_name,
            max_tool_rounds=max_tool_rounds,
            tool_count=len(tools),
            prompt_chars=len(input_text),
            **trace_context,
        )
        response = create_with_retry(**{**base_kwargs, "input": input_text, "timeout": self.response_timeout_seconds})
        usage = self.combine_usages(usage, self._generation_usage(active_model, getattr(response, "usage", None)))
        response_ids.append(response.id)
        for round_index in range(max_tool_rounds):
            function_calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
            self.trace(
                "tool_agent_round",
                model=active_model,
                schema_name=schema_name,
                round_index=round_index,
                response_id=getattr(response, "id", None),
                function_call_count=len(function_calls),
                **trace_context,
            )
            if not function_calls:
                output_text = getattr(response, "output_text", None)
                if not output_text:
                    raise RuntimeError("Tool agent finished without structured output.")
                self.trace(
                    "tool_agent_finish",
                    model=active_model,
                    schema_name=schema_name,
                    round_index=round_index,
                    response_id=getattr(response, "id", None),
                    usage=usage,
                    tool_call_count=len(tool_trace),
                    **trace_context,
                )
                try:
                    parsed = self._parse_output_text(output_text)
                    repair_result = None
                except RuntimeError:
                    repair_result = self._repair_malformed_json(
                        model=active_model,
                        malformed_text=output_text,
                        json_schema=json_schema,
                        schema_name=schema_name,
                        max_output_tokens=max_output_tokens,
                        reasoning_effort=reasoning_effort,
                        text_verbosity=text_verbosity,
                        metadata=metadata,
                        repair_model=repair_model,
                    )
                    if not repair_result:
                        raise
                    parsed = repair_result["parsed"]
                    usage = self.combine_usages(usage, repair_result["usage"])
                    if repair_result.get("response_id"):
                        response_ids.append(repair_result.get("response_id"))
                return {
                    "parsed": parsed,
                    "usage": usage,
                    "response_ids": response_ids,
                    "tool_trace": tool_trace,
                    "raw_output_text": output_text,
                    "repaired_output_text": None if not repair_result else repair_result.get("raw_output_text"),
                }

            tool_outputs = []
            for call in function_calls:
                try:
                    arguments = json.loads(call.arguments or "{}")
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid tool call arguments for {call.name}: {call.arguments}") from exc
                self.trace(
                    "tool_agent_tool_call",
                    model=active_model,
                    schema_name=schema_name,
                    round_index=round_index,
                    tool=call.name,
                    call_id=getattr(call, "call_id", None),
                    arguments=arguments,
                    **trace_context,
                )
                tool_result = tool_handler(call.name, arguments)
                tool_trace.append({"tool": call.name, "arguments": arguments, "output_preview": str(tool_result)[:400]})
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

            response = create_with_retry(
                **{**base_kwargs, "previous_response_id": response.id, "input": tool_outputs, "timeout": self.response_timeout_seconds}
            )
            usage = self.combine_usages(usage, self._generation_usage(active_model, getattr(response, "usage", None)))
            response_ids.append(response.id)

        self.trace(
            "tool_agent_exceeded_rounds",
            model=active_model,
            schema_name=schema_name,
            max_tool_rounds=max_tool_rounds,
            tool_call_count=len(tool_trace),
            **trace_context,
        )
        raise RuntimeError(f"Exceeded max_tool_rounds={max_tool_rounds} without final structured output.")

    def embed_texts(self, *, model: str, texts: List[str], max_retries: Optional[int] = None) -> Dict[str, Any]:
        self._assert_embedding_model(model)
        max_retries = int(max_retries or self.embedding_max_retries)
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            started_at = time.perf_counter()
            self.trace(
                "embedding_start",
                model=model,
                attempt=attempt,
                text_count=len(texts),
                timeout_seconds=self.embedding_timeout_seconds,
            )
            try:
                response = self.client.embeddings.create(model=model, input=texts, timeout=self.embedding_timeout_seconds)
                usage = getattr(response, "usage", None)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                payload = self._embedding_usage(model, prompt_tokens)
                self.trace(
                    "embedding_success",
                    model=model,
                    attempt=attempt,
                    text_count=len(texts),
                    duration_ms=round((time.perf_counter() - started_at) * 1000, 1),
                    usage=payload,
                )
                return {
                    "vectors": [row.embedding for row in response.data],
                    "usage": payload,
                }
            except Exception as exc:
                last_error = exc
                self.trace(
                    "embedding_error",
                    model=model,
                    attempt=attempt,
                    text_count=len(texts),
                    duration_ms=round((time.perf_counter() - started_at) * 1000, 1),
                    error=str(exc),
                )
                if attempt >= max_retries or not self._is_retryable_error(exc):
                    break
                time.sleep(self._retry_sleep_seconds(attempt))
        raise RuntimeError(f"Embedding API call failed after {max_retries} attempts: {last_error}") from last_error
