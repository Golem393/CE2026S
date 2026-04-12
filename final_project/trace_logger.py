from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _shrink(value: Any, *, depth: int = 0) -> Any:
    if depth >= 3:
        return str(value)[:200]
    if isinstance(value, str):
        return value if len(value) <= 240 else value[:237] + "..."
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for index, (key, inner) in enumerate(value.items()):
            if index >= 16:
                out["..."] = f"{len(value) - 16} more keys"
                break
            out[str(key)] = _shrink(inner, depth=depth + 1)
        return out
    if isinstance(value, list):
        return [_shrink(inner, depth=depth + 1) for inner in value[:8]] + (
            [f"... {len(value) - 8} more"] if len(value) > 8 else []
        )
    return str(value)[:200]


class TraceLogger:
    def __init__(self, path: str | Path, *, console: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.console = console
        self._started_at = time.perf_counter()
        self._handle = self.path.open("w", encoding="utf-8")
        self._lock = threading.Lock()

    def log(self, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(time.perf_counter() - self._started_at, 3),
            "event": event,
            **{key: _shrink(value) for key, value in fields.items()},
        }
        console_line = None
        if self.console:
            summary = []
            for key in ("system", "trip_id", "role", "agent", "tool", "attempt", "round_index", "status"):
                if key in record:
                    summary.append(f"{key}={record[key]}")
            detail = " ".join(summary)
            console_line = f"[trace] {event} {detail}".rstrip()
        with self._lock:
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._handle.flush()
            if console_line is not None:
                print(console_line, flush=True)

    def close(self) -> None:
        with self._lock:
            self._handle.flush()
            self._handle.close()
