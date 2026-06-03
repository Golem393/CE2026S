import datetime
from agent_logger import SmartAgentLogger

logger = SmartAgentLogger(trip_id="test_trip")
ctx = {"budget_total": 1000.0, "quiet_weight": 5.0, "red_eye_ok": False, "rejected_ids": ["HT12"]}

logger._print(f"   \033[38;5;180m🔧 Reranking Context:\033[0m {ctx}")
logger._write_json({
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "event": "rerank_context",
    "trip_id": getattr(logger, "trip_id", ""),
    "context": ctx
})
logger.close()
