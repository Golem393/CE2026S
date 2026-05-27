from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm_runner import LLMRunner
from llm_tools import TravelToolbox, TravelToolSession


@dataclass
class StudentRuntime:
    runner: LLMRunner
    toolbox: TravelToolbox
    episode: Dict[str, Any]
    system_config: Dict[str, Any]
    role: str = "student"
    _sessions: List[TravelToolSession] = field(default_factory=list, init=False, repr=False)

    def _register_session(self, session: TravelToolSession) -> TravelToolSession:
        """Bind and register a tool session for official trace accounting.

        This is called both by StudentRuntime.new_session(...) and, during the
        official solve_episode(runtime) call, by TravelToolbox.new_session(...).
        The latter keeps older student solvers compatible even if they create
        sessions through runtime.toolbox.new_session(...).
        """
        if getattr(session, "runner", None) is not self.runner:
            session.bind_runner(self.runner)
        if all(existing is not session for existing in self._sessions):
            self._sessions.append(session)
        return session

    def new_session(
        self,
        *,
        role: Optional[str] = None,
        retrieval_strategy: Optional[str] = None,
        embedding_model: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> TravelToolSession:
        session = self.toolbox.new_session(
            episode=self.episode,
            retrieval_strategy=retrieval_strategy or self.system_config.get("retrieval_strategy", "lexical"),
            embedding_model=embedding_model if embedding_model is not None else self.system_config.get("embedding_model"),
            max_results=max_results or self.system_config.get("max_tool_results", 4),
            role=role or self.role,
        )
        return self._register_session(session)


    def trace_summary(self) -> Dict[str, Any]:
        docs_seen: List[str] = []
        rejected_memory_seen: List[str] = []
        rejected_option_notes_seen: List[str] = []
        retrieved_keys_seen: List[str] = []
        tool_trace: List[Dict[str, Any]] = []
        for session in self._sessions:
            summary = session.summary()
            tool_trace.extend(summary.get("tool_trace", []))
            for target, key in (
                (docs_seen, "docs_seen"),
                (rejected_memory_seen, "rejected_memory_seen"),
                (rejected_option_notes_seen, "rejected_option_notes_seen"),
                (retrieved_keys_seen, "retrieved_keys_seen"),
            ):
                for value in summary.get(key, []) or []:
                    if value not in target:
                        target.append(value)
        return {
            "tool_trace": tool_trace,
            "docs_seen": docs_seen,
            "rejected_memory_seen": rejected_memory_seen,
            "rejected_option_notes_seen": rejected_option_notes_seen,
            "retrieved_keys_seen": retrieved_keys_seen,
            "tool_call_count": len(tool_trace),
        }

    def primitive_tool_specs(self, session: TravelToolSession) -> List[Dict[str, Any]]:
        return session.tool_specs(primitive_only=True)

    def available_tool_specs(self, session: TravelToolSession) -> List[Dict[str, Any]]:
        return session.tool_specs(primitive_only=bool(self.system_config.get("primitive_tools_only", False)))

    def combine_usages(self, *usages: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return self.runner.combine_usages(*usages)

    def empty_usage(self) -> Dict[str, Any]:
        return self.runner.empty_usage()


def runtime_trace_summary(runtime: StudentRuntime) -> Dict[str, Any]:
    """Compatibility helper for TA graders that receive a StudentRuntime."""
    return runtime.trace_summary()
