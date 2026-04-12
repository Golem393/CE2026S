from __future__ import annotations

from dataclasses import dataclass
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
        session.bind_runner(self.runner)
        return session

    def primitive_tool_specs(self, session: TravelToolSession) -> List[Dict[str, Any]]:
        return session.tool_specs(primitive_only=True)

    def available_tool_specs(self, session: TravelToolSession) -> List[Dict[str, Any]]:
        return session.tool_specs(primitive_only=bool(self.system_config.get("primitive_tools_only", False)))

    def combine_usages(self, *usages: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return self.runner.combine_usages(*usages)

    def empty_usage(self) -> Dict[str, Any]:
        return self.runner.empty_usage()
