from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class FlexibleCompatModel(BaseModel):
    """Backward-compatible base for older student helper schemas.

    Some early submissions imported extra typed containers from schemas.py
    (for example TripDetails or WorkingMemoryBoard).  The official evaluator
    only requires TravelDecision/MemoryReport, but keeping these permissive
    containers lets those submissions import and run without allowing students
    to override framework files such as llm_tools.py.
    """

    model_config = ConfigDict(extra="allow")


class TripDetails(FlexibleCompatModel):
    trip_id: Optional[str] = None
    city: Optional[str] = None
    origin: Optional[str] = None
    traveler_id: Optional[str] = None
    nights: Optional[int] = None
    budget_total: Optional[float] = None
    meeting_zone: Optional[str] = None
    weather: Optional[str] = None
    difficulty_tier: Optional[str] = None
    benchmark_family: Optional[str] = None


class WorkingMemoryBoard(FlexibleCompatModel):
    retrieved: List[str] = Field(default_factory=list)
    retired: List[str] = Field(default_factory=list)
    retired_docs: List[str] = Field(default_factory=list)
    active_context_keys: List[str] = Field(default_factory=list)
    active_docs: List[str] = Field(default_factory=list)
    docs_retrieved: List[str] = Field(default_factory=list)
    ignored_distractors: List[str] = Field(default_factory=list)
    rejected_option_notes: List[str] = Field(default_factory=list)
    notes: str = ""


class SpokenRuleHits(BaseModel):
    must_remember: List[str] = Field(default_factory=list)
    forbidden: List[str] = Field(default_factory=list)
    one_off_only: List[str] = Field(default_factory=list)
    retire: List[str] = Field(default_factory=list)
    do_not_reconsider: List[str] = Field(default_factory=list)
    keep_context_lean: List[str] = Field(default_factory=list)


class MemoryReport(BaseModel):
    retrieved: List[str] = Field(default_factory=list)
    retired: List[str] = Field(default_factory=list)
    retired_docs: List[str] = Field(default_factory=list)
    rejected_option_notes: List[str] = Field(default_factory=list)
    active_context_keys: List[str] = Field(default_factory=list)
    docs_retrieved: List[str] = Field(default_factory=list)
    active_docs: List[str] = Field(default_factory=list)
    ignored_distractors: List[str] = Field(default_factory=list)
    spoken_rule_hits: SpokenRuleHits = Field(default_factory=SpokenRuleHits)


class TravelDecision(BaseModel):
    flight_id: Optional[str] = None
    hotel_id: Optional[str] = None
    restaurant_id: Optional[str] = None
    activity_id: Optional[str] = None
    memory_report: MemoryReport = Field(default_factory=MemoryReport)
    notes: str = ''
    debug: Dict[str, Any] = Field(default_factory=dict)
    usage: Dict[str, Any] = Field(default_factory=dict)

    def to_evaluator_payload(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        payload = self.model_dump()
        payload["usage"] = usage
        return payload
