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

class TripDetails(BaseModel):
    trip_id: str = Field(default="", description="Unique identifier for the trip")
    family: str = Field(default="", description="The type of trip (e.g., business_travel, conference_trip)")
    origin: str = Field(default="", description="The departure city code (e.g., GMP)")
    city: str = Field(default="", description="The destination city code (e.g., OSA)")
    nights: int = Field(default=0, description="Number of nights for the trip")
    traveler_id: str = Field(default="", description="The ID of the traveler (e.g., traveler_consultant)")
    budget_total: float = Field(default=0.0, description="The total budget for the trip")
    meeting_zone: str = Field(default="", description="The zone where the meeting takes place (e.g., namba)")
    weather: str = Field(default="", description="Current weather condition (e.g., clear, rainy)")
    scenario_hooks: Dict[str, Any] = Field(default_factory=dict, description="Hooks indicating schedule volatility, event sensitivity, etc.")
    scenario_state: Dict[str, Any] = Field(default_factory=dict, description="Hidden state variables indicating true conditions (e.g., rainy, client_dinner)")

class CurrentItinerary(BaseModel):
    flight: Optional[Dict[str, Any]] = None
    hotel: Optional[Dict[str, Any]] = None
    restaurant: Optional[Dict[str, Any]] = None
    activity: Optional[Dict[str, Any]] = None

class MemoryAgentUpdate(BaseModel):
    """
    The strict JSON schema the Memory Agent will output on its turn.
    """
    hard_constraints: List[str] = Field(default_factory=list, description="Non-negotiable requirements (e.g., 'Budget strictly under $3000', 'Must arrive before 5 PM').")
    soft_constraints: List[str] = Field(default_factory=list, description="Preferences that should be accommodated if possible, but can be broken if necessary (e.g., 'Prefers Delta Airlines').")
    retired_constraints: List[str] = Field(default_factory=list, description="Constraints that were previously active but have been changed or removed by the user. Kept here so the Planner knows NOT to follow them anymore.")
    failed_searches: List[str] = Field(default_factory=list, description="A list of dead ends (e.g., 'Tried Marriott, fully booked') so the Planner doesn't repeat mistakes.")
    next_steps: str = Field(default="", description="The immediate next instruction for the Planner (e.g., 'Search for a flight on Tuesday').")
    evaluator_tracking: MemoryReport = Field(default_factory=MemoryReport)

class WorkingMemoryBoard(BaseModel):
    """
    This is the internal 'Memory Board' that your Planner agent will read from.
    It contains both the human-readable constraints needed by the Planner, 
    and the 'evaluator_tracking' object which you will return at the end for grading.
    """
    
    # ==========================================
    # STATIC DATA: Parsed once at the beginning
    # Do NOT ask the LLM to output or update this.
    # ==========================================
    trip_details: TripDetails = Field(default_factory=TripDetails)
    
    # ==========================================
    # ITINERARY DATA: Updated by the Orchestrator
    # ==========================================
    current_itinerary: CurrentItinerary = Field(default_factory=CurrentItinerary, description="What is currently booked. Use this to avoid booking the same thing twice.")

    # ==========================================
    # DYNAMIC DATA: Updated by merging MemoryAgentUpdate
    # ==========================================
    hard_constraints: List[str] = Field(default_factory=list, description="Non-negotiable requirements (e.g., 'Budget strictly under $3000', 'Must arrive before 5 PM').")
    soft_constraints: List[str] = Field(default_factory=list, description="Preferences that should be accommodated if possible, but can be broken if necessary (e.g., 'Prefers Delta Airlines').")
    retired_constraints: List[str] = Field(default_factory=list, description="Constraints that were previously active but have been changed or removed by the user. Kept here so the Planner knows NOT to follow them anymore.")
    failed_searches: List[str] = Field(default_factory=list, description="A list of dead ends (e.g., 'Tried Marriott, fully booked') so the Planner doesn't repeat mistakes.")
    next_steps: str = Field(default="", description="The immediate next instruction for the Planner (e.g., 'Search for a flight on Tuesday').")
    
    # This stores the exact keys the evaluator grades you on. 
    # Update this during the run by merging from MemoryAgentUpdate
    evaluator_tracking: MemoryReport = Field(default_factory=MemoryReport)
