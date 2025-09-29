from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Seed(BaseModel):
    seed_name: str
    domain: Optional[str] = None


class IntakeAnswers(BaseModel):
    website: Optional[str] = None
    # optional structured fields we may persist as-is
    geos: Optional[List[str]] = None
    integrations: Optional[List[str]] = None
    acv_usd: Optional[float] = None
    cycle_weeks: Optional[int] = Field(default=None, description="deal cycle in weeks")
    price_floor_usd: Optional[float] = None
    champion_titles: Optional[List[str]] = None
    triggers: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None


class IntakePayload(BaseModel):
    answers: Dict[str, Any]
    seeds: List[Seed]


class SuggestionCard(BaseModel):
    id: str
    title: str
    evidence_count: int = 0
    rationale: Optional[str] = None
    targeting_pack: Optional[Dict[str, Any]] = None
    negative_icp: Optional[List[Dict[str, Any]]] = None


class AcceptRequest(BaseModel):
    suggestion_id: Optional[str] = None
    suggestion_payload: Optional[Dict[str, Any]] = None
