from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import MessagesState


class ProfileState(TypedDict, total=False):
    company_profile: Dict[str, Any]
    icp_profile: Dict[str, Any]
    company_profile_confirmed: bool
    icp_profile_confirmed: bool
    micro_icp_selected: bool
    awaiting_discovery_confirmation: bool
    outstanding_prompts: List[str]
    last_updated_at: Optional[str]


class NormalizeState(TypedDict, total=False):
    processed_rows: int
    errors: List[str]
    last_run_at: Optional[str]


class DiscoveryState(TypedDict, total=False):
    candidate_ids: List[int]
    strategy: str
    last_ssic_attempt: Optional[str]
    diagnostics: Dict[str, Any]


class EnrichmentResult(TypedDict, total=False):
    company_id: int
    completed: bool
    error: Optional[str]
    vendor_usage: Dict[str, Any]


class ScoringState(TypedDict, total=False):
    scores: List[Dict[str, Any]]
    last_run_at: Optional[str]


class ExportState(TypedDict, total=False):
    next40_enqueued: bool
    odoo_exported: bool
    job_ids: List[str]
    last_run_at: Optional[str]


class StatusState(TypedDict, total=False):
    phase: str
    message: str
    updated_at: Optional[str]


class OrchestrationState(MessagesState, total=False):
    entry_context: Dict[str, Any]
    profile_state: ProfileState
    icp_payload: Dict[str, Any]
    normalize: NormalizeState
    discovery: DiscoveryState
    top10: Dict[str, Any]
    enrichment_results: List[EnrichmentResult]
    scoring: ScoringState
    exports: ExportState
    status: StatusState
