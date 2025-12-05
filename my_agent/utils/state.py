from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, Literal

from langgraph.graph import MessagesState


class ProfileState(TypedDict, total=False):
    company_profile: Dict[str, Any]
    icp_profile: Dict[str, Any]
    company_profile_confirmed: bool
    icp_profile_confirmed: bool
    icp_profile_generated: bool
    icp_discovery_confirmed: bool
    micro_icp_selected: bool
    awaiting_discovery_confirmation: bool
    awaiting_enrichment_confirmation: bool
    enrichment_confirmed: bool
    discovery_retry_requested: bool
    outstanding_prompts: List[str]
    customer_websites: List[str]
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
    planned_candidates: List[Dict[str, Any]]
    top10_ids: List[int]
    next40_ids: List[int]
    top10_domains: List[str]
    next40_domains: List[str]
    top10_details: List[Dict[str, Any]]
    next40_details: List[Dict[str, Any]]


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


class RunState(TypedDict, total=False):
    active_job_id: Optional[int]
    status: Literal['idle','running','pending_cancel','cancelled']
    awaiting_cancel_confirmation: bool


class OrchestrationState(MessagesState, total=False):
    tenant_id: Optional[int]
    thread_id: Optional[str]
    is_return_user: bool
    decisions: Dict[str, Any]
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
    status_history: List[Dict[str, Any]]
    # Run-cancellation guard: captures mid-run update/cancel state
    run: RunState
