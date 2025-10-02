from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ResearchArtifactIn(BaseModel):
    tenant_id: int
    company_hint: Optional[str] = None
    website: Optional[str] = None
    path: str
    snapshot_md: str = Field(..., description="Markdown snapshot (bounded)")
    source_urls: List[str] = Field(default_factory=list)
    fit_signals: Dict[str, Any] = Field(default_factory=dict)


class ResearchImportRequest(BaseModel):
    tenant_id: int
    root: Optional[str] = Field(default=None, description="Server path to docs root (optional)")


class ResearchImportResult(BaseModel):
    files_scanned: int
    leads_upserted: int
    errors: List[str] = Field(default_factory=list)

