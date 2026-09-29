from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, status
from pydantic import BaseModel

router = APIRouter(tags=["findings"])


class TriagePatch(BaseModel):
    status: str
    notes: str | None = None


@router.get("/findings", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def list_findings(
    project_id: UUID | None = None,
    severity: str | None = None,
    detector: str | None = None,
    owasp_sc_2026: str | None = None,
    status_: str | None = None,
) -> dict[str, str]:
    return {"todo": "filter findings"}


@router.get("/findings/{finding_id}", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def get_finding(finding_id: UUID) -> dict[str, str]:
    return {"todo": "get finding", "id": str(finding_id)}


@router.post("/findings/{finding_id}/triage", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def triage(finding_id: UUID, body: TriagePatch) -> dict[str, str]:
    return {"todo": "triage", "id": str(finding_id)}


@router.post("/findings/merge", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def merge(ids: list[UUID]) -> dict[str, str]:
    return {"todo": "manual merge", "count": str(len(ids))}
