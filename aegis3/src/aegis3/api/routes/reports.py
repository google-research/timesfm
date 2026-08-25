from __future__ import annotations

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, status
from pydantic import BaseModel

router = APIRouter(tags=["reports"])


class ReportRequest(BaseModel):
    format: Literal["md", "pdf", "sarif"] = "md"
    template: Literal["bounty", "internal"] = "bounty"


@router.post("/projects/{project_id}/reports", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def render(project_id: UUID, body: ReportRequest) -> dict[str, str]:
    return {"todo": "render report", "format": body.format, "template": body.template}


@router.get("/reports/{report_id}", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def get_report(report_id: UUID) -> dict[str, str]:
    return {"todo": "signed download url", "id": str(report_id)}
