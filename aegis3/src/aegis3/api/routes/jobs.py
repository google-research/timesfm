from __future__ import annotations

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

router = APIRouter(tags=["jobs"])


class ToolStep(BaseModel):
    tool: Literal["slither", "foundry", "echidna", "medusa", "mythril", "halmos"]
    options: dict = Field(default_factory=dict)
    timeout_s: int = 600


class JobCreate(BaseModel):
    source_id: UUID | None = None
    tools: list[ToolStep]
    budget_seconds: int = 7200
    egress_policy: Literal["deny", "allowlist"] = "deny"
    egress_allowlist: list[str] = Field(default_factory=list)


@router.post("/projects/{project_id}/jobs", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def create_job(project_id: UUID, body: JobCreate) -> dict[str, str]:
    return {"todo": "enqueue job", "project_id": str(project_id)}


@router.get("/jobs/{job_id}", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def get_job(job_id: UUID) -> dict[str, str]:
    return {"todo": "get job", "id": str(job_id)}


@router.get("/jobs/{job_id}/steps/{step_id}/logs", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def get_step_logs(job_id: UUID, step_id: UUID) -> dict[str, str]:
    return {"todo": "stream logs", "job_id": str(job_id), "step_id": str(step_id)}


@router.post("/jobs/{job_id}/cancel", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def cancel_job(job_id: UUID) -> dict[str, str]:
    return {"todo": "cancel", "job_id": str(job_id)}


@router.post("/jobs/{job_id}/rerun", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def rerun_job(job_id: UUID, only_failed: bool = False) -> dict[str, str]:
    return {"todo": "rerun", "job_id": str(job_id), "only_failed": str(only_failed)}
