from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

router = APIRouter(tags=["projects"])


class ProjectCreate(BaseModel):
    name: str
    scope: str = Field(..., description="Written authorization scope (mandatory).")
    default_chain: int = 1


class ProjectOut(BaseModel):
    id: UUID
    name: str
    scope: str
    default_chain: int


@router.post("/projects", status_code=status.HTTP_501_NOT_IMPLEMENTED, response_model=None)
def create_project(body: ProjectCreate) -> dict[str, str]:
    return {"todo": "persist project", "echo": body.model_dump_json()}


@router.get("/projects", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def list_projects() -> dict[str, str]:
    return {"todo": "list projects"}


@router.get("/projects/{project_id}", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def get_project(project_id: UUID) -> dict[str, str]:
    return {"todo": "get project", "id": str(project_id)}


@router.get("/projects/{project_id}/contracts", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def list_contracts(project_id: UUID) -> dict[str, str]:
    return {"todo": "list contracts", "project_id": str(project_id)}
