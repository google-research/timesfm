from __future__ import annotations

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, status
from pydantic import BaseModel

router = APIRouter(tags=["sources"])


class SourceCreate(BaseModel):
    kind: Literal["git", "local", "address", "abi_bytecode"]
    uri: str
    ref: str | None = None
    abi: list[dict] | None = None
    bytecode: str | None = None


@router.post("/projects/{project_id}/sources", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def add_source(project_id: UUID, body: SourceCreate) -> dict[str, str]:
    return {"todo": "ingest source", "project_id": str(project_id)}


@router.get("/projects/{project_id}/sources", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def list_sources(project_id: UUID) -> dict[str, str]:
    return {"todo": "list sources", "project_id": str(project_id)}


@router.post("/sources/{source_id}/compile", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def compile_source(source_id: UUID) -> dict[str, str]:
    return {"todo": "compile", "source_id": str(source_id)}
