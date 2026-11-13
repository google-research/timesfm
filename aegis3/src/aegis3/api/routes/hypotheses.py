from __future__ import annotations

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, status
from pydantic import BaseModel

router = APIRouter(tags=["hypotheses"])


class GenerateRequest(BaseModel):
    backend: Literal["rule", "ollama", "anthropic", "openai", "none"] = "rule"
    max: int = 10


@router.post("/projects/{project_id}/hypotheses/generate", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def generate(project_id: UUID, body: GenerateRequest) -> dict[str, str]:
    return {"todo": "generate hypotheses", "backend": body.backend}


@router.get("/projects/{project_id}/hypotheses", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def list_hypotheses(project_id: UUID) -> dict[str, str]:
    return {"todo": "list hypotheses"}


@router.post("/hypotheses/{hypothesis_id}/reproduce", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def reproduce(hypothesis_id: UUID) -> dict[str, str]:
    return {"todo": "spawn forge stub", "id": str(hypothesis_id)}
