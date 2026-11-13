from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, status
from pydantic import BaseModel

router = APIRouter(tags=["graph"])


class GraphAnnotation(BaseModel):
    node_id: UUID | None = None
    edge_id: UUID | None = None
    attrs: dict


@router.get("/projects/{project_id}/graph", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def get_graph(project_id: UUID) -> dict[str, str]:
    return {"todo": "cytoscape json", "project_id": str(project_id)}


@router.get("/projects/{project_id}/graph/paths", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def get_paths(
    project_id: UUID, src: str, dst: str, max_hops: int = 4
) -> dict[str, str]:
    return {"todo": "paths", "from": src, "to": dst, "max_hops": str(max_hops)}


@router.post("/projects/{project_id}/graph/annotate", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def annotate(project_id: UUID, body: GraphAnnotation) -> dict[str, str]:
    return {"todo": "annotate", "project_id": str(project_id)}
