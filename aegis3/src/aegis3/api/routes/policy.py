from __future__ import annotations

from fastapi import APIRouter, status

from aegis3.policy.main import load_policy

router = APIRouter(tags=["policy"])


@router.get("/policy")
def get_policy() -> dict:
    return load_policy()


@router.put("/policy", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def replace_policy(body: dict) -> dict[str, str]:
    return {"todo": "verify signature, persist"}


@router.get("/system/tools", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def list_tools() -> dict[str, str]:
    return {"todo": "report installed tool versions"}
