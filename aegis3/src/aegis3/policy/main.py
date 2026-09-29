"""Policy bundle: signed JSON loaded at startup."""

from __future__ import annotations

import json

from aegis3.config import settings

DEFAULT_POLICY: dict = {
    "version": 1,
    "egress_policy": "deny",
    "egress_allowlist": [],
    "tool_caps": {
        "slither": {"timeout_s": 600, "cpus": 2.0, "memory": "4g"},
        "mythril": {"timeout_s": 1200, "cpus": 2.0, "memory": "4g"},
        "foundry": {"timeout_s": 1200, "cpus": 4.0, "memory": "8g"},
        "echidna": {"timeout_s": 3600, "cpus": 4.0, "memory": "8g"},
        "medusa": {"timeout_s": 3600, "cpus": 4.0, "memory": "8g"},
        "halmos": {"timeout_s": 1800, "cpus": 4.0, "memory": "8g"},
    },
    "mainnet_attest_required": True,
}


def load_policy() -> dict:
    if not settings.policy_path.exists():
        return DEFAULT_POLICY
    try:
        return json.loads(settings.policy_path.read_text())
    except Exception:  # noqa: BLE001 — bad policy → fall back to safe defaults
        return DEFAULT_POLICY
