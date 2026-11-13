"""Pre/post-step hooks — same idea as Claude Code's PreToolUse/PostToolUse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

HookOutcome = Literal["allow", "deny"]


@dataclass
class HookResult:
    outcome: HookOutcome
    reason: str = ""


def pre_step(step: dict, policy: dict) -> HookResult:
    tool = step.get("tool")
    if tool not in policy.get("tool_caps", {}):
        return HookResult("deny", f"tool {tool!r} not in policy.tool_caps")

    cap = policy["tool_caps"][tool]
    if step.get("timeout_s", 0) > cap["timeout_s"]:
        return HookResult("deny", f"timeout exceeds cap for {tool}")

    if step.get("egress_requested") and policy.get("egress_policy") == "deny":
        return HookResult("deny", "egress requested but policy is deny")

    for key, val in step.get("options", {}).items():
        if isinstance(val, str) and any(c in val for c in ";|&`$\n"):
            return HookResult("deny", f"shell metacharacter in option {key!r}")

    return HookResult("allow")


def post_step(artifact_meta: dict, policy: dict) -> HookResult:  # noqa: ARG001
    if artifact_meta.get("size_bytes", 0) > 2 * 1024 * 1024 * 1024:
        return HookResult("deny", "artifact > 2GiB")
    if not artifact_meta.get("sha256"):
        return HookResult("deny", "missing sha256")
    return HookResult("allow")
