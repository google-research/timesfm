from __future__ import annotations

from enum import StrEnum


class StepState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


TERMINAL: frozenset[StepState] = frozenset(
    {StepState.SUCCEEDED, StepState.FAILED, StepState.TIMED_OUT, StepState.CANCELLED}
)


_ALLOWED: dict[StepState, frozenset[StepState]] = {
    StepState.QUEUED: frozenset({StepState.RUNNING, StepState.CANCELLED}),
    StepState.RUNNING: frozenset(
        {StepState.SUCCEEDED, StepState.FAILED, StepState.TIMED_OUT, StepState.CANCELLED}
    ),
}


def can_transition(src: StepState, dst: StepState) -> bool:
    if src in TERMINAL:
        return False
    return dst in _ALLOWED.get(src, frozenset())
