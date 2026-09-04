"""Convert a JobSpec into an execution DAG.

The DAG shape (from ARCHITECTURE.md §5.2):

    ingest -> compile -+-> slither   -+
                       +-> mythril   |
                       +-> foundry   +-> normalize -> graph_build -> hypo -> report
                       +-> echidna   |
                       +-> medusa    |
                       +-> halmos   -+
"""

from __future__ import annotations

from dataclasses import dataclass, field

ANALYZERS: tuple[str, ...] = ("slither", "mythril", "foundry", "echidna", "medusa", "halmos")


@dataclass(frozen=True)
class Step:
    id: str
    tool: str
    depends_on: tuple[str, ...] = ()
    options: dict = field(default_factory=dict)
    timeout_s: int = 600


def plan(job_spec: dict) -> list[Step]:
    enabled = {t["tool"]: t for t in job_spec.get("tools", [])}
    steps: list[Step] = [
        Step("ingest", "ingest"),
        Step("compile", "compile", depends_on=("ingest",)),
    ]
    analyzer_ids: list[str] = []
    for name in ANALYZERS:
        if name not in enabled:
            continue
        sid = f"analyze.{name}"
        analyzer_ids.append(sid)
        steps.append(
            Step(
                id=sid,
                tool=name,
                depends_on=("compile",),
                options=enabled[name].get("options", {}),
                timeout_s=enabled[name].get("timeout_s", 600),
            )
        )
    steps.append(Step("normalize", "normalize", depends_on=tuple(analyzer_ids)))
    steps.append(Step("graph_build", "graph_build", depends_on=("normalize",)))
    steps.append(Step("hypo", "hypo", depends_on=("graph_build",)))
    steps.append(Step("report", "report", depends_on=("hypo",)))
    return steps
