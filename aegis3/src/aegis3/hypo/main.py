from __future__ import annotations

from dataclasses import asdict, dataclass, field

from aegis3.hypo.templates import TEMPLATES, Template
from aegis3.schema.findings import Finding


@dataclass
class Hypothesis:
    template_id: str
    title: str
    narrative: str
    preconditions: list[str]
    steps: list[str]
    impact: str
    owasp_sc_2026: str
    supporting_findings: list[str] = field(default_factory=list)


def _matches(t: Template, findings: list[Finding]) -> list[str]:
    """Return supporting finding ids that the template can rest on.

    Stub matcher: any finding whose detector or owasp tag overlaps the
    template's owasp_sc_2026 group is considered supporting.
    """
    out: list[str] = []
    for f in findings:
        if f.owasp_sc_2026 == t.owasp_sc_2026 or f.detector.startswith(t.id.split("-")[0]):
            out.append(f.detector)
    return out


def generate(findings: list[Finding], max_n: int = 10) -> list[Hypothesis]:
    candidates: list[Hypothesis] = []
    for t in TEMPLATES:
        support = _matches(t, findings)
        if not support:
            continue
        candidates.append(
            Hypothesis(
                template_id=t.id,
                title=t.title,
                narrative=(
                    f"An attacker can exploit {t.title.lower()} given the listed "
                    "preconditions, leading to: " + t.impact + "."
                ),
                preconditions=list(t.preconditions),
                steps=list(t.steps),
                impact=t.impact,
                owasp_sc_2026=t.owasp_sc_2026,
                supporting_findings=support,
            )
        )
    return candidates[:max_n]


def to_dicts(hs: list[Hypothesis]) -> list[dict]:
    return [asdict(h) for h in hs]
