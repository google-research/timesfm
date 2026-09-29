from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from aegis3.schema.findings import Finding

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape([]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_markdown(
    project_name: str,
    findings: list[Finding],
    hypotheses: list[dict],
    template: str = "bounty",
) -> str:
    env = _env()
    tpl = env.get_template(f"{template}.md.j2")
    by_severity: dict[str, list[Finding]] = {}
    for f in findings:
        by_severity.setdefault(f.severity, []).append(f)
    return tpl.render(
        project=project_name,
        findings=findings,
        by_severity=by_severity,
        hypotheses=hypotheses,
    )
