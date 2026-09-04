from __future__ import annotations

import shutil
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from aegis3 import __version__
from aegis3.config import settings

app = typer.Typer(
    name="aegis",
    help="Aegis3 — local-first offensive security platform for smart contracts.",
    no_args_is_help=True,
    add_completion=False,
)
project_app = typer.Typer(help="Project management.")
source_app = typer.Typer(help="Source ingestion (repo, local path, address).")
findings_app = typer.Typer(help="Inspect and triage findings.")
hypo_app = typer.Typer(help="Generate and reproduce exploit hypotheses.")
graph_app = typer.Typer(help="Query the attack graph.")
tools_app = typer.Typer(help="Tool inventory and health checks.")
job_app = typer.Typer(help="Manage scan jobs.")

app.add_typer(project_app, name="project")
app.add_typer(source_app, name="source")
app.add_typer(findings_app, name="findings")
app.add_typer(hypo_app, name="hypo")
app.add_typer(graph_app, name="graph")
app.add_typer(tools_app, name="tools")
app.add_typer(job_app, name="job")

console = Console()

REQUIRED_TOOLS = ["slither", "forge", "anvil", "cast", "echidna", "medusa", "myth", "halmos", "solc"]


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"aegis {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False, "--version", help="Show version and exit.", callback=_version_callback, is_eager=True
    ),
) -> None:
    pass


@app.command()
def init() -> None:
    """First-time setup: write config dirs and default policy."""
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    settings.policy_path.parent.mkdir(parents=True, exist_ok=True)
    if not settings.policy_path.exists():
        settings.policy_path.write_text(
            '{"version": 1, "egress_policy": "deny", "egress_allowlist": []}\n'
        )
    console.print(f"[green]initialized[/green] artifacts={settings.artifacts_dir} policy={settings.policy_path}")


@app.command()
def ui() -> None:
    """Open the local UI (loopback)."""
    url = f"http://{settings.api_host}:{settings.api_port}"
    console.print(f"open: {url}")


@app.command()
def up() -> None:
    """Start API + orchestrator (TODO: wire to systemd/launchd)."""
    console.print("[yellow]not implemented[/yellow] — for now run `make api` and `make orchestrator` in two shells.")


@app.command()
def down() -> None:
    """Stop API + orchestrator."""
    console.print("[yellow]not implemented[/yellow]")


@app.command()
def prune(older_than: str = typer.Option("30d", "--older-than", help="e.g. 30d, 7d, 1d")) -> None:
    """Delete old run bundles."""
    console.print(f"[yellow]not implemented[/yellow] — would prune {settings.runs_dir} older than {older_than}")


# ---- project ---------------------------------------------------------------

@project_app.command("create")
def project_create(
    name: str = typer.Option(..., "--name"),
    scope: str = typer.Option(..., "--scope", help="Written authorization scope (mandatory)."),
    chain: int = typer.Option(1, "--chain"),
) -> None:
    """Create a project. The scope is a non-empty written attestation."""
    console.print(f"[yellow]stub[/yellow] would create project name={name!r} scope={scope!r} chain={chain}")


@project_app.command("list")
def project_list() -> None:
    console.print("[yellow]stub[/yellow] would list projects from db")


# ---- source ----------------------------------------------------------------

@source_app.command("add")
def source_add(
    project: str = typer.Option(..., "--project"),
    kind: str = typer.Option(..., "--kind", help="git|local|address|abi_bytecode"),
    uri: str = typer.Option(..., "--uri"),
    ref: str | None = typer.Option(None, "--ref"),
    rpc: str | None = typer.Option(None, "--rpc"),
) -> None:
    console.print(f"[yellow]stub[/yellow] add source project={project} kind={kind} uri={uri} ref={ref} rpc={rpc}")


# ---- compile ---------------------------------------------------------------

@app.command()
def compile(  # noqa: A001
    project: str = typer.Option(..., "--project"),
    framework: str | None = typer.Option(None, "--framework", help="foundry|hardhat|solc"),
) -> None:
    console.print(f"[yellow]stub[/yellow] compile project={project} framework={framework}")


# ---- scan ------------------------------------------------------------------

@app.command()
def scan(
    project: str = typer.Option(..., "--project"),
    tools: str = typer.Option("slither,foundry,echidna", "--tools", help="comma-sep list"),
    budget: str = typer.Option("30m", "--budget"),
    egress: str = typer.Option("deny", "--egress"),
) -> None:
    """Run a scan job (DAG of analyzers, sandboxed)."""
    console.print(f"[yellow]stub[/yellow] scan project={project} tools={tools} budget={budget} egress={egress}")


# ---- findings --------------------------------------------------------------

@findings_app.command("list")
def findings_list(
    project: str = typer.Option(..., "--project"),
    severity: str | None = typer.Option(None, "--severity"),
) -> None:
    console.print(f"[yellow]stub[/yellow] findings list project={project} severity={severity}")


@findings_app.command("show")
def findings_show(finding_id: str = typer.Argument(...)) -> None:
    console.print(f"[yellow]stub[/yellow] findings show id={finding_id}")


# ---- hypotheses ------------------------------------------------------------

@hypo_app.command("generate")
def hypo_generate(
    project: str = typer.Option(..., "--project"),
    backend: str | None = typer.Option(None, "--backend", help="rule|ollama|anthropic"),
) -> None:
    console.print(f"[yellow]stub[/yellow] hypo generate project={project} backend={backend or settings.hypo_backend}")


# ---- graph -----------------------------------------------------------------

@graph_app.command("paths")
def graph_paths(
    project: str = typer.Option(..., "--project"),
    src: str = typer.Option(..., "--from"),
    dst: str = typer.Option(..., "--to"),
    max_hops: int = typer.Option(4, "--max-hops"),
) -> None:
    console.print(f"[yellow]stub[/yellow] graph paths project={project} from={src} to={dst} hops={max_hops}")


# ---- jobs ------------------------------------------------------------------

@job_app.command("cancel")
def job_cancel(job_id: str = typer.Argument(...)) -> None:
    console.print(f"[yellow]stub[/yellow] cancel job {job_id}")


# ---- report ----------------------------------------------------------------

@app.command()
def report(
    project: str = typer.Option(..., "--project"),
    format_: str = typer.Option("md", "--format"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    console.print(f"[yellow]stub[/yellow] report project={project} format={format_} out={out}")


# ---- tools doctor ----------------------------------------------------------

@tools_app.command("doctor")
def tools_doctor() -> None:
    """Check that the EVM toolchain is installed and on PATH."""
    table = Table(title="Aegis3 toolchain")
    table.add_column("tool")
    table.add_column("found")
    table.add_column("path")
    missing = 0
    for tool in REQUIRED_TOOLS:
        path = shutil.which(tool)
        ok = path is not None
        missing += 0 if ok else 1
        table.add_row(tool, "[green]yes[/green]" if ok else "[red]no[/red]", path or "-")
    console.print(table)
    if missing:
        console.print(f"[red]{missing} tool(s) missing. See docs/OPERATIONS.md §2.[/red]")
        sys.exit(1)
