"""Smoke tests — verify the package imports and stubs respond as expected."""

from __future__ import annotations

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from aegis3.api.main import create_app
from aegis3.cli.main import app as cli_app
from aegis3.orchestrator.dag import plan
from aegis3.orchestrator.state import StepState, can_transition
from aegis3.policy.hooks import post_step, pre_step
from aegis3.policy.main import DEFAULT_POLICY
from aegis3.schema.findings import Finding, Location


def test_cli_help() -> None:
    result = CliRunner().invoke(cli_app, ["--help"])
    assert result.exit_code == 0
    assert "aegis" in result.stdout.lower()


def test_cli_version() -> None:
    result = CliRunner().invoke(cli_app, ["--version"])
    assert result.exit_code == 0
    assert "aegis" in result.stdout.lower()


def test_api_health() -> None:
    app = create_app()
    client = TestClient(app, base_url="http://127.0.0.1")
    r = client.get("/v1/system/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_rejects_non_loopback_host() -> None:
    app = create_app()
    client = TestClient(app, base_url="http://evil.example")
    r = client.get("/v1/system/health")
    assert r.status_code == 403


def test_dag_plan_orders_analyzers_after_compile() -> None:
    spec = {"tools": [{"tool": "slither"}, {"tool": "echidna"}]}
    steps = plan(spec)
    ids = [s.id for s in steps]
    assert ids[0] == "ingest"
    assert ids[1] == "compile"
    assert "analyze.slither" in ids
    assert "analyze.echidna" in ids
    assert ids[-1] == "report"
    normalize = next(s for s in steps if s.id == "normalize")
    assert set(normalize.depends_on) == {"analyze.slither", "analyze.echidna"}


def test_state_machine_blocks_terminal_transitions() -> None:
    assert can_transition(StepState.QUEUED, StepState.RUNNING)
    assert can_transition(StepState.RUNNING, StepState.SUCCEEDED)
    assert not can_transition(StepState.SUCCEEDED, StepState.RUNNING)
    assert not can_transition(StepState.FAILED, StepState.SUCCEEDED)


def test_finding_dedupe_key_is_stable() -> None:
    f1 = Finding(
        detector="AEG-REENT-001",
        source_tool="slither",
        source_rule="reentrancy-eth",
        severity="high",
        confidence="medium",
        title="reentrancy",
        description="…",
        locations=[Location(file="src/Vault.sol", line=42, contract="Vault")],
    )
    f2 = Finding(**f1.model_dump())
    assert f1.dedupe_key() == f2.dedupe_key()


def test_pre_step_hook_denies_unknown_tool() -> None:
    res = pre_step({"tool": "nmap", "timeout_s": 1}, DEFAULT_POLICY)
    assert res.outcome == "deny"


def test_pre_step_hook_denies_shell_metacharacter() -> None:
    res = pre_step(
        {"tool": "slither", "timeout_s": 60, "options": {"foo": "bar; rm -rf /"}},
        DEFAULT_POLICY,
    )
    assert res.outcome == "deny"


def test_post_step_hook_requires_sha256() -> None:
    assert post_step({"size_bytes": 1}, DEFAULT_POLICY).outcome == "deny"
    assert post_step({"size_bytes": 1, "sha256": "abc"}, DEFAULT_POLICY).outcome == "allow"
