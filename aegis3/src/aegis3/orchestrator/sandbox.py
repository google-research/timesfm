"""Docker sandbox wrapper for analyzer steps.

Defaults: --network=none, --cap-drop=ALL, --read-only, pids/mem/cpu caps,
custom seccomp profile. Egress, when allowed, is mediated through a sidecar
proxy — never enabled directly on the worker container.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SandboxOptions:
    image: str
    image_digest: str
    cmd: list[str]
    source_mount: Path
    artifacts_mount: Path
    cpus: float = 4.0
    memory: str = "8g"
    pids_limit: int = 256
    timeout_s: int = 600
    network_mode: str = "none"  # 'none' | 'sidecar'
    seccomp_profile: Path | None = None
    extra_env: dict[str, str] = field(default_factory=dict)


def build_docker_args(opts: SandboxOptions) -> list[str]:
    args: list[str] = [
        "docker", "run", "--rm",
        "--name", f"aegis3-step-{abs(hash(tuple(opts.cmd))) & 0xFFFF:04x}",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        f"--pids-limit={opts.pids_limit}",
        f"--memory={opts.memory}",
        f"--cpus={opts.cpus}",
        "--read-only",
        "--tmpfs", "/tmp:size=512m",
        "-v", f"{opts.source_mount}:/src:ro",
        "-v", f"{opts.artifacts_mount}:/out",
    ]
    if opts.network_mode == "none":
        args += ["--network", "none"]
    elif opts.network_mode == "sidecar":
        args += ["--network", "aegis3-egress"]
    if opts.seccomp_profile:
        args += ["--security-opt", f"seccomp={opts.seccomp_profile}"]
    for k, v in opts.extra_env.items():
        args += ["-e", f"{k}={v}"]
    args += [f"{opts.image}@{opts.image_digest}"]
    args += opts.cmd
    return args


def run(opts: SandboxOptions) -> subprocess.CompletedProcess[bytes]:
    args = build_docker_args(opts)
    return subprocess.run(args, capture_output=True, timeout=opts.timeout_s, check=False)


def render(opts: SandboxOptions) -> str:
    return " ".join(shlex.quote(a) for a in build_docker_args(opts))
