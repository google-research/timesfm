"""Optional LLM refinement for hypotheses.

Off by default. When enabled, this module is the *only* component that may
egress to a remote LLM, and it must respect the per-job egress policy. Source
code is filtered through a redactor before egress (see redact_source).
"""

from __future__ import annotations

import re
from typing import Protocol


class LLMClient(Protocol):
    def refine(self, prompt: str) -> str: ...


def redact_source(text: str) -> str:
    text = re.sub(r"0x[0-9a-fA-F]{40}", "0x<address>", text)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


class NoopClient:
    def refine(self, prompt: str) -> str:  # noqa: ARG002
        return ""


def get_client(backend: str) -> LLMClient:
    if backend in {"none", "rule"}:
        return NoopClient()
    if backend == "anthropic":  # pragma: no cover — requires extras
        from anthropic import Anthropic

        client = Anthropic()

        class AnthropicClient:
            def refine(self, prompt: str) -> str:
                msg = client.messages.create(
                    model="claude-fable-5",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(getattr(b, "text", "") for b in msg.content)

        return AnthropicClient()
    raise ValueError(f"unknown backend: {backend}")
