"""Normalized Aegis3 finding schema (canonical model)."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["info", "low", "medium", "high", "critical"]
Confidence = Literal["low", "medium", "high"]
FindingStatus = Literal["open", "triaged", "fp", "confirmed", "fixed"]


class Location(BaseModel):
    file: str | None = None
    line: int | None = None
    col: int | None = None
    bytecode_offset: int | None = None
    contract: str | None = None
    function: str | None = None


class Evidence(BaseModel):
    counter_example: dict | None = None
    witness_tx: str | None = None
    fuzz_seed: int | None = None
    coverage_uri: str | None = None
    notes: str | None = None


class Finding(BaseModel):
    """Canonical finding emitted by the normalizer."""

    detector: str = Field(..., description="Aegis canonical detector id, e.g. AEG-REENT-001")
    source_tool: Literal["slither", "foundry", "echidna", "medusa", "mythril", "halmos"]
    source_rule: str
    severity: Severity
    confidence: Confidence
    title: str
    description: str
    swc_id: str | None = None
    owasp_sc_2026: str | None = None
    cwe: str | None = None
    locations: list[Location] = Field(default_factory=list)
    evidence: Evidence | None = None
    status: FindingStatus = "open"

    def dedupe_key(self) -> bytes:
        """Stable sha256 over (detector, contract, locations canonical)."""
        canon = {
            "detector": self.detector,
            "locations": [
                {
                    "contract": loc.contract,
                    "file": loc.file,
                    "line": loc.line,
                }
                for loc in sorted(self.locations, key=lambda x: (x.file or "", x.line or 0))
            ],
        }
        payload = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).digest()
