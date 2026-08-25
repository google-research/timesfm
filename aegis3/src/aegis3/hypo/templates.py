"""Rule-based exploit hypothesis templates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    id: str
    title: str
    preconditions: tuple[str, ...]
    steps: tuple[str, ...]
    impact: str
    owasp_sc_2026: str


TEMPLATES: tuple[Template, ...] = (
    Template(
        id="REENT-CALL-VALUE",
        title="Cross-function reentrancy via low-level call",
        preconditions=(
            "external call before state mutation",
            "no reentrancy guard on entry function",
            "recipient is attacker-controlled",
        ),
        steps=(
            "deploy malicious receiver contract",
            "call vulnerable function, triggering call to receiver",
            "from receiver, re-enter sibling function before state update",
            "drain protocol-held asset",
        ),
        impact="loss of funds",
        owasp_sc_2026="SC01:2026",
    ),
    Template(
        id="ACCESS-CONTROL-MISSING",
        title="Missing access control on privileged function",
        preconditions=(
            "privileged function (mint/setOwner/upgrade)",
            "no onlyOwner / onlyRole modifier",
        ),
        steps=(
            "call privileged function from EOA",
            "assert state changed in attacker's favor",
        ),
        impact="full takeover or unbounded mint",
        owasp_sc_2026="SC02:2026",
    ),
    Template(
        id="ORACLE-PRICE-MANIPULATION",
        title="Spot-price oracle manipulation via flash loan",
        preconditions=(
            "price read from a single AMM pool",
            "no TWAP, no deviation cap",
            "flash-loan venue available for the pair",
        ),
        steps=(
            "borrow flash loan",
            "swap to skew the pool price",
            "trigger protocol action that reads the manipulated price",
            "swap back, repay loan",
        ),
        impact="theft of collateral or arbitrage drain",
        owasp_sc_2026="SC04:2026",
    ),
    Template(
        id="UPGRADE-HIJACK",
        title="Upgrade authority hijack",
        preconditions=(
            "EIP-1967 / UUPS proxy",
            "admin is EOA or unprotected multisig",
            "no timelock between upgrade and execution",
        ),
        steps=(
            "compromise admin key (out of band)",
            "upgrade implementation to attacker contract",
            "drain via the new implementation",
        ),
        impact="full protocol takeover",
        owasp_sc_2026="SC07:2026",
    ),
    Template(
        id="SIG-REPLAY",
        title="ECDSA signature replay across chains or epochs",
        preconditions=(
            "signed message lacks chainid or nonce",
            "signature accepted by a verifier",
        ),
        steps=(
            "capture a valid signed payload on one context",
            "replay on another chain / after the intended epoch",
        ),
        impact="duplicated state transitions, double-spend",
        owasp_sc_2026="SC05:2026",
    ),
    Template(
        id="DONATION-INFLATION",
        title="ERC-4626 first-deposit donation share inflation",
        preconditions=(
            "ERC-4626 vault with simple share = assets * supply / total",
            "no virtual shares / dead shares",
        ),
        steps=(
            "deposit 1 wei as first depositor",
            "donate large amount via direct transfer",
            "victim deposits, receives 0 shares due to truncation",
        ),
        impact="theft of victim's deposit",
        owasp_sc_2026="SC04:2026",
    ),
)
