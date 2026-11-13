"""Attack graph builder.

Inputs: Slither IR JSON, ABI, bytecode disasm.
Outputs: graph_nodes + graph_edges rows ready for insertion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

NodeKind = Literal[
    "contract", "role", "function", "asset", "external_dep", "upgrade_slot", "eoa"
]
EdgeRelation = Literal[
    "has_role", "can_call", "controls", "holds_asset", "depends_on",
    "can_upgrade", "delegatecalls", "reads", "writes", "mints", "burns",
]


@dataclass
class Node:
    kind: NodeKind
    label: str
    attrs: dict = field(default_factory=dict)


@dataclass
class Edge:
    src: str  # node label
    dst: str
    relation: EdgeRelation
    attrs: dict = field(default_factory=dict)


@dataclass
class Graph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)


def build(slither_ir: dict, abi: list[dict] | None = None) -> Graph:  # noqa: ARG001
    """TODO: real implementation pending Slither IR parsing.

    For now, return a deterministic stub from contracts found in the IR.
    """
    g = Graph()
    contracts = (slither_ir or {}).get("contracts", []) or []
    for c in contracts:
        g.nodes.append(Node("contract", c.get("name", "Unknown"), {"path": c.get("path")}))
    return g
