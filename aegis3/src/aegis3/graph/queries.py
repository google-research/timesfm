"""Graph queries — path-finding from a role to an asset."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from aegis3.graph.builder import Edge


def shortest_paths(
    edges: Iterable[Edge],
    source: str,
    target: str,
    max_hops: int = 4,
) -> list[list[str]]:
    """BFS over edges, returning shortest label-paths from source to target."""
    adj: dict[str, list[str]] = {}
    for e in edges:
        adj.setdefault(e.src, []).append(e.dst)

    queue: deque[list[str]] = deque([[source]])
    found: list[list[str]] = []
    while queue:
        path = queue.popleft()
        if len(path) - 1 > max_hops:
            continue
        last = path[-1]
        if last == target and len(path) > 1:
            found.append(path)
            continue
        for nxt in adj.get(last, []):
            if nxt in path:
                continue
            queue.append([*path, nxt])
    return found
