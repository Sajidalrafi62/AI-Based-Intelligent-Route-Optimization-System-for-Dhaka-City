"""
Depth-First Search (DFS)
=========================
Explores as deep as possible before backtracking.
Does NOT guarantee the shortest or lowest-cost path.

Use case : demonstrates why uninformed depth-first is ill-suited to
           large road networks — useful as a worst-case comparison.
Optimal  : no.
Complete : yes (with visited set to prevent cycles).
"""

from __future__ import annotations

import time
from typing import Optional

import networkx as nx

from algorithms.base import SearchResult, path_stats


def run(
    G:       nx.MultiDiGraph,
    source:  int,
    target:  int,
    cost_fn,
) -> SearchResult:
    """
    Run iterative DFS from source to target.

    Iterative (stack-based) implementation avoids Python recursion limits
    on large graphs.

    Parameters
    ----------
    G        : enriched NetworkX MultiDiGraph
    source   : start node ID
    target   : destination node ID
    cost_fn  : CostFunction instance (used only for path_stats, not for search)

    Returns
    -------
    SearchResult
    """
    start_time     = time.perf_counter()
    nodes_explored = 0

    if source == target:
        return SearchResult(
            algorithm="DFS", source=source, target=target,
            path=[source], nodes_explored=0,
            execution_time=0.0, path_edges=0,
            path_length_m=0.0, avg_safety=1.0,
            avg_gender_safety=1.0, found=True,
        )

    if source not in G or target not in G:
        return SearchResult(
            algorithm="DFS", source=source, target=target,
            found=False, execution_time=time.perf_counter() - start_time,
        )

    # Stack entries: (current_node, path_so_far)
    stack   = [(source, [source])]
    visited = set()

    while stack:
        node, path = stack.pop()

        if node in visited:
            continue
        visited.add(node)
        nodes_explored += 1

        if node == target:
            stats = path_stats(G, path, cost_fn)
            return SearchResult(
                algorithm="DFS",
                source=source,
                target=target,
                path=path,
                nodes_explored=nodes_explored,
                execution_time=time.perf_counter() - start_time,
                found=True,
                **stats,
            )

        for neighbour in G.successors(node):
            if neighbour not in visited:
                stack.append((neighbour, path + [neighbour]))

    return SearchResult(
        algorithm="DFS", source=source, target=target,
        found=False, nodes_explored=nodes_explored,
        execution_time=time.perf_counter() - start_time,
    )
