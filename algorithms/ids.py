"""
Iterative Deepening Search (IDS)
==================================
Performs depth-limited DFS with increasing depth limits until the target
is found. Combines BFS's completeness with DFS's low memory footprint.

Depth metric : hop count (number of edges traversed), not cost.

Use case : memory-constrained environments; demonstrates the IDS trade-off
           (optimal by hop count, slower than BFS on large graphs due to
           repeated node expansions).
Optimal  : yes — for unweighted (hop-count) path length.
Complete : yes.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import networkx as nx

from algorithms.base import SearchResult, path_stats

# Safety cap: stop if no path found within this many hops.
# Dhaka's graph diameter is well under 500 hops for connected components.
MAX_DEPTH = 500


def _depth_limited_search(
    G:           nx.MultiDiGraph,
    node:        int,
    target:      int,
    path:        List[int],
    visited:     set,
    depth_limit: int,
    counter:     List[int],   # mutable int passed by reference via list
) -> Optional[List[int]]:
    """
    Recursive depth-limited DFS. Returns path if found, else None.
    counter[0] is incremented for every node expanded.
    """
    counter[0] += 1

    if node == target:
        return path

    if len(path) - 1 >= depth_limit:
        return None

    for neighbour in G.successors(node):
        if neighbour not in visited:
            visited.add(neighbour)
            result = _depth_limited_search(
                G, neighbour, target,
                path + [neighbour], visited,
                depth_limit, counter,
            )
            if result is not None:
                return result
            visited.discard(neighbour)   # backtrack

    return None


def run(
    G:         nx.MultiDiGraph,
    source:    int,
    target:    int,
    cost_fn,
    max_depth: int = MAX_DEPTH,
) -> SearchResult:
    """
    Run Iterative Deepening Search from source to target.

    Parameters
    ----------
    G         : enriched NetworkX MultiDiGraph
    source    : start node ID
    target    : destination node ID
    cost_fn   : CostFunction (used only for path_stats, not for search)
    max_depth : maximum hop depth before declaring failure

    Returns
    -------
    SearchResult
    """
    start_time = time.perf_counter()
    counter    = [0]   # total nodes expanded across all iterations

    if source == target:
        return SearchResult(
            algorithm="IDS", source=source, target=target,
            path=[source], nodes_explored=0,
            execution_time=0.0, path_edges=0,
            path_length_m=0.0, avg_safety=1.0,
            avg_gender_safety=1.0, found=True,
        )

    if source not in G or target not in G:
        return SearchResult(
            algorithm="IDS", source=source, target=target,
            found=False, execution_time=time.perf_counter() - start_time,
        )

    for depth_limit in range(1, max_depth + 1):
        visited = {source}
        path = _depth_limited_search(
            G, source, target, [source], visited, depth_limit, counter
        )
        if path is not None:
            stats = path_stats(G, path, cost_fn)
            return SearchResult(
                algorithm="IDS",
                source=source,
                target=target,
                path=path,
                nodes_explored=counter[0],
                execution_time=time.perf_counter() - start_time,
                found=True,
                **stats,
            )

    return SearchResult(
        algorithm="IDS", source=source, target=target,
        found=False, nodes_explored=counter[0],
        execution_time=time.perf_counter() - start_time,
    )
