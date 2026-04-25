"""
Breadth-First Search (BFS)
===========================
Explores the graph layer by layer (by hop count).
Finds the path with the fewest edges, ignoring edge weights entirely.

Use case : baseline comparison; useful when hop count matters more than cost.
Optimal  : yes — for unweighted (hop-count) path length.
Complete : yes — finds a path if one exists.
"""

from __future__ import annotations

import time
from collections import deque
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
    Run BFS from source to target.

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
            algorithm="BFS", source=source, target=target,
            path=[source], nodes_explored=0,
            execution_time=0.0, path_edges=0,
            path_length_m=0.0, avg_safety=1.0,
            avg_gender_safety=1.0, found=True,
        )

    if source not in G or target not in G:
        return SearchResult(
            algorithm="BFS", source=source, target=target,
            found=False, execution_time=time.perf_counter() - start_time,
        )

    # Queue entries: (current_node, path_so_far)
    queue   = deque([(source, [source])])
    visited = {source}

    while queue:
        node, path = queue.popleft()
        nodes_explored += 1

        for neighbour in G.successors(node):
            if neighbour in visited:
                continue
            new_path = path + [neighbour]

            if neighbour == target:
                stats = path_stats(G, new_path, cost_fn)
                return SearchResult(
                    algorithm="BFS",
                    source=source,
                    target=target,
                    path=new_path,
                    nodes_explored=nodes_explored,
                    execution_time=time.perf_counter() - start_time,
                    found=True,
                    **stats,
                )

            visited.add(neighbour)
            queue.append((neighbour, new_path))

    return SearchResult(
        algorithm="BFS", source=source, target=target,
        found=False, nodes_explored=nodes_explored,
        execution_time=time.perf_counter() - start_time,
    )
