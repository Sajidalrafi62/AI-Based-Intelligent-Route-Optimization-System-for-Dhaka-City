"""
Greedy Best-First Search
=========================
Always expands the node that appears closest to the target according to
the heuristic, with no regard for the cost already incurred.

Use case : fast approximate routing; good for real-time preview before
           running A* or UCS.
Optimal  : no — ignores path cost so far.
Complete : yes (with visited set).
"""

from __future__ import annotations

import heapq
import time
from typing import Callable, Optional

import networkx as nx

from algorithms.base import SearchResult, path_stats
from algorithms.heuristics import euclidean_heuristic, get_heuristic


def run(
    G:          nx.MultiDiGraph,
    source:     int,
    target:     int,
    cost_fn,
    heuristic:  str | Callable = "euclidean",
) -> SearchResult:
    """
    Run Greedy Best-First Search from source to target.

    Parameters
    ----------
    G          : enriched NetworkX MultiDiGraph
    source     : start node ID
    target     : destination node ID
    cost_fn    : CostFunction (used for path_stats and heuristic weight access)
    heuristic  : 'euclidean' | 'travel_time' | 'risk_aware'
                 or a callable h(G, u, target, cost_fn) → float

    Returns
    -------
    SearchResult (path may not be cost-optimal)
    """
    start_time = time.perf_counter()
    nodes_explored = 0

    h_fn = get_heuristic(heuristic) if isinstance(heuristic, str) else heuristic

    if source == target:
        return SearchResult(
            algorithm="Greedy", source=source, target=target,
            path=[source], nodes_explored=0,
            execution_time=0.0, path_edges=0,
            path_length_m=0.0, avg_safety=1.0,
            avg_gender_safety=1.0, found=True,
        )

    if source not in G or target not in G:
        return SearchResult(
            algorithm="Greedy", source=source, target=target,
            found=False, execution_time=time.perf_counter() - start_time,
        )

    # Priority queue: (h_value, tie_break_counter, node, path)
    counter  = 0
    h_start  = h_fn(G, source, target, cost_fn)
    frontier = [(h_start, counter, source, [source])]
    visited  = set()

    while frontier:
        h_val, _, node, path = heapq.heappop(frontier)

        if node in visited:
            continue
        visited.add(node)
        nodes_explored += 1

        if node == target:
            stats = path_stats(G, path, cost_fn)
            return SearchResult(
                algorithm="Greedy",
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
                h_val = h_fn(G, neighbour, target, cost_fn)
                counter += 1
                heapq.heappush(
                    frontier,
                    (h_val, counter, neighbour, path + [neighbour]),
                )

    return SearchResult(
        algorithm="Greedy", source=source, target=target,
        found=False, nodes_explored=nodes_explored,
        execution_time=time.perf_counter() - start_time,
    )
