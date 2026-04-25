"""
Uniform Cost Search (UCS)
==========================
Expands the frontier node with the lowest cumulative cost.
Equivalent to Dijkstra's algorithm with a general cost function.

Use case : optimal weighted routing — the gold standard for cost-based paths.
Optimal  : yes — guaranteed to find the minimum-cost path.
Complete : yes.
"""

from __future__ import annotations

import heapq
import time
from typing import Dict, Optional

import networkx as nx

from algorithms.base import SearchResult, path_stats, min_edge_cost


def run(
    G:       nx.MultiDiGraph,
    source:  int,
    target:  int,
    cost_fn,
) -> SearchResult:
    """
    Run Uniform Cost Search from source to target.

    Parameters
    ----------
    G        : enriched NetworkX MultiDiGraph
    source   : start node ID
    target   : destination node ID
    cost_fn  : CostFunction — called as cost_fn(u, v, edge_data) → float

    Returns
    -------
    SearchResult with optimal (minimum-cost) path
    """
    start_time     = time.perf_counter()
    nodes_explored = 0

    if source == target:
        return SearchResult(
            algorithm="UCS", source=source, target=target,
            path=[source], nodes_explored=0,
            execution_time=0.0, path_edges=0,
            path_length_m=0.0, avg_safety=1.0,
            avg_gender_safety=1.0, found=True,
        )

    if source not in G or target not in G:
        return SearchResult(
            algorithm="UCS", source=source, target=target,
            found=False, execution_time=time.perf_counter() - start_time,
        )

    # Priority queue: (cumulative_cost, tie_break_counter, node, path)
    counter  = 0
    frontier = [(0.0, counter, source, [source])]
    # Best known cost to reach each node
    best: Dict[int, float] = {source: 0.0}

    while frontier:
        g_cost, _, node, path = heapq.heappop(frontier)
        nodes_explored += 1

        # Lazy deletion: skip if we already found a cheaper path to this node
        if g_cost > best.get(node, float("inf")):
            continue

        if node == target:
            stats = path_stats(G, path, cost_fn)
            return SearchResult(
                algorithm="UCS",
                source=source,
                target=target,
                path=path,
                nodes_explored=nodes_explored,
                execution_time=time.perf_counter() - start_time,
                found=True,
                **stats,
            )

        for neighbour in G.successors(node):
            edge_cost  = min_edge_cost(G, node, neighbour, cost_fn)
            new_g_cost = g_cost + edge_cost

            if new_g_cost < best.get(neighbour, float("inf")):
                best[neighbour] = new_g_cost
                counter += 1
                heapq.heappush(
                    frontier,
                    (new_g_cost, counter, neighbour, path + [neighbour]),
                )

    return SearchResult(
        algorithm="UCS", source=source, target=target,
        found=False, nodes_explored=nodes_explored,
        execution_time=time.perf_counter() - start_time,
    )
