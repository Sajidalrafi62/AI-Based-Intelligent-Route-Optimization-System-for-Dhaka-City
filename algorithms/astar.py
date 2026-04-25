"""
A* Search
==========
Expands the node with the lowest f(n) = g(n) + h(n), where:
    g(n) = actual cost from source to n  (via CostFunction)
    h(n) = heuristic estimate from n to target

With an admissible heuristic, A* finds the optimal path while exploring
fewer nodes than UCS on most real-world graphs.

Heuristics (all admissible for this cost function):
    euclidean   — straight-line distance lower bound (default)
    travel_time — free-flow time lower bound
    risk_aware  — distance + safety floor (tightest, most informed)

Use case : primary routing algorithm — optimal and significantly faster
           than UCS on large graphs thanks to the heuristic guidance.
Optimal  : yes (with admissible heuristic).
Complete : yes.
"""

from __future__ import annotations

import heapq
import time
from typing import Callable, Dict, Optional

import networkx as nx

from algorithms.base import SearchResult, path_stats, min_edge_cost
from algorithms.heuristics import get_heuristic


def run(
    G:         nx.MultiDiGraph,
    source:    int,
    target:    int,
    cost_fn,
    heuristic: str | Callable = "euclidean",
) -> SearchResult:
    """
    Run A* from source to target.

    Parameters
    ----------
    G          : enriched NetworkX MultiDiGraph
    source     : start node ID
    target     : destination node ID
    cost_fn    : CostFunction — used for g(n) edge costs
    heuristic  : 'euclidean' | 'travel_time' | 'risk_aware'
                 or a callable h(G, u, target, cost_fn) → float

    Returns
    -------
    SearchResult with optimal path (given the heuristic is admissible)
    """
    start_time = time.perf_counter()
    nodes_explored = 0

    h_fn = get_heuristic(heuristic) if isinstance(heuristic, str) else heuristic

    if source == target:
        return SearchResult(
            algorithm=f"A* ({heuristic})", source=source, target=target,
            path=[source], nodes_explored=0,
            execution_time=0.0, path_edges=0,
            path_length_m=0.0, avg_safety=1.0,
            avg_gender_safety=1.0, found=True,
        )

    if source not in G or target not in G:
        return SearchResult(
            algorithm=f"A* ({heuristic})", source=source, target=target,
            found=False, execution_time=time.perf_counter() - start_time,
        )

    # Priority queue: (f_cost, tie_break_counter, node, g_cost, path)
    counter   = 0
    h_source  = h_fn(G, source, target, cost_fn)
    frontier  = [(h_source, counter, source, 0.0, [source])]

    # Best known g(n) for each node
    best_g: Dict[int, float] = {source: 0.0}

    while frontier:
        f_cost, _, node, g_cost, path = heapq.heappop(frontier)
        nodes_explored += 1

        # Lazy deletion: skip outdated entries
        if g_cost > best_g.get(node, float("inf")):
            continue

        if node == target:
            stats = path_stats(G, path, cost_fn)
            return SearchResult(
                algorithm=f"A* ({heuristic})",
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
            new_g      = g_cost + edge_cost

            if new_g < best_g.get(neighbour, float("inf")):
                best_g[neighbour] = new_g
                h_val  = h_fn(G, neighbour, target, cost_fn)
                f_val  = new_g + h_val
                counter += 1
                heapq.heappush(
                    frontier,
                    (f_val, counter, neighbour, new_g, path + [neighbour]),
                )

    return SearchResult(
        algorithm=f"A* ({heuristic})", source=source, target=target,
        found=False, nodes_explored=nodes_explored,
        execution_time=time.perf_counter() - start_time,
    )
