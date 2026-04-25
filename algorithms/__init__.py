"""
algorithms package — unified interface for all six search algorithms.

Usage
-----
    from algorithms import run_algorithm, run_all, ALGORITHMS

    result  = run_algorithm("astar", G, source, target, cost_fn)
    results = run_all(G, source, target, cost_fn)

Available algorithm keys
------------------------
    "bfs", "dfs", "ucs", "ids", "greedy", "astar"
"""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx

from algorithms import bfs, dfs, ucs, ids, greedy, astar
from algorithms.base import SearchResult
from algorithms.heuristics import HEURISTICS, get_heuristic

# Default heuristic used by informed algorithms when none is specified
DEFAULT_HEURISTIC = "euclidean"

# Registry: algorithm key → module
ALGORITHMS = {
    "bfs":    bfs,
    "dfs":    dfs,
    "ucs":    ucs,
    "ids":    ids,
    "greedy": greedy,
    "astar":  astar,
}


def run_algorithm(
    name:      str,
    G:         nx.MultiDiGraph,
    source:    int,
    target:    int,
    cost_fn,
    heuristic: str = DEFAULT_HEURISTIC,
) -> SearchResult:
    """
    Run a single algorithm by name.

    Parameters
    ----------
    name      : one of 'bfs', 'dfs', 'ucs', 'ids', 'greedy', 'astar'
    G         : enriched NetworkX MultiDiGraph
    source    : start node ID
    target    : destination node ID
    cost_fn   : CostFunction instance
    heuristic : heuristic name for greedy/astar ('euclidean' | 'travel_time' | 'risk_aware')

    Returns
    -------
    SearchResult
    """
    if name not in ALGORITHMS:
        raise KeyError(f"Unknown algorithm '{name}'. Available: {list(ALGORITHMS)}")

    module = ALGORITHMS[name]

    # Informed algorithms accept a heuristic parameter
    if name in ("greedy", "astar"):
        return module.run(G, source, target, cost_fn, heuristic=heuristic)

    return module.run(G, source, target, cost_fn)


def run_all(
    G:         nx.MultiDiGraph,
    source:    int,
    target:    int,
    cost_fn,
    heuristic: str = DEFAULT_HEURISTIC,
    names:     Optional[List[str]] = None,
) -> Dict[str, SearchResult]:
    """
    Run multiple algorithms and return all results.

    Parameters
    ----------
    names : subset of algorithm keys to run (runs all six if None)

    Returns
    -------
    Dict[algorithm_name, SearchResult]
    """
    keys = names or list(ALGORITHMS)
    return {
        name: run_algorithm(name, G, source, target, cost_fn, heuristic)
        for name in keys
    }
