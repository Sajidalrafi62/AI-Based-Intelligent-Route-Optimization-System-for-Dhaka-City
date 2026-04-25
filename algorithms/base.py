"""
Shared types and utilities for all search algorithms.

SearchResult is the single return type every algorithm produces,
ensuring Module 7 (comparative analysis) can treat all results uniformly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import networkx as nx


@dataclass
class SearchResult:
    """
    Uniform output from every search algorithm.

    Attributes
    ----------
    algorithm       : algorithm name, e.g. 'A*'
    source          : start node ID
    target          : destination node ID
    path            : ordered list of node IDs (empty if not found)
    path_cost       : total weighted cost along the path (using CostFunction)
    nodes_explored  : nodes popped from the frontier during search
    execution_time  : wall-clock seconds
    path_edges      : number of edges (hops) in the path
    path_length_m   : total physical distance in metres
    avg_safety      : mean safety_score across path edges (0–1)
    avg_gender_safety: mean gender_safety_score across path edges (0–1)
    found           : False if no path exists between source and target
    """
    algorithm:         str
    source:            int
    target:            int
    path:              List[int]      = field(default_factory=list)
    path_cost:         float         = 0.0
    nodes_explored:    int           = 0
    execution_time:    float         = 0.0
    path_edges:        int           = 0
    path_length_m:     float         = 0.0
    avg_safety:        float         = 0.0
    avg_gender_safety: float         = 0.0
    found:             bool          = False

    def summary(self) -> str:
        if not self.found:
            return (f"[{self.algorithm}]  No path found  "
                    f"({self.nodes_explored} nodes explored, "
                    f"{self.execution_time*1000:.1f} ms)")
        return (
            f"[{self.algorithm}]  "
            f"hops={self.path_edges}  "
            f"dist={self.path_length_m:.0f}m  "
            f"cost={self.path_cost:.4f}  "
            f"explored={self.nodes_explored}  "
            f"time={self.execution_time*1000:.1f}ms  "
            f"safety={self.avg_safety:.2f}  "
            f"gender_safety={self.avg_gender_safety:.2f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Edge helpers for MultiDiGraph
# ──────────────────────────────────────────────────────────────────────────────

def min_edge_cost(G: nx.MultiDiGraph, u: int, v: int, cost_fn) -> float:
    """
    Return the minimum cost among all parallel edges from u to v.
    MultiDiGraph can have multiple edges between the same pair of nodes.
    """
    return min(cost_fn(u, v, data) for data in G[u][v].values())


def min_edge_data(G: nx.MultiDiGraph, u: int, v: int, cost_fn) -> dict:
    """Return the edge data dict of the cheapest parallel edge from u to v."""
    return min(G[u][v].values(), key=lambda data: cost_fn(u, v, data))


# ──────────────────────────────────────────────────────────────────────────────
# Path statistics
# ──────────────────────────────────────────────────────────────────────────────

def path_stats(G: nx.MultiDiGraph, path: List[int], cost_fn) -> dict:
    """
    Compute aggregate statistics for a node-list path.

    Returns
    -------
    dict with keys: path_cost, path_edges, path_length_m,
                    avg_safety, avg_gender_safety
    """
    if len(path) < 2:
        return {
            "path_cost":         0.0,
            "path_edges":        0,
            "path_length_m":     0.0,
            "avg_safety":        0.0,
            "avg_gender_safety": 0.0,
        }

    total_cost   = 0.0
    total_dist   = 0.0
    safety_vals  = []
    gender_vals  = []

    for u, v in zip(path[:-1], path[1:]):
        data          = min_edge_data(G, u, v, cost_fn)
        total_cost   += cost_fn(u, v, data)
        total_dist   += float(data.get("distance", data.get("length", 0.0)))
        safety_vals.append(float(data.get("safety_score",        0.5)))
        gender_vals.append(float(data.get("gender_safety_score", 0.5)))

    n = len(safety_vals)
    return {
        "path_cost":         total_cost,
        "path_edges":        len(path) - 1,
        "path_length_m":     total_dist,
        "avg_safety":        sum(safety_vals) / n if n else 0.0,
        "avg_gender_safety": sum(gender_vals)  / n if n else 0.0,
    }
