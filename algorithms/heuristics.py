"""
Pluggable heuristics for informed search (Greedy Best-First, A*).

All heuristics share the same signature:
    h(G, u, target, cost_fn) → float

The returned value estimates the remaining cost from node u to target.
For A* to find the optimal path, h must be admissible — it must never
overestimate the true remaining cost.

Admissibility argument
----------------------
The CostFunction's minimum possible edge cost for a unit-distance edge
comes purely from the w2_distance term (all other penalties can be zero).
Our lower bound therefore uses only the distance component of cost_fn,
which guarantees h(u) ≤ true_remaining_cost.

Heuristics available
--------------------
    euclidean       — straight-line distance lower bound (fastest to compute)
    travel_time     — time lower bound assuming free-flow at avg urban speed
    risk_aware      — distance bound + best-case safety floor (tighter estimate)
"""

from __future__ import annotations

import math
from typing import Callable, Dict

import networkx as nx

# Earth radius in metres (WGS84 mean)
_R = 6_371_000.0

# Normalisation constants — must match cost/cost_function.py
_DIST_MAX = 2000.0
_TIME_MAX  = 600.0

# Assumed average free-flow speed for Dhaka urban roads (km/h → m/s)
_AVG_SPEED_MS = 30.0 * 1000.0 / 3600.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance in metres between two WGS84 coordinates."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2.0 * _R * math.asin(math.sqrt(a))


def _node_coords(G: nx.MultiDiGraph, node: int):
    """Return (lat, lon) for a node."""
    data = G.nodes[node]
    return data.get("y", 0.0), data.get("x", 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Heuristic implementations
# ──────────────────────────────────────────────────────────────────────────────

def euclidean_heuristic(
    G: nx.MultiDiGraph, u: int, target: int, cost_fn
) -> float:
    """
    Lower bound on remaining cost using straight-line distance only.

    Uses the w2_distance weight from cost_fn to stay in the same units
    as the actual edge costs, ensuring admissibility.
    """
    lat_u, lon_u = _node_coords(G, u)
    lat_t, lon_t = _node_coords(G, target)
    dist   = haversine(lat_u, lon_u, lat_t, lon_t)
    dist_n = min(1.0, dist / _DIST_MAX)
    return cost_fn.weights.w2_distance * dist_n


def travel_time_heuristic(
    G: nx.MultiDiGraph, u: int, target: int, cost_fn
) -> float:
    """
    Lower bound on remaining cost using free-flow travel time estimate.

    Assumes best-case speed (30 km/h average for Dhaka urban roads) with
    zero traffic, giving a valid lower bound on the w1_time component.
    """
    lat_u, lon_u = _node_coords(G, u)
    lat_t, lon_t = _node_coords(G, target)
    dist      = haversine(lat_u, lon_u, lat_t, lon_t)
    free_flow = dist / _AVG_SPEED_MS           # seconds at best-case speed
    time_n    = min(1.0, free_flow / _TIME_MAX)
    return cost_fn.weights.w1_time * time_n


def risk_aware_heuristic(
    G: nx.MultiDiGraph, u: int, target: int, cost_fn
) -> float:
    """
    Tighter lower bound combining distance, time, and minimum safety floor.

    Adds a small constant representing the best-case safety penalty any
    path must incur (even the safest road has some non-zero risk).
    This makes the heuristic more informed while staying admissible.
    """
    lat_u, lon_u = _node_coords(G, u)
    lat_t, lon_t = _node_coords(G, target)
    dist   = haversine(lat_u, lon_u, lat_t, lon_t)

    dist_n    = min(1.0, dist / _DIST_MAX)
    time_n    = min(1.0, (dist / _AVG_SPEED_MS) / _TIME_MAX)

    # Best-case safety along any real road: safety_score ≈ 0.85 (motorway)
    # → safety_risk floor = 1 − 0.85 = 0.15
    safety_floor      = 0.15
    gender_floor      = 0.25   # even best roads have some gender risk
    condition_floor   = 0.10   # even best roads have some wear

    w = cost_fn.weights
    return (
        w.w1_time           * time_n        +
        w.w2_distance       * dist_n        +
        w.w4_safety         * safety_floor  +
        w.w5_gender_safety  * gender_floor  +
        w.w6_road_condition * condition_floor
    )


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

HEURISTICS: Dict[str, Callable] = {
    "euclidean":   euclidean_heuristic,
    "travel_time": travel_time_heuristic,
    "risk_aware":  risk_aware_heuristic,
}


def get_heuristic(name: str) -> Callable:
    """
    Return a heuristic function by name.

    Parameters
    ----------
    name : 'euclidean' | 'travel_time' | 'risk_aware'
    """
    if name not in HEURISTICS:
        raise KeyError(f"Unknown heuristic '{name}'. Available: {list(HEURISTICS)}")
    return HEURISTICS[name]
