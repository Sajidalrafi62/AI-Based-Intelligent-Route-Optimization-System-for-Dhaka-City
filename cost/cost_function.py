"""
Module 4 — Cost Function
==========================
Computes a scalar edge cost from the six enriched attributes.

Formula (per the project spec)
--------------------------------
    cost = w1×travel_time  +  w2×distance     +  w3×traffic_level
         + w4×safety_risk  +  w5×gender_risk   +  w6×condition_penalty

where:
    travel_time  — normalised to [0, 1]  (higher = slower)
    distance     — normalised to [0, 1]  (higher = farther)
    traffic_level           already in [0, 1]  (higher = more congested)
    safety_risk      = 1 − safety_score        (higher = more dangerous)
    gender_risk      = 1 − gender_safety_score (higher = less safe for women)
    condition_penalty= 1 − road_condition      (higher = worse road)

Weights w1–w6 are fully configurable and recomputable at runtime.
Changing weights takes effect instantly on the next path query — no
need to re-enrich or re-simulate the graph.

Public API
----------
    CostWeights           — dataclass holding w1–w6
    CostFunction          — callable (u, v, data) → float;
                            drop-in for NetworkX weight= parameter
    PRESETS               — dict of named CostWeights configurations
    preset(name)          — return a ready-to-use CostFunction for a preset
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)

# ── Normalisation reference ranges (Dhaka urban road network estimates) ────────
# travel_time : 0 – 600 s  (10 min is an extreme urban edge)
# distance    : 0 – 2000 m (2 km is a very long inner-city segment)
_TIME_MAX = 600.0
_DIST_MAX = 2000.0


def _norm(value: float, lo: float, hi: float) -> float:
    """Min-max normalise to [0, 1]; clamps values outside the range."""
    if hi == lo:
        return 0.0
    return min(1.0, max(0.0, (value - lo) / (hi - lo)))


# ──────────────────────────────────────────────────────────────────────────────
# CostWeights
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CostWeights:
    """
    Relative importance of each cost component.

    Parameters (match spec w1–w6)
    ------------------------------
    w1_time           : penalty for slow travel (travel_time)
    w2_distance       : penalty for longer physical distance
    w3_traffic        : penalty for traffic congestion
    w4_safety         : penalty for general road danger (inverted safety_score)
    w5_gender_safety  : penalty for low women's safety (inverted gender_safety_score)
    w6_road_condition : penalty for poor road surface (inverted road_condition)

    Weights do NOT need to sum to 1; they are normalised internally.
    Set any weight to 0 to ignore that factor entirely.
    """
    w1_time:           float = 0.30
    w2_distance:       float = 0.20
    w3_traffic:        float = 0.20
    w4_safety:         float = 0.10
    w5_gender_safety:  float = 0.10
    w6_road_condition: float = 0.10

    def total(self) -> float:
        return (self.w1_time + self.w2_distance + self.w3_traffic
                + self.w4_safety + self.w5_gender_safety + self.w6_road_condition)

    def normalised(self) -> "CostWeights":
        """Return a copy with all weights scaled so they sum to 1.0."""
        t = self.total()
        if t == 0:
            raise ValueError("All weights are zero — cannot normalise.")
        s = 1.0 / t
        return CostWeights(
            w1_time           = self.w1_time           * s,
            w2_distance       = self.w2_distance       * s,
            w3_traffic        = self.w3_traffic        * s,
            w4_safety         = self.w4_safety         * s,
            w5_gender_safety  = self.w5_gender_safety  * s,
            w6_road_condition = self.w6_road_condition * s,
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "w1_time":           self.w1_time,
            "w2_distance":       self.w2_distance,
            "w3_traffic":        self.w3_traffic,
            "w4_safety":         self.w4_safety,
            "w5_gender_safety":  self.w5_gender_safety,
            "w6_road_condition": self.w6_road_condition,
        }

    def __repr__(self) -> str:
        w = self.normalised()
        return (f"CostWeights("
                f"time={w.w1_time:.2f}, dist={w.w2_distance:.2f}, "
                f"traffic={w.w3_traffic:.2f}, safety={w.w4_safety:.2f}, "
                f"gender={w.w5_gender_safety:.2f}, cond={w.w6_road_condition:.2f})")


# ──────────────────────────────────────────────────────────────────────────────
# Named presets
# ──────────────────────────────────────────────────────────────────────────────

PRESETS: Dict[str, CostWeights] = {
    # Fastest route — prioritise travel time and avoid traffic
    "fastest": CostWeights(
        w1_time=0.50, w2_distance=0.10, w3_traffic=0.30,
        w4_safety=0.05, w5_gender_safety=0.00, w6_road_condition=0.05,
    ),
    # Shortest physical path
    "shortest": CostWeights(
        w1_time=0.10, w2_distance=0.60, w3_traffic=0.10,
        w4_safety=0.10, w5_gender_safety=0.00, w6_road_condition=0.10,
    ),
    # Safest general route
    "safest": CostWeights(
        w1_time=0.15, w2_distance=0.10, w3_traffic=0.15,
        w4_safety=0.40, w5_gender_safety=0.10, w6_road_condition=0.10,
    ),
    # Optimised for women's safety — strongest gender_safety and lighting weight
    "women_safe": CostWeights(
        w1_time=0.10, w2_distance=0.05, w3_traffic=0.10,
        w4_safety=0.20, w5_gender_safety=0.45, w6_road_condition=0.10,
    ),
    # Balanced — equal emphasis on all factors
    "balanced": CostWeights(
        w1_time=0.20, w2_distance=0.15, w3_traffic=0.20,
        w4_safety=0.15, w5_gender_safety=0.15, w6_road_condition=0.15,
    ),
    # Comfort — smooth roads, low traffic, reasonable speed
    "comfort": CostWeights(
        w1_time=0.20, w2_distance=0.10, w3_traffic=0.25,
        w4_safety=0.10, w5_gender_safety=0.05, w6_road_condition=0.30,
    ),
}


def preset(name: str) -> "CostFunction":
    """
    Return a ready-to-use CostFunction for a named preset.

    Available presets: fastest, shortest, safest, women_safe, balanced, comfort

    Example
    -------
        cf = preset("women_safe")
        nx.shortest_path(G, source, target, weight=cf)
    """
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {list(PRESETS)}")
    return CostFunction(PRESETS[name])


# ──────────────────────────────────────────────────────────────────────────────
# CostFunction
# ──────────────────────────────────────────────────────────────────────────────

class CostFunction:
    """
    Callable edge-weight function compatible with NetworkX's weight= parameter.

    Usage
    -----
        cf = CostFunction(weights)
        cost = cf(u, v, edge_data)                      # scalar cost
        nx.shortest_path(G, src, dst, weight=cf)        # use in routing

    Updating weights at runtime
    ---------------------------
        cf.set_weights(CostWeights(w1_time=0.5, ...))   # instant, no rebuild needed

    Parameters
    ----------
    weights : CostWeights instance (or None for project defaults)
    """

    def __init__(self, weights: CostWeights = None):
        self._raw     = weights or CostWeights()
        self._weights = self._raw.normalised()

    def set_weights(self, weights: CostWeights) -> None:
        """
        Update weights at runtime.
        The next call to __call__ or breakdown() uses the new weights.
        No graph rebuild is required.
        """
        self._raw     = weights
        self._weights = weights.normalised()
        logger.debug("CostFunction weights updated: %s", self._weights)

    @property
    def weights(self) -> CostWeights:
        """Current normalised weights (read-only view)."""
        return self._weights

    # ── core computation ────────────────────────────────────────────────────

    def _components(self, data: dict) -> Dict[str, float]:
        """
        Extract and normalise all six cost components from edge data.
        Returns raw component values (before weighting).
        """
        return {
            "travel_time":     _norm(data.get("travel_time",        0.0), 0.0, _TIME_MAX),
            "distance":        _norm(data.get("distance",           0.0), 0.0, _DIST_MAX),
            "traffic":         float(data.get("traffic_level",      0.5)),
            "safety_risk":     1.0 - float(data.get("safety_score",        0.5)),
            "gender_risk":     1.0 - float(data.get("gender_safety_score", 0.5)),
            "condition_pen":   1.0 - float(data.get("road_condition",      0.5)),
        }

    def __call__(self, u: int, v: int, data: dict) -> float:
        """
        Compute scalar edge cost.

        Signature is compatible with NetworkX weight= callables:
            weight(u, v, edge_attr_dict) → float

        Returns a value in (0, 1] — never zero so no edge is treated as free.
        """
        w = self._weights
        c = self._components(data)

        cost = (
            w.w1_time           * c["travel_time"]  +
            w.w2_distance       * c["distance"]     +
            w.w3_traffic        * c["traffic"]      +
            w.w4_safety         * c["safety_risk"]  +
            w.w5_gender_safety  * c["gender_risk"]  +
            w.w6_road_condition * c["condition_pen"]
        )
        return max(cost, 1e-6)

    def breakdown(self, u: int, v: int, data: dict) -> Dict[str, float]:
        """
        Return the weighted contribution of each component for one edge.
        Useful for the Streamlit UI to show why a route was chosen.

        Returns
        -------
        dict with keys:
            travel_time_contrib, distance_contrib, traffic_contrib,
            safety_contrib, gender_safety_contrib, road_condition_contrib,
            total
        """
        w = self._weights
        c = self._components(data)

        contribs = {
            "travel_time_contrib":       round(w.w1_time           * c["travel_time"],  4),
            "distance_contrib":          round(w.w2_distance       * c["distance"],     4),
            "traffic_contrib":           round(w.w3_traffic        * c["traffic"],      4),
            "safety_contrib":            round(w.w4_safety         * c["safety_risk"],  4),
            "gender_safety_contrib":     round(w.w5_gender_safety  * c["gender_risk"],  4),
            "road_condition_contrib":    round(w.w6_road_condition * c["condition_pen"],4),
        }
        contribs["total"] = round(sum(contribs.values()), 4)
        return contribs

    def path_cost(self, G, path: list) -> float:
        """
        Sum the cost of every edge along a node-list path.

        Parameters
        ----------
        G    : NetworkX MultiDiGraph
        path : list of node IDs (as returned by nx.shortest_path)
        """
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            # MultiDiGraph may have parallel edges; take the minimum-cost key
            edge_costs = [
                self(u, v, data)
                for data in G[u][v].values()
            ]
            total += min(edge_costs)
        return total

    def __repr__(self) -> str:
        return f"CostFunction({self._weights})"


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test / demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    print("=== Module 4 — Cost Function ===\n")

    # Synthetic edge representing a busy primary road
    sample_edge = {
        "length":              450.0,
        "travel_time":         120.0,
        "distance":            450.0,
        "traffic_level":       0.75,
        "road_condition":      0.60,
        "safety_score":        0.65,
        "gender_safety_score": 0.55,
    }

    print(f"  Sample edge attributes:")
    for k, v in sample_edge.items():
        print(f"    {k:<24} {v}")

    print()

    for name, wts in PRESETS.items():
        cf   = CostFunction(wts)
        cost = cf(0, 1, sample_edge)
        bd   = cf.breakdown(0, 1, sample_edge)
        print(f"  Preset: {name:<12}  total cost = {cost:.4f}")
        for comp, val in bd.items():
            if comp != "total":
                print(f"    {comp:<30} {val:.4f}")
        print()

    # Demonstrate runtime weight update
    cf = CostFunction()
    print(f"  Default weights:      cost = {cf(0, 1, sample_edge):.4f}")
    cf.set_weights(CostWeights(w1_time=0.0, w2_distance=0.0, w3_traffic=0.0,
                                w4_safety=0.5, w5_gender_safety=0.5, w6_road_condition=0.0))
    print(f"  Safety-only weights:  cost = {cf(0, 1, sample_edge):.4f}")

    print("\nModule 4 complete.")
