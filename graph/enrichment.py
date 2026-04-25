"""
Module 2 — Edge Attribute Enrichment
======================================
Adds six cost-relevant attributes to every edge of the Dhaka graph.

Attribute          Type     Range    Source
-----------------  -------  -------  ----------------------------------------
distance           float    metres   OSMnx `length` (direct alias)
travel_time        float    seconds  free-flow: length / speed  (M1 baseline)
traffic_level      float    0–1      highway-type heuristic + stable noise
                                     (Module 3 overrides this dynamically)
road_condition     float    0–1      OSM `surface` tag → highway fallback
safety_score       float    0–1      highway type + speed + lanes + `lit` tag
gender_safety_score float   0–1      lighting + isolation + road type

All scores are static baselines derived from OSM tags only.
Module 3 (dynamic simulation) will update traffic_level, road_condition,
safety_score, and gender_safety_score at runtime.

Scores are deterministic: identical graph → identical scores across runs.
Per-edge variation uses a hash of the edge identity, not random().
"""

import os
import pickle
import hashlib
import logging
import statistics
from typing import Dict, Optional

import networkx as nx

# Ensure project root is on sys.path when this file is run directly
import sys as _sys, os as _os
_pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _pkg_root not in _sys.path:
    _sys.path.insert(0, _pkg_root)

from graph.map_loader import load_graph, DATA_DIR

logger = logging.getLogger(__name__)

ENRICHED_PKL = os.path.join(DATA_DIR, "dhaka_enriched.pkl")


# ──────────────────────────────────────────────────────────────────────────────
# Stable per-edge noise  (deterministic, no random())
# ──────────────────────────────────────────────────────────────────────────────

def _noise(u: int, v: int, k: int, seed: int, lo: float, hi: float) -> float:
    """Return a stable float in [lo, hi] derived from the edge identity."""
    digest = hashlib.md5(f"{u}_{v}_{k}_{seed}".encode()).hexdigest()
    ratio  = int(digest[:8], 16) / 0xFFFF_FFFF   # 0.0 – 1.0
    return lo + ratio * (hi - lo)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: normalise highway tag (OSMnx sometimes returns a list)
# ──────────────────────────────────────────────────────────────────────────────

def _highway(data: dict) -> str:
    hw = data.get("highway", "unclassified")
    return hw[0] if isinstance(hw, list) else str(hw)


# ──────────────────────────────────────────────────────────────────────────────
# 1. distance
# ──────────────────────────────────────────────────────────────────────────────

def _distance(data: dict) -> float:
    """Metres — taken directly from the OSMnx `length` field."""
    return float(data.get("length", 0.0))


# ──────────────────────────────────────────────────────────────────────────────
# 2. traffic_level  (static baseline; Module 3 updates this at runtime)
# ──────────────────────────────────────────────────────────────────────────────

# Dhaka calibration: primary/secondary corridors are chronically congested.
_TRAFFIC_BASE: Dict[str, float] = {
    "motorway":      0.28,
    "trunk":         0.50,
    "primary":       0.72,   # Mirpur Rd, Gulshan Ave — heavily jammed
    "secondary":     0.65,
    "tertiary":      0.55,
    "residential":   0.40,
    "unclassified":  0.45,
    "service":       0.35,
    "living_street": 0.30,
}
_TRAFFIC_DEFAULT = 0.50


def _traffic_level(data: dict, u: int, v: int, k: int) -> float:
    """
    Static congestion baseline for this edge (0 = free, 1 = gridlock).
    Noise ± 0.10 gives realistic variation between adjacent roads.
    """
    base  = _TRAFFIC_BASE.get(_highway(data), _TRAFFIC_DEFAULT)
    noise = _noise(u, v, k, seed=0, lo=-0.10, hi=0.10)
    return min(1.0, max(0.0, base + noise))


# ──────────────────────────────────────────────────────────────────────────────
# 3. road_condition
# ──────────────────────────────────────────────────────────────────────────────

# OSM surface tag → condition score (1 = perfect, 0 = impassable)
_SURFACE_SCORE: Dict[str, float] = {
    "asphalt":        0.90,
    "paved":          0.85,
    "concrete":       0.88,
    "cobblestone":    0.58,
    "sett":           0.60,
    "compacted":      0.55,
    "fine_gravel":    0.50,
    "unpaved":        0.40,
    "gravel":         0.42,
    "dirt":           0.30,
    "ground":         0.35,
    "mud":            0.18,
    "sand":           0.22,
}

# Fallback when surface tag is absent — Dhaka roads degrade faster than OSM
# type alone would suggest (waterlogging, heavy loads, poor maintenance).
_CONDITION_BASE: Dict[str, float] = {
    "motorway":      0.80,
    "trunk":         0.72,
    "primary":       0.65,
    "secondary":     0.60,
    "tertiary":      0.55,
    "residential":   0.48,
    "unclassified":  0.43,
    "service":       0.40,
    "living_street": 0.46,
}
_CONDITION_DEFAULT = 0.50


def _road_condition(data: dict, u: int, v: int, k: int) -> float:
    """
    Road surface quality (1 = perfect, 0 = impassable).
    Prefers OSM `surface` tag; falls back to highway-type heuristic.
    """
    surface = data.get("surface")
    if isinstance(surface, list):
        surface = surface[0]

    if surface and surface in _SURFACE_SCORE:
        base = _SURFACE_SCORE[surface]
    else:
        base = _CONDITION_BASE.get(_highway(data), _CONDITION_DEFAULT)

    noise = _noise(u, v, k, seed=1, lo=-0.08, hi=0.08)
    return min(1.0, max(0.05, base + noise))


# ──────────────────────────────────────────────────────────────────────────────
# 4. safety_score
# ──────────────────────────────────────────────────────────────────────────────

_SAFETY_BASE: Dict[str, float] = {
    "motorway":      0.82,   # controlled access, rare pedestrian conflict
    "trunk":         0.73,
    "primary":       0.66,
    "secondary":     0.63,
    "tertiary":      0.60,
    "residential":   0.68,   # low speed → safer
    "unclassified":  0.53,
    "service":       0.55,
    "living_street": 0.70,
}
_SAFETY_DEFAULT = 0.58


def _safety_score(data: dict, u: int, v: int, k: int) -> float:
    """
    General road safety (1 = safest, 0 = most dangerous).
    Penalises high speed limits and many lanes; rewards lighting.
    """
    base = _SAFETY_BASE.get(_highway(data), _SAFETY_DEFAULT)

    # Higher speed limit → greater injury risk
    speed = float(data.get("speed_kph", 40.0))
    speed_penalty = min(0.15, max(0.0, (speed - 30.0) / 200.0))

    # More lanes → more conflict points
    try:
        lanes_raw = data.get("lanes", 1)
        lanes = int(lanes_raw[0] if isinstance(lanes_raw, list) else lanes_raw)
    except (ValueError, TypeError):
        lanes = 1
    lane_penalty = min(0.10, (lanes - 1) * 0.025)

    # Street lighting reduces night-time risk
    lit = str(data.get("lit", "")).lower()
    light_bonus = 0.08 if lit in ("yes", "24/7") else 0.0

    noise = _noise(u, v, k, seed=2, lo=-0.06, hi=0.06)
    score = base - speed_penalty - lane_penalty + light_bonus + noise
    return min(1.0, max(0.05, score))


# ──────────────────────────────────────────────────────────────────────────────
# 5. gender_safety_score
# ──────────────────────────────────────────────────────────────────────────────

# Busy, observable roads score higher; isolated back-alleys score lower.
_GENDER_SAFETY_BASE: Dict[str, float] = {
    "motorway":      0.62,
    "trunk":         0.66,
    "primary":       0.74,   # high footfall, observable
    "secondary":     0.70,
    "tertiary":      0.66,
    "residential":   0.56,
    "unclassified":  0.46,
    "service":       0.40,   # back-service lanes — isolated
    "living_street": 0.58,
}
_GENDER_SAFETY_DEFAULT = 0.53


def _gender_safety_score(data: dict, u: int, v: int, k: int) -> float:
    """
    Safety score specifically for women travellers (1 = safest, 0 = least safe).
    Key factors: street lighting (strongest signal), road isolation,
    and general observability (footfall proxy via road type).
    Time-of-day adjustment is handled in Module 3 at runtime.
    """
    hw   = _highway(data)
    base = _GENDER_SAFETY_BASE.get(hw, _GENDER_SAFETY_DEFAULT)

    # Lighting tag
    lit = str(data.get("lit", "")).lower()
    if lit in ("yes", "24/7"):
        light_mod = +0.12
    elif lit == "no":
        light_mod = -0.15
    else:
        light_mod = 0.0   # unknown — Module 3 applies night penalty at runtime

    # Isolated road types (alleys, service lanes) are penalised
    isolation_penalty = 0.08 if hw in ("service", "unclassified") else 0.0

    noise = _noise(u, v, k, seed=3, lo=-0.05, hi=0.05)
    score = base + light_mod - isolation_penalty + noise
    return min(1.0, max(0.05, score))


# ──────────────────────────────────────────────────────────────────────────────
# Main enrichment pass
# ──────────────────────────────────────────────────────────────────────────────

def enrich_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Write all six attributes onto every edge of G in-place and return G.

    Attribute order matches the cost function signature in Module 4:
        distance, travel_time, traffic_level,
        road_condition, safety_score, gender_safety_score
    """
    logger.info("Enriching %d edges ...", G.number_of_edges())

    for u, v, k, data in G.edges(keys=True, data=True):
        data["distance"]             = _distance(data)
        data["traffic_level"]        = _traffic_level(data, u, v, k)
        data["road_condition"]       = _road_condition(data, u, v, k)
        data["safety_score"]         = _safety_score(data, u, v, k)
        data["gender_safety_score"]  = _gender_safety_score(data, u, v, k)
        # travel_time is already on the edge from Module 1 (free-flow baseline).
        # We keep it as-is here; Module 3 will recompute it with traffic applied.

    logger.info("Enrichment complete.")
    return G


# ──────────────────────────────────────────────────────────────────────────────
# Persist & load
# ──────────────────────────────────────────────────────────────────────────────

def save_enriched(G: nx.MultiDiGraph) -> None:
    """Save the enriched graph to data/dhaka_enriched.pkl."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ENRICHED_PKL, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Enriched graph saved → %s", ENRICHED_PKL)


def load_enriched() -> nx.MultiDiGraph:
    """
    Load the enriched graph from cache.
    Raises FileNotFoundError if enrich_and_save() has not been run yet.
    """
    if not os.path.exists(ENRICHED_PKL):
        raise FileNotFoundError(
            f"Enriched graph not found at '{ENRICHED_PKL}'.\n"
            "Run enrich_and_save() or 'python graph/enrichment.py' first."
        )
    with open(ENRICHED_PKL, "rb") as f:
        G = pickle.load(f)
    logger.info("Enriched graph loaded — nodes: %d  edges: %d",
                G.number_of_nodes(), G.number_of_edges())
    return G


def enrich_and_save(force_rebuild: bool = False) -> nx.MultiDiGraph:
    """
    Full pipeline: load raw graph → enrich → save → return.

    Parameters
    ----------
    force_rebuild : re-enrich even if dhaka_enriched.pkl already exists
    """
    if not force_rebuild and os.path.exists(ENRICHED_PKL):
        logger.info("Enriched cache found — skipping. Use force_rebuild=True to redo.")
        return load_enriched()

    G = load_graph()
    G = enrich_graph(G)
    save_enriched(G)
    return G


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────────────

ENRICHED_ATTRS = [
    "distance",
    "travel_time",
    "traffic_level",
    "road_condition",
    "safety_score",
    "gender_safety_score",
]


def enrichment_summary(G: nx.MultiDiGraph) -> None:
    """Print per-attribute statistics across all edges."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║              Edge Attribute Summary (Module 2)              ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"  {'Attribute':<24} {'Min':>8} {'Max':>8} {'Mean':>8} {'Stdev':>8}")
    print("  " + "─" * 56)

    for attr in ENRICHED_ATTRS:
        vals = [
            d[attr] for _, _, d in G.edges(data=True)
            if attr in d and d[attr] is not None
        ]
        if not vals:
            print(f"  {attr:<24}  (no data)")
            continue
        lo    = min(vals)
        hi    = max(vals)
        mean  = statistics.mean(vals)
        stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {attr:<24} {lo:>8.3f} {hi:>8.3f} {mean:>8.3f} {stdev:>8.3f}")

    print("╚══════════════════════════════════════════════════════════════╝")

    # Show all six attributes on one sample edge
    sample = next(
        ((u, v, d) for u, v, d in G.edges(data=True)
         if all(a in d for a in ENRICHED_ATTRS)),
        None,
    )
    if sample:
        u, v, d = sample
        print(f"\n  Sample edge  {u} → {v}")
        for attr in ENRICHED_ATTRS:
            print(f"    {attr:<24} {d[attr]:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=== Module 2 — Edge Attribute Enrichment ===\n")
    G = enrich_and_save(force_rebuild=False)
    enrichment_summary(G)
    print("\nModule 2 complete — enriched graph ready for Module 3.")
