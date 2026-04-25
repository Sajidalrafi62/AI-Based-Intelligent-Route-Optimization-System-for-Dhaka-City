"""
Module 1 — Map Data Extraction
================================
Responsibilities:
    - Download Dhaka's road network from OpenStreetMap via OSMnx
    - Convert to a NetworkX MultiDiGraph (nodes = intersections, edges = roads)
    - Persist the graph to data/ for reuse by all downstream modules
    - Provide a fast load_graph() used everywhere else in the project

Graph conventions
-----------------
    Node attributes : osmid, y (lat), x (lon), street_count
    Edge attributes : osmid, length (m), highway, name, oneway,
                      speed_kph, travel_time (s)   ← added here
    Edge key        : integer (MultiDiGraph parallel-edge index)
"""

import os
import pickle
import logging
from typing import Optional

import osmnx as ox
import networkx as nx

logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.dirname(_HERE)
DATA_DIR   = os.path.join(_ROOT, "data")
GRAPH_PKL  = os.path.join(DATA_DIR, "dhaka_graph.pkl")
GRAPH_GML  = os.path.join(DATA_DIR, "dhaka_graph.graphml")

# ── OSMnx settings ────────────────────────────────────────────────────────────
PLACE_NAME   = "Dhaka, Bangladesh"
NETWORK_TYPE = "drive"          # only drivable roads


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _download(place: str, network_type: str) -> nx.MultiDiGraph:
    """Fetch the road network from OpenStreetMap."""
    logger.info("Downloading road network for '%s' ...", place)
    G = ox.graph_from_place(place, network_type=network_type)
    logger.info("Raw graph — nodes: %d  edges: %d",
                G.number_of_nodes(), G.number_of_edges())
    return G


def _add_speeds_and_times(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Use OSMnx to impute missing speed limits and compute base travel times.
    These are the *free-flow* values; Module 3 will apply traffic adjustments.
    """
    G = ox.add_edge_speeds(G)       # adds speed_kph (from maxspeed tag or defaults)
    G = ox.add_edge_travel_times(G) # adds travel_time = length / speed
    return G


def _save(G: nx.MultiDiGraph) -> None:
    """Persist graph as both pickle (fast) and GraphML (portable)."""
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(GRAPH_PKL, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Pickle saved → %s", GRAPH_PKL)

    ox.save_graphml(G, filepath=GRAPH_GML)
    logger.info("GraphML saved → %s", GRAPH_GML)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_graph(
    place: str = PLACE_NAME,
    network_type: str = NETWORK_TYPE,
    force_rebuild: bool = False,
) -> nx.MultiDiGraph:
    """
    Download, enrich with speeds/times, save, and return the graph.

    Parameters
    ----------
    place         : OSM place name string passed to osmnx
    network_type  : 'drive' | 'walk' | 'bike' | 'all'
    force_rebuild : if True, re-download even when a cached file exists

    Returns
    -------
    nx.MultiDiGraph  with node attrs (x, y) and edge attrs
                     (length, speed_kph, travel_time, highway, ...)
    """
    if not force_rebuild and os.path.exists(GRAPH_PKL):
        logger.info("Cache found — skipping download. Use force_rebuild=True to refresh.")
        return load_graph()

    G = _download(place, network_type)
    G = _add_speeds_and_times(G)
    _save(G)
    return G


def load_graph() -> nx.MultiDiGraph:
    """
    Load the graph from the local pickle cache.
    Raises FileNotFoundError if build_graph() has not been run yet.
    """
    if not os.path.exists(GRAPH_PKL):
        raise FileNotFoundError(
            f"Graph cache not found at '{GRAPH_PKL}'.\n"
            "Run build_graph() or 'python graph/map_loader.py' first."
        )
    with open(GRAPH_PKL, "rb") as f:
        G = pickle.load(f)
    logger.info("Graph loaded from cache — nodes: %d  edges: %d",
                G.number_of_nodes(), G.number_of_edges())
    return G


def graph_summary(G: nx.MultiDiGraph) -> None:
    """Print a human-readable summary of the graph to stdout."""
    print("\n╔══════════════════════════════════════╗")
    print("║        Dhaka Graph — Summary         ║")
    print("╠══════════════════════════════════════╣")
    print(f"║  Nodes  (intersections) : {G.number_of_nodes():>8,} ║")
    print(f"║  Edges  (road segments) : {G.number_of_edges():>8,} ║")
    print(f"║  CRS                    : {'WGS84':>8}  ║")

    # Check which base attributes are present on edges
    sample = next(iter(G.edges(data=True)), (None, None, {}))
    present = [a for a in ("length", "speed_kph", "travel_time", "highway")
               if a in sample[2]]
    print(f"║  Edge attrs present     : {', '.join(present)}")

    print("╠══════════════════════════════════════╣")
    print("║  Sample edges                        ║")
    print("╚══════════════════════════════════════╝")

    for u, v, data in list(G.edges(data=True))[:3]:
        attrs = {k: (round(v, 2) if isinstance(v, float) else v)
                 for k, v in data.items()
                 if k in ("length", "speed_kph", "travel_time", "highway")}
        print(f"  {u} → {v}  {attrs}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point — run this file directly to extract and save the graph
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=== Module 1 — Map Data Extraction ===\n")
    G = build_graph(force_rebuild=False)
    graph_summary(G)
    print("\nModule 1 complete — graph ready for downstream modules.")
