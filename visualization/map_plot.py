"""
Module 6 — Visualization
==========================
Renders Dhaka routes on an interactive Folium map.

Features
--------
- plot_route()      : single algorithm result on a clean map
- plot_comparison() : multiple algorithm results on one map with layer toggles
- Each algorithm has a fixed, distinct colour for consistency across all views
- Source / destination markers with popups
- Path polylines with tooltips (algorithm name + cost) and stat popups
- Optional attribute overlay: colour road segments by traffic, safety, etc.

Colour palette  (consistent across visualization and UI)
--------------------------------------------------------
    BFS     #2196F3  blue
    DFS     #F44336  red
    UCS     #4CAF50  green
    IDS     #FF9800  orange
    Greedy  #9C27B0  purple
    A*      #212121  dark grey / black
"""

from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional, Tuple

import folium
from folium import plugins
import networkx as nx

# Ensure project root is on sys.path when this file is run directly
import sys as _sys, os as _os
_pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _pkg_root not in _sys.path:
    _sys.path.insert(0, _pkg_root)

from algorithms.base import SearchResult

logger = logging.getLogger(__name__)

# ── Output directory ───────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(_HERE)
MAPS_DIR  = os.path.join(_ROOT, "data", "maps")

# ── Dhaka map defaults ─────────────────────────────────────────────────────────
DHAKA_CENTER = [23.7808, 90.4000]
DEFAULT_ZOOM = 13

# ── Algorithm colour palette ───────────────────────────────────────────────────
ALGO_COLORS: Dict[str, str] = {
    "bfs":    "#2196F3",   # blue
    "dfs":    "#F44336",   # red
    "ucs":    "#4CAF50",   # green
    "ids":    "#FF9800",   # orange
    "greedy": "#9C27B0",   # purple
    "astar":  "#212121",   # dark grey
}
_FALLBACK_COLOR = "#607D8B"   # blue-grey for unknown algorithms

# Attribute overlay colour scales (value 0→1 mapped to colour gradient)
_TRAFFIC_COLORS  = ["#00C853", "#FFD600", "#FF6D00", "#D50000"]  # green→red
_SAFETY_COLORS   = ["#D50000", "#FF6D00", "#FFD600", "#00C853"]  # red→green
_COND_COLORS     = ["#D50000", "#FF6D00", "#FFD600", "#00C853"]  # red→green


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _node_latlon(G: nx.MultiDiGraph, node: int) -> Tuple[float, float]:
    """Return (lat, lon) for a graph node. Folium expects (lat, lon)."""
    d = G.nodes[node]
    return float(d.get("y", 0.0)), float(d.get("x", 0.0))


def _path_coords(G: nx.MultiDiGraph, path: List[int]) -> List[Tuple[float, float]]:
    """Convert a list of node IDs to (lat, lon) pairs for Folium."""
    return [_node_latlon(G, n) for n in path]


def _algo_key(name: str) -> str:
    """Normalise algorithm name to a colour-palette key."""
    n = name.lower()
    for key in ALGO_COLORS:
        if key in n:
            return key
    return ""


def get_algorithm_color(name: str) -> str:
    """Return the hex colour for an algorithm name (or fallback)."""
    return ALGO_COLORS.get(_algo_key(name), _FALLBACK_COLOR)


def _base_map(center: List[float] = None, zoom: int = DEFAULT_ZOOM) -> folium.Map:
    """Create a Folium map centred on Dhaka with OpenStreetMap tiles."""
    m = folium.Map(
        location=center or DHAKA_CENTER,
        zoom_start=zoom,
        tiles="OpenStreetMap",
    )
    # Add a fullscreen button
    plugins.Fullscreen(position="topright").add_to(m)
    return m


def _path_popup(result: SearchResult) -> str:
    """Build an HTML popup body with route statistics."""
    status  = "Found" if result.found else "Not found"
    km      = result.path_length_m / 1000.0
    ms      = result.execution_time * 1000.0
    minutes = result.path_length_m / (30_000 / 60)   # assume 30 km/h
    return (
        f"<b>{result.algorithm}</b><br>"
        f"Status: {status}<br>"
        f"Hops: {result.path_edges}<br>"
        f"Distance: {km:.2f} km<br>"
        f"Est. travel: {minutes:.1f} min<br>"
        f"Path cost: {result.path_cost:.4f}<br>"
        f"Nodes explored: {result.nodes_explored:,}<br>"
        f"Calc. time: {ms:.1f} ms<br>"
        f"Avg safety: {result.avg_safety:.2f}<br>"
        f"Avg gender safety: {result.avg_gender_safety:.2f}"
    )


def _add_route_layer(
    feature_group: folium.FeatureGroup,
    G:             nx.MultiDiGraph,
    result:        SearchResult,
    color:         str,
    weight:        int = 5,
    opacity:       float = 0.85,
) -> None:
    """Draw a path polyline + source/destination markers onto a FeatureGroup."""
    if not result.found or not result.path:
        return

    coords = _path_coords(G, result.path)

    # Path polyline
    folium.PolyLine(
        locations=coords,
        color=color,
        weight=weight,
        opacity=opacity,
        tooltip=f"{result.algorithm} — cost {result.path_cost:.4f}",
        popup=folium.Popup(_path_popup(result), max_width=280),
    ).add_to(feature_group)

    # Animated dashes for visual distinction between overlapping routes
    plugins.AntPath(
        locations=coords,
        color=color,
        weight=weight - 2,
        opacity=0.5,
        delay=1200,
        dash_array=[10, 20],
    ).add_to(feature_group)


def _add_endpoint_markers(
    m:             folium.Map | folium.FeatureGroup,
    G:             nx.MultiDiGraph,
    source:        int,
    target:        int,
    source_name:   str = "Source",
    target_name:   str = "Destination",
) -> None:
    """Add distinct circle markers for source (green) and target (red)."""
    src_lat, src_lon = _node_latlon(G, source)
    tgt_lat, tgt_lon = _node_latlon(G, target)

    folium.CircleMarker(
        location=[src_lat, src_lon],
        radius=10,
        color="#1B5E20",
        fill=True,
        fill_color="#43A047",
        fill_opacity=0.95,
        popup=folium.Popup(f"<b>Start</b><br>{source_name}<br>Node {source}", max_width=200),
        tooltip=f"Start: {source_name}",
    ).add_to(m)

    folium.CircleMarker(
        location=[tgt_lat, tgt_lon],
        radius=10,
        color="#B71C1C",
        fill=True,
        fill_color="#E53935",
        fill_opacity=0.95,
        popup=folium.Popup(f"<b>End</b><br>{target_name}<br>Node {target}", max_width=200),
        tooltip=f"End: {target_name}",
    ).add_to(m)


def _gradient_color(value: float, palette: List[str]) -> str:
    """Interpolate hex colour for value in [0, 1] across a palette."""
    n   = len(palette) - 1
    idx = min(int(value * n), n - 1)
    return palette[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def plot_route(
    G:            nx.MultiDiGraph,
    result:       SearchResult,
    source_name:  str = "Source",
    target_name:  str = "Destination",
    zoom:         int = DEFAULT_ZOOM,
) -> folium.Map:
    """
    Plot a single algorithm result on an interactive Folium map.

    Parameters
    ----------
    G            : enriched NetworkX MultiDiGraph
    result       : SearchResult from any algorithm
    source_name  : display name for the start location (for popup)
    target_name  : display name for the end location (for popup)
    zoom         : initial zoom level

    Returns
    -------
    folium.Map  — call .save(path) or display in a Jupyter/Streamlit context
    """
    if result.path:
        src_lat, src_lon = _node_latlon(G, result.source)
        mid_idx  = len(result.path) // 2
        mid_lat, mid_lon = _node_latlon(G, result.path[mid_idx])
        center   = [(src_lat + mid_lat) / 2, (src_lon + mid_lon) / 2]
    else:
        center = DHAKA_CENTER

    m     = _base_map(center=center, zoom=zoom)
    color = get_algorithm_color(result.algorithm)

    fg = folium.FeatureGroup(name=result.algorithm).add_to(m)
    _add_route_layer(fg, G, result, color=color, weight=6)
    _add_endpoint_markers(m, G, result.source, result.target,
                          source_name, target_name)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def plot_comparison(
    G:            nx.MultiDiGraph,
    results:      Dict[str, SearchResult],
    source_name:  str = "Source",
    target_name:  str = "Destination",
    zoom:         int = DEFAULT_ZOOM,
) -> folium.Map:
    """
    Plot multiple algorithm results on a single map with layer toggles.

    Each algorithm appears as its own toggleable layer so users can
    show/hide individual routes for comparison.

    Parameters
    ----------
    G        : enriched NetworkX MultiDiGraph
    results  : dict of {algorithm_name: SearchResult}
    zoom     : initial zoom level

    Returns
    -------
    folium.Map
    """
    # Centre on the first found path's midpoint, fallback to Dhaka centre
    center = DHAKA_CENTER
    for res in results.values():
        if res.found and res.path:
            src_lat, src_lon = _node_latlon(G, res.source)
            mid_idx  = len(res.path) // 2
            mid_lat, mid_lon = _node_latlon(G, res.path[mid_idx])
            center = [(src_lat + mid_lat) / 2, (src_lon + mid_lon) / 2]
            break

    m = _base_map(center=center, zoom=zoom)

    found_results = {k: v for k, v in results.items() if v.found}
    not_found     = [k for k, v in results.items() if not v.found]

    if not_found:
        logger.warning("No path found for: %s", ", ".join(not_found))

    # One FeatureGroup per algorithm
    for name, result in found_results.items():
        color = get_algorithm_color(name)
        fg    = folium.FeatureGroup(
            name=f"{name.upper()} — cost {result.path_cost:.4f}",
            show=True,
        ).add_to(m)
        _add_route_layer(fg, G, result, color=color, weight=5)

    # Source/destination markers sit on top (not in a toggleable layer)
    first = next(iter(results.values()))
    _add_endpoint_markers(m, G, first.source, first.target,
                          source_name, target_name)

    # Legend
    legend_html = _build_legend(found_results)
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def plot_attribute_overlay(
    G:          nx.MultiDiGraph,
    attribute:  str = "traffic_level",
    sample_pct: float = 0.15,
    zoom:       int   = DEFAULT_ZOOM,
) -> folium.Map:
    """
    Colour-code road segments by a dynamic attribute (traffic, safety, etc.)
    to provide map context behind route visualizations.

    Parameters
    ----------
    attribute  : 'traffic_level' | 'road_condition' | 'safety_score' |
                 'gender_safety_score'
    sample_pct : fraction of edges to render (keep low for performance)

    Returns
    -------
    folium.Map
    """
    palette_map = {
        "traffic_level":       _TRAFFIC_COLORS,
        "road_condition":      _COND_COLORS,
        "safety_score":        _SAFETY_COLORS,
        "gender_safety_score": _SAFETY_COLORS,
    }
    palette = palette_map.get(attribute, _TRAFFIC_COLORS)

    m       = _base_map(zoom=zoom)
    edges   = list(G.edges(data=True))
    step    = max(1, int(1.0 / sample_pct))
    sampled = edges[::step]

    fg = folium.FeatureGroup(name=f"{attribute} overlay").add_to(m)

    for u, v, data in sampled:
        val   = float(data.get(attribute, 0.5))
        color = _gradient_color(val, palette)
        try:
            coords = [_node_latlon(G, u), _node_latlon(G, v)]
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=3,
                opacity=0.6,
                tooltip=f"{attribute}: {val:.2f}",
            ).add_to(fg)
        except KeyError:
            continue

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ──────────────────────────────────────────────────────────────────────────────
# Legend builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_legend(results: Dict[str, SearchResult]) -> str:
    rows = ""
    for name, res in results.items():
        color = get_algorithm_color(name)
        km    = res.path_length_m / 1000.0
        rows += (
            f"<tr>"
            f"<td><span style='background:{color};display:inline-block;"
            f"width:16px;height:16px;border-radius:3px;'></span></td>"
            f"<td style='padding:2px 6px'><b>{name.upper()}</b></td>"
            f"<td style='padding:2px 6px'>{km:.2f} km</td>"
            f"<td style='padding:2px 6px'>cost {res.path_cost:.4f}</td>"
            f"<td style='padding:2px 6px'>{res.nodes_explored:,} nodes</td>"
            f"</tr>"
        )
    return f"""
    <div style='position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px 14px;border-radius:8px;
                border:1px solid #ccc;font-size:12px;font-family:sans-serif;
                box-shadow:2px 2px 6px rgba(0,0,0,0.2);'>
      <b style='font-size:13px'>Route Comparison</b>
      <table style='margin-top:6px;border-collapse:collapse'>
        <tr style='color:#666;font-size:11px'>
          <th></th><th>Algorithm</th><th>Distance</th>
          <th>Cost</th><th>Explored</th>
        </tr>
        {rows}
      </table>
    </div>
    """


# ──────────────────────────────────────────────────────────────────────────────
# Save utility
# ──────────────────────────────────────────────────────────────────────────────

def save_map(m: folium.Map, filename: str = "route.html") -> str:
    """
    Save a Folium map to data/maps/<filename>.

    Returns the absolute file path.
    """
    os.makedirs(MAPS_DIR, exist_ok=True)
    filepath = os.path.join(MAPS_DIR, filename)
    m.save(filepath)
    logger.info("Map saved → %s", filepath)
    return filepath
