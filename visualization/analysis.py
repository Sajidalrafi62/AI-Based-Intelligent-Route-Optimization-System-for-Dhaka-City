"""
Module 7 — Comparative Analysis
=================================
Records, charts, and tabulates performance metrics across all algorithms.

Metrics tracked (all drawn from SearchResult)
----------------------------------------------
    path_cost        weighted cost (lower = better)
    execution_time   wall-clock seconds (lower = better)
    nodes_explored   frontier pops (lower = more efficient)
    path_length_m    physical distance in metres (lower = shorter)
    avg_safety       mean safety_score along path (higher = safer)
    avg_gender_safety mean gender_safety_score (higher = safer)

Outputs
-------
    bar_charts  — 2×3 subplot grid, one metric per panel
    radar_chart — spider/radar chart normalised so "bigger = better" on every axis
    summary CSV — one row per algorithm, all metrics
    print_summary() — formatted table to stdout

Colour palette is shared with visualization/map_plot.py so charts and
the map legend use the same algorithm colours.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Ensure project root is on sys.path when this file is run directly
import sys as _sys, os as _os
_pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _pkg_root not in _sys.path:
    _sys.path.insert(0, _pkg_root)

from algorithms.base import SearchResult
from visualization.map_plot import ALGO_COLORS, get_algorithm_color

logger = logging.getLogger(__name__)

# ── Output directory ───────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.dirname(_HERE)
REPORTS_DIR  = os.path.join(_ROOT, "data", "reports")

# Metrics shown in bar-chart grid (label, field, unit, lower_is_better)
_BAR_METRICS = [
    ("Path Cost",          "path_cost",         "",    True),
    ("Execution Time",     "execution_time_ms", "ms",  True),
    ("Nodes Explored",     "nodes_explored",    "",    True),
    ("Path Length",        "path_length_km",    "km",  True),
    ("Avg Safety Score",   "avg_safety",        "",    False),
    ("Avg Gender Safety",  "avg_gender_safety", "",    False),
]

# Radar axes (label, field, higher_is_better)
_RADAR_AXES = [
    ("Cost\nefficiency",     "path_cost",         False),
    ("Speed",                "execution_time_ms", False),
    ("Exploration\neff.",    "nodes_explored",    False),
    ("Route\nlength",        "path_length_km",    False),
    ("Safety",               "avg_safety",        True),
    ("Gender\nsafety",       "avg_gender_safety", True),
]


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────────────

def to_dataframe(results: Dict[str, SearchResult]) -> pd.DataFrame:
    """
    Convert a dict of {name: SearchResult} to a tidy pandas DataFrame.
    Algorithms that did not find a path have NaN for path metrics.
    """
    rows = []
    for name, r in results.items():
        row = {
            "algorithm":         name,
            "found":             r.found,
            "path_cost":         r.path_cost         if r.found else float("nan"),
            "execution_time_ms": r.execution_time * 1000,
            "nodes_explored":    r.nodes_explored,
            "path_edges":        r.path_edges         if r.found else float("nan"),
            "path_length_km":    r.path_length_m / 1000 if r.found else float("nan"),
            "avg_safety":        r.avg_safety          if r.found else float("nan"),
            "avg_gender_safety": r.avg_gender_safety   if r.found else float("nan"),
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("algorithm")


def print_summary(results: Dict[str, SearchResult]) -> None:
    """Print a formatted comparison table to stdout."""
    df = to_dataframe(results)

    print("\n╔══════════════════════════════════════════════════════════════════════════╗")
    print("║                     Algorithm Comparison Summary                        ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print(f"\n  {'Algorithm':<16} {'Found':<6} {'Cost':>8} {'Time(ms)':>9} "
          f"{'Nodes':>8} {'Dist(km)':>9} {'Safety':>7} {'GndSfty':>8}")
    print("  " + "─" * 74)

    for name, r in results.items():
        if r.found:
            print(
                f"  {name:<16} {'✓':<6} {r.path_cost:>8.4f} "
                f"{r.execution_time*1000:>9.1f} {r.nodes_explored:>8,} "
                f"{r.path_length_m/1000:>9.3f} {r.avg_safety:>7.3f} "
                f"{r.avg_gender_safety:>8.3f}"
            )
        else:
            print(f"  {name:<16} {'✗':<6} {'—':>8} "
                  f"{r.execution_time*1000:>9.1f} {r.nodes_explored:>8,} "
                  f"{'—':>9} {'—':>7} {'—':>8}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Bar chart grid
# ──────────────────────────────────────────────────────────────────────────────

def plot_bar_charts(
    results: Dict[str, SearchResult],
    title:   str = "Algorithm Performance Comparison",
) -> plt.Figure:
    """
    2 × 3 subplot grid — one bar chart per metric.

    Parameters
    ----------
    results : dict of {algorithm_name: SearchResult}
    title   : overall figure title

    Returns
    -------
    matplotlib.figure.Figure
    """
    df     = to_dataframe(results)
    names  = list(df.index)
    colors = [get_algorithm_color(n) for n in names]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)

    axes_flat = axes.flatten()

    for ax, (label, field, unit, lower_is_better) in zip(axes_flat, _BAR_METRICS):
        vals = df[field].tolist()

        bars = ax.bar(names, vals, color=colors, edgecolor="white",
                      linewidth=0.8, zorder=3)
        ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(f"{label} ({unit})" if unit else label, fontsize=9)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        # Annotate bars with values
        for bar, val in zip(bars, vals):
            if not math.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.3g}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                )

        # Highlight best bar
        valid = [(i, v) for i, v in enumerate(vals) if not math.isnan(v)]
        if valid:
            best_idx = min(valid, key=lambda x: x[1])[0] if lower_is_better \
                       else max(valid, key=lambda x: x[1])[0]
            bars[best_idx].set_edgecolor("#FFD700")
            bars[best_idx].set_linewidth(2.5)

        # Mark "not found" bars with a pattern
        for bar, val in zip(bars, vals):
            if math.isnan(val):
                bar.set_hatch("///")
                bar.set_alpha(0.3)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Radar / Spider chart
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_radar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise each axis to [0, 1] where 1 = best.
    For "lower is better" metrics, invert after normalisation.
    """
    norm_df = pd.DataFrame(index=df.index)
    for label, field, higher_is_better in _RADAR_AXES:
        col   = df[field].copy()
        lo, hi = col.min(), col.max()
        if hi > lo:
            col = (col - lo) / (hi - lo)
        else:
            col = col * 0 + 0.5          # all equal: put at midpoint
        if not higher_is_better:
            col = 1.0 - col              # invert so bigger = better on radar
        norm_df[label] = col
    return norm_df


def plot_radar(
    results: Dict[str, SearchResult],
    title:   str = "Algorithm Radar Comparison",
) -> plt.Figure:
    """
    Radar / spider chart — all six metrics on a single normalised plot.
    Bigger area = better overall performer.

    Parameters
    ----------
    results : dict of {algorithm_name: SearchResult}

    Returns
    -------
    matplotlib.figure.Figure
    """
    df      = to_dataframe(results)
    norm_df = _normalise_radar(df)
    labels  = [ax[0] for ax in _RADAR_AXES]
    n_axes  = len(labels)

    angles  = [2 * math.pi * i / n_axes for i in range(n_axes)]
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8),
                            subplot_kw={"projection": "polar"})
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="grey")
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    legend_patches = []
    for name, row in norm_df.iterrows():
        values = row.tolist() + row.tolist()[:1]
        color  = get_algorithm_color(name)
        ax.plot(angles, values, color=color, linewidth=2, linestyle="solid")
        ax.fill(angles, values, color=color, alpha=0.12)
        legend_patches.append(
            mpatches.Patch(color=color, label=name.upper(), alpha=0.7)
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    ax.legend(handles=legend_patches, loc="upper right",
              bbox_to_anchor=(1.3, 1.1), fontsize=9)

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Execution-time line chart  (useful when same pair is run multiple times)
# ──────────────────────────────────────────────────────────────────────────────

def plot_time_bars(
    results: Dict[str, SearchResult],
) -> plt.Figure:
    """
    Horizontal bar chart of execution times — makes small differences visible.
    """
    df     = to_dataframe(results).sort_values("execution_time_ms", ascending=True)
    names  = list(df.index)
    times  = df["execution_time_ms"].tolist()
    colors = [get_algorithm_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(names, times, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Execution Time (ms)", fontsize=10)
    ax.set_title("Execution Time Comparison", fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, times):
        ax.text(val + max(times) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f} ms", va="center", fontsize=9)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Save full report
# ──────────────────────────────────────────────────────────────────────────────

def save_report(
    results:    Dict[str, SearchResult],
    output_dir: str = REPORTS_DIR,
    prefix:     str = "report",
) -> Dict[str, str]:
    """
    Save all charts and the CSV summary to output_dir.

    Parameters
    ----------
    output_dir : directory to write files into
    prefix     : filename prefix (e.g. 'shahbagh_to_gulshan')

    Returns
    -------
    dict mapping artifact type → absolute file path
    """
    os.makedirs(output_dir, exist_ok=True)
    saved: Dict[str, str] = {}

    # CSV
    csv_path = os.path.join(output_dir, f"{prefix}_summary.csv")
    to_dataframe(results).to_csv(csv_path)
    saved["csv"] = csv_path
    logger.info("CSV saved → %s", csv_path)

    # Bar charts
    fig_bars = plot_bar_charts(results)
    bar_path = os.path.join(output_dir, f"{prefix}_bar_charts.png")
    fig_bars.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig_bars)
    saved["bar_charts"] = bar_path
    logger.info("Bar charts saved → %s", bar_path)

    # Radar chart
    fig_radar = plot_radar(results)
    radar_path = os.path.join(output_dir, f"{prefix}_radar.png")
    fig_radar.savefig(radar_path, dpi=150, bbox_inches="tight")
    plt.close(fig_radar)
    saved["radar"] = radar_path
    logger.info("Radar chart saved → %s", radar_path)

    # Time bars
    fig_time = plot_time_bars(results)
    time_path = os.path.join(output_dir, f"{prefix}_time.png")
    fig_time.savefig(time_path, dpi=150, bbox_inches="tight")
    plt.close(fig_time)
    saved["time_bars"] = time_path
    logger.info("Time chart saved → %s", time_path)

    return saved


# ──────────────────────────────────────────────────────────────────────────────
# Entry point — demo with synthetic results
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    print("=== Module 7 — Comparative Analysis (demo with mock data) ===\n")

    # Synthetic results to demonstrate charts without needing a live graph
    from algorithms.base import SearchResult

    mock: Dict[str, SearchResult] = {
        "bfs": SearchResult(
            algorithm="BFS", source=0, target=1, found=True,
            path=[0, 1], path_cost=0.42, nodes_explored=1850,
            execution_time=0.31, path_edges=18, path_length_m=4200,
            avg_safety=0.63, avg_gender_safety=0.58,
        ),
        "dfs": SearchResult(
            algorithm="DFS", source=0, target=1, found=True,
            path=[0, 1], path_cost=0.71, nodes_explored=3200,
            execution_time=0.45, path_edges=34, path_length_m=8100,
            avg_safety=0.54, avg_gender_safety=0.50,
        ),
        "ucs": SearchResult(
            algorithm="UCS", source=0, target=1, found=True,
            path=[0, 1], path_cost=0.31, nodes_explored=4100,
            execution_time=0.62, path_edges=15, path_length_m=3700,
            avg_safety=0.70, avg_gender_safety=0.65,
        ),
        "ids": SearchResult(
            algorithm="IDS", source=0, target=1, found=True,
            path=[0, 1], path_cost=0.43, nodes_explored=5800,
            execution_time=1.10, path_edges=18, path_length_m=4200,
            avg_safety=0.63, avg_gender_safety=0.58,
        ),
        "greedy": SearchResult(
            algorithm="Greedy", source=0, target=1, found=True,
            path=[0, 1], path_cost=0.38, nodes_explored=620,
            execution_time=0.08, path_edges=16, path_length_m=3900,
            avg_safety=0.67, avg_gender_safety=0.61,
        ),
        "astar": SearchResult(
            algorithm="A* (euclidean)", source=0, target=1, found=True,
            path=[0, 1], path_cost=0.31, nodes_explored=890,
            execution_time=0.12, path_edges=15, path_length_m=3700,
            avg_safety=0.70, avg_gender_safety=0.65,
        ),
    }

    print_summary(mock)
    paths = save_report(mock, prefix="demo")

    print("Saved artifacts:")
    for kind, path in paths.items():
        print(f"  {kind:<12} → {path}")

    print("\nModule 7 complete.")
