"""
Module 8 — Streamlit UI
========================
Interactive route-optimization dashboard for Dhaka city.

Layout
------
Sidebar   : location picker, algorithm selector, weight sliders,
            heuristic toggle, time-of-day / weather simulation
Main area : Map tab | Charts tab | Table tab | Cost Breakdown tab

Run with:
    streamlit run app/streamlit_app.py
from the project root directory.
"""

from __future__ import annotations

import os
import sys
import copy
import logging

# Ensure project root is on the path regardless of where Streamlit is launched
_APP_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(_APP_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import osmnx as ox
import pandas as pd

from graph.enrichment          import load_enriched, enrich_and_save
from cost.cost_function        import CostFunction, CostWeights, PRESETS
from algorithms                import run_algorithm, run_all, ALGORITHMS
from algorithms.heuristics     import HEURISTICS
from dynamic.traffic_simulation import build_simulation
from visualization.map_plot    import plot_comparison, plot_route, ALGO_COLORS, save_map
from visualization.analysis    import (
    plot_bar_charts, plot_radar, plot_time_bars, to_dataframe, print_summary
)

logging.basicConfig(level=logging.WARNING)

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dhaka Route Optimizer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Named Dhaka locations  { display name : (lat, lon) }
# ──────────────────────────────────────────────────────────────────────────────

LOCATIONS: dict[str, tuple[float, float]] = {
    "Shahbagh":          (23.7387, 90.3949),
    "TSC (Dhaka Uni.)":  (23.7337, 90.3928),
    "Bangla Motor":      (23.7500, 90.3919),
    "Farmgate":          (23.7584, 90.3897),
    "Dhanmondi 27":      (23.7461, 90.3742),
    "Mohammadpur":       (23.7586, 90.3568),
    "Mirpur 10":         (23.8083, 90.3670),
    "Uttara Sector 10":  (23.8759, 90.3981),
    "Gulshan 1":         (23.7808, 90.4142),
    "Banani":            (23.7939, 90.4045),
    "Badda":             (23.7760, 90.4287),
    "Rampura":           (23.7700, 90.4273),
    "Tejgaon":           (23.7681, 90.3980),
    "Motijheel":         (23.7226, 90.4178),
    "Old Dhaka (Sadarghat)": (23.7104, 90.4074),
}

LOCATION_NAMES = list(LOCATIONS.keys())

# ──────────────────────────────────────────────────────────────────────────────
# Cached graph loader
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Dhaka road network …")
def _load_graph():
    return load_enriched()


@st.cache_data(show_spinner=False)
def _nearest_node(_G, lat: float, lon: float) -> int:
    """Cached osmnx nearest-node lookup (hash on lat/lon)."""
    return ox.nearest_nodes(_G, X=lon, Y=lat)


# ──────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ──────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "results":      None,
        "folium_map":   None,
        "src_name":     LOCATION_NAMES[0],
        "dst_name":     LOCATION_NAMES[8],    # Gulshan
        "ran_once":     False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — all controls
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🗺️ Dhaka Route Optimizer")
    st.caption("AI-Based Intelligent Routing System")
    st.divider()

    # ── Route settings ─────────────────────────────────────────────────────
    st.subheader("📍 Route")

    col_s, col_swap = st.columns([5, 1])
    with col_s:
        src_name = st.selectbox(
            "Source", LOCATION_NAMES,
            index=LOCATION_NAMES.index(st.session_state["src_name"]),
            key="src_select",
        )
    with col_swap:
        st.write("")
        st.write("")
        if st.button("↕", help="Swap source and destination"):
            st.session_state["src_name"], st.session_state["dst_name"] = (
                st.session_state["dst_name"], st.session_state["src_name"]
            )
            st.rerun()

    dst_name = st.selectbox(
        "Destination", LOCATION_NAMES,
        index=LOCATION_NAMES.index(st.session_state["dst_name"]),
        key="dst_select",
    )
    st.session_state["src_name"] = src_name
    st.session_state["dst_name"] = dst_name

    st.divider()

    # ── Algorithm settings ──────────────────────────────────────────────────
    st.subheader("🤖 Algorithms")

    selected_algos = st.multiselect(
        "Run algorithms",
        options=list(ALGORITHMS.keys()),
        default=["bfs", "ucs", "astar"],
        format_func=str.upper,
    )

    heuristic = st.radio(
        "Heuristic (Greedy & A*)",
        options=list(HEURISTICS.keys()),
        format_func=lambda x: {"euclidean": "Euclidean distance",
                               "travel_time": "Travel time estimate",
                               "risk_aware": "Risk-aware (tightest)"}[x],
        horizontal=False,
    )

    st.divider()

    # ── Cost weights ────────────────────────────────────────────────────────
    st.subheader("⚖️ Cost Weights")

    preset_name = st.selectbox(
        "Load preset",
        options=["custom"] + list(PRESETS.keys()),
        format_func=lambda x: x.replace("_", " ").title(),
    )

    # Pre-fill slider defaults from preset
    pw = PRESETS[preset_name].normalised() if preset_name != "custom" else CostWeights()

    w1 = st.slider("w1 — Travel Time",    0.0, 1.0, float(f"{pw.w1_time:.2f}"),           0.05)
    w2 = st.slider("w2 — Distance",       0.0, 1.0, float(f"{pw.w2_distance:.2f}"),       0.05)
    w3 = st.slider("w3 — Traffic",        0.0, 1.0, float(f"{pw.w3_traffic:.2f}"),        0.05)
    w4 = st.slider("w4 — Safety",         0.0, 1.0, float(f"{pw.w4_safety:.2f}"),         0.05)
    w5 = st.slider("w5 — Gender Safety",  0.0, 1.0, float(f"{pw.w5_gender_safety:.2f}"),  0.05)
    w6 = st.slider("w6 — Road Condition", 0.0, 1.0, float(f"{pw.w6_road_condition:.2f}"), 0.05)

    total_w = w1 + w2 + w3 + w4 + w5 + w6
    if total_w == 0:
        st.warning("All weights are zero — set at least one above 0.")
    else:
        st.caption(f"Weights sum: {total_w:.2f} (auto-normalised on run)")

    st.divider()

    # ── Simulation settings ─────────────────────────────────────────────────
    st.subheader("🕐 Simulation")

    time_of_day = st.slider(
        "Time of day", 0.0, 23.9, 8.5, 0.5,
        format="%.1f h",
        help="Affects traffic congestion and gender-safety scores",
    )
    h, m_frac   = int(time_of_day), int((time_of_day % 1) * 60)
    st.caption(f"Simulated time: {h:02d}:{m_frac:02d}")

    weather = st.selectbox(
        "Weather",
        ["dry", "rain", "flood"],
        format_func=lambda x: {"dry": "☀️ Dry", "rain": "🌧️ Rain",
                               "flood": "🌊 Flood"}[x],
    )

    st.divider()

    # ── Run button ──────────────────────────────────────────────────────────
    run_disabled = (total_w == 0 or len(selected_algos) == 0
                    or src_name == dst_name)
    run_clicked  = st.button(
        "🚀 Run Route Search",
        type="primary",
        disabled=run_disabled,
        use_container_width=True,
    )

    if src_name == dst_name:
        st.warning("Source and destination must differ.")

# ──────────────────────────────────────────────────────────────────────────────
# Graph loading
# ──────────────────────────────────────────────────────────────────────────────

try:
    G = _load_graph()
    graph_ok = True
except FileNotFoundError:
    graph_ok = False

# ──────────────────────────────────────────────────────────────────────────────
# Run logic
# ──────────────────────────────────────────────────────────────────────────────

if run_clicked and graph_ok and total_w > 0 and selected_algos:
    with st.spinner("Applying simulation …"):
        sim      = build_simulation(weather=weather, start_hour=time_of_day, G=G)
        sim.clock.set_hour(time_of_day)
        sim.update()                         # writes dynamic attrs onto G in-place

    weights  = CostWeights(w1_time=w1, w2_distance=w2, w3_traffic=w3,
                           w4_safety=w4, w5_gender_safety=w5, w6_road_condition=w6)
    cost_fn  = CostFunction(weights)

    src_node = _nearest_node(G, *LOCATIONS[src_name])
    dst_node = _nearest_node(G, *LOCATIONS[dst_name])

    with st.spinner(f"Running {len(selected_algos)} algorithm(s) …"):
        results = {}
        prog    = st.progress(0)
        for i, algo in enumerate(selected_algos):
            results[algo] = run_algorithm(
                algo, G, src_node, dst_node, cost_fn, heuristic=heuristic
            )
            prog.progress((i + 1) / len(selected_algos))
        prog.empty()

    with st.spinner("Building map …"):
        if len(results) == 1:
            name, res = next(iter(results.items()))
            fmap = plot_route(G, res, source_name=src_name, target_name=dst_name)
        else:
            fmap = plot_comparison(G, results, source_name=src_name,
                                   target_name=dst_name)

    st.session_state["results"]    = results
    st.session_state["folium_map"] = fmap
    st.session_state["cost_fn"]    = cost_fn
    st.session_state["ran_once"]   = True
    st.session_state["src_node"]   = src_node
    st.session_state["dst_node"]   = dst_node

# ──────────────────────────────────────────────────────────────────────────────
# Main content area
# ──────────────────────────────────────────────────────────────────────────────

if not graph_ok:
    st.error("Graph not found. Run the pipeline first:", icon="⚠️")
    st.code("python graph/map_loader.py\npython graph/enrichment.py", language="bash")
    st.stop()

st.title("Dhaka Intelligent Route Optimizer")

if not st.session_state["ran_once"]:
    st.info(
        "Configure your route and settings in the sidebar, then click "
        "**🚀 Run Route Search** to compute and compare routes.",
        icon="ℹ️",
    )
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/"
        "Flag_of_Bangladesh.svg/320px-Flag_of_Bangladesh.svg.png",
        width=160,
        caption="Dhaka Road Network — OpenStreetMap",
    )
    st.stop()

results  = st.session_state["results"]
fmap     = st.session_state["folium_map"]
cost_fn  = st.session_state["cost_fn"]
src_node = st.session_state["src_node"]
dst_node = st.session_state["dst_node"]

# ── Result header ────────────────────────────────────────────────────────────
found_n = sum(1 for r in results.values() if r.found)
st.success(
    f"**{src_name}  →  {dst_name}** · "
    f"{found_n}/{len(results)} algorithms found a path · "
    f"Time: {h:02d}:{m_frac:02d} · Weather: {weather}",
    icon="✅",
)

# ── Tabs ────────────────────────────────────────────────────────────────────
tab_map, tab_charts, tab_table, tab_breakdown = st.tabs(
    ["🗺️  Map", "📊  Charts", "📋  Table", "💡  Cost Breakdown"]
)

# ── Tab 1: Map ───────────────────────────────────────────────────────────────
with tab_map:
    try:
        from streamlit_folium import st_folium
        st_folium(fmap, height=600, use_container_width=True,
                  returned_objects=[], key="route_map")
    except ImportError:
        st.warning(
            "Install `streamlit-folium` to view the interactive map: "
            "`pip install streamlit-folium`"
        )
        map_path = save_map(fmap, "latest_route.html")
        st.info(f"Map saved to: `{map_path}` — open in browser.")

# ── Tab 2: Charts ────────────────────────────────────────────────────────────
with tab_charts:
    found_results = {k: v for k, v in results.items() if v.found}

    if not found_results:
        st.warning("No paths found — no charts to display.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_bars = plot_bar_charts(found_results,
                                       title=f"{src_name} → {dst_name}")
            st.pyplot(fig_bars, use_container_width=True)
        with c2:
            fig_radar = plot_radar(found_results)
            st.pyplot(fig_radar, use_container_width=True)

        fig_time = plot_time_bars(found_results)
        st.pyplot(fig_time, use_container_width=True)

# ── Tab 3: Table ─────────────────────────────────────────────────────────────
with tab_table:
    df = to_dataframe(results)

    # Colour-code cost column: green = lowest, red = highest
    def _highlight_best(col):
        if col.name in ("path_cost", "execution_time_ms", "nodes_explored",
                        "path_length_km"):
            best = col.min()
            return ["background-color: #C8E6C9" if v == best else "" for v in col]
        if col.name in ("avg_safety", "avg_gender_safety"):
            best = col.max()
            return ["background-color: #C8E6C9" if v == best else "" for v in col]
        return [""] * len(col)

    styled = (
        df.style
        .apply(_highlight_best)
        .format({
            "path_cost":         "{:.4f}",
            "execution_time_ms": "{:.2f} ms",
            "nodes_explored":    "{:,.0f}",
            "path_edges":        "{:.0f}",
            "path_length_km":    "{:.3f} km",
            "avg_safety":        "{:.3f}",
            "avg_gender_safety": "{:.3f}",
            "found":             lambda x: "✓" if x else "✗",
        }, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, height=280)

    # Download CSV
    csv = df.to_csv().encode("utf-8")
    st.download_button(
        "⬇️  Download CSV",
        data=csv,
        file_name=f"routes_{src_name}_to_{dst_name}.csv".replace(" ", "_"),
        mime="text/csv",
    )

# ── Tab 4: Cost Breakdown ─────────────────────────────────────────────────────
with tab_breakdown:
    st.markdown("#### Per-component cost contribution for each algorithm's best edge")
    st.caption(
        "Shows how much each weight factor contributes to the cost of one "
        "representative edge on each path."
    )

    found = {k: v for k, v in results.items() if v.found and len(v.path) >= 2}
    if not found:
        st.warning("No paths found.")
    else:
        breakdown_rows = []
        for algo_name, res in found.items():
            u, v_node = res.path[0], res.path[1]
            # Pick the cheapest edge between these two nodes
            edge_data = min(G[u][v_node].values(),
                            key=lambda d: cost_fn(u, v_node, d))
            bd = cost_fn.breakdown(u, v_node, edge_data)
            bd["algorithm"] = algo_name.upper()
            breakdown_rows.append(bd)

        bd_df = pd.DataFrame(breakdown_rows).set_index("algorithm")
        st.dataframe(
            bd_df.style.background_gradient(cmap="YlOrRd", axis=None),
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("**Active weight configuration (normalised):**")
        w  = cost_fn.weights
        wc = st.columns(6)
        labels = ["w1 Time", "w2 Dist", "w3 Traffic",
                  "w4 Safety", "w5 Gender", "w6 Cond"]
        vals   = [w.w1_time, w.w2_distance, w.w3_traffic,
                  w.w4_safety, w.w5_gender_safety, w.w6_road_condition]
        for col, lbl, val in zip(wc, labels, vals):
            col.metric(lbl, f"{val:.3f}")
