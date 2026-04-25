"""
main.py — End-to-end CLI runner
================================
Runs the full pipeline from graph loading to visualisation.
Use this to verify the entire stack before launching the Streamlit UI.

Usage
-----
    python main.py                        # default: Shahbagh → Gulshan, all algos
    python main.py --src Shahbagh --dst "Gulshan 1" --algos ucs astar
    python main.py --preset women_safe --weather rain --hour 21.0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Named locations (shared with streamlit_app) ───────────────────────────────
LOCATIONS: dict[str, tuple[float, float]] = {
    "Shahbagh":               (23.7387, 90.3949),
    "TSC (Dhaka Uni.)":       (23.7337, 90.3928),
    "Bangla Motor":           (23.7500, 90.3919),
    "Farmgate":               (23.7584, 90.3897),
    "Dhanmondi 27":           (23.7461, 90.3742),
    "Mohammadpur":            (23.7586, 90.3568),
    "Mirpur 10":              (23.8083, 90.3670),
    "Uttara Sector 10":       (23.8759, 90.3981),
    "Gulshan 1":              (23.7808, 90.4142),
    "Banani":                 (23.7939, 90.4045),
    "Motijheel":              (23.7226, 90.4178),
    "Old Dhaka (Sadarghat)":  (23.7104, 90.4074),
}


def main():
    parser = argparse.ArgumentParser(description="Dhaka Route Optimizer — CLI runner")
    parser.add_argument("--src",     default="Shahbagh",   help="Source location name")
    parser.add_argument("--dst",     default="Gulshan 1",  help="Destination location name")
    parser.add_argument("--algos",   nargs="+",
                        default=["bfs", "ucs", "greedy", "astar"],
                        help="Algorithms to run")
    parser.add_argument("--heuristic", default="euclidean",
                        choices=["euclidean", "travel_time", "risk_aware"])
    parser.add_argument("--preset",  default="balanced",
                        choices=["fastest", "shortest", "safest",
                                 "women_safe", "balanced", "comfort"])
    parser.add_argument("--weather", default="dry",
                        choices=["dry", "rain", "flood"])
    parser.add_argument("--hour",    type=float, default=8.5,
                        help="Simulation time (24-h float, e.g. 8.5 = 08:30)")
    parser.add_argument("--no-map",  action="store_true",
                        help="Skip map generation")
    args = parser.parse_args()

    # ── Validate locations ────────────────────────────────────────────────────
    if args.src not in LOCATIONS:
        print(f"Unknown source '{args.src}'. Available: {list(LOCATIONS)}")
        sys.exit(1)
    if args.dst not in LOCATIONS:
        print(f"Unknown destination '{args.dst}'. Available: {list(LOCATIONS)}")
        sys.exit(1)

    print("\n" + "═" * 60)
    print("  Dhaka Intelligent Route Optimizer")
    print("═" * 60)

    # ── Step 1: Load graph ────────────────────────────────────────────────────
    print("\n[1/5] Loading enriched graph …")
    from graph.enrichment import enrich_and_save
    G = enrich_and_save(force_rebuild=False)
    logger.info("Graph ready — %d nodes, %d edges",
                G.number_of_nodes(), G.number_of_edges())

    # ── Step 2: Apply dynamic simulation ─────────────────────────────────────
    print(f"\n[2/5] Applying simulation (hour={args.hour:.1f}, weather={args.weather}) …")
    from dynamic.traffic_simulation import build_simulation
    sim = build_simulation(weather=args.weather, start_hour=args.hour, G=G)
    sim.clock.set_hour(args.hour)
    snap = sim.update()
    logger.info("Snapshot applied: %s", snap.summary().split("\n")[0])

    # ── Step 3: Build cost function ───────────────────────────────────────────
    print(f"\n[3/5] Building cost function (preset: {args.preset}) …")
    from cost.cost_function import CostFunction, PRESETS
    cost_fn = CostFunction(PRESETS[args.preset])
    logger.info("Cost function: %s", cost_fn)

    # ── Step 4: Find source / destination nodes ───────────────────────────────
    print("\n[4/5] Resolving node IDs …")
    import osmnx as ox
    src_lat, src_lon = LOCATIONS[args.src]
    dst_lat, dst_lon = LOCATIONS[args.dst]
    src_node = ox.nearest_nodes(G, X=src_lon, Y=src_lat)
    dst_node = ox.nearest_nodes(G, X=dst_lon, Y=dst_lat)
    logger.info("Source: %s → node %d", args.src, src_node)
    logger.info("Destination: %s → node %d", args.dst, dst_node)

    # ── Step 5: Run algorithms ────────────────────────────────────────────────
    print(f"\n[5/5] Running algorithms: {args.algos} …")
    from algorithms import run_algorithm
    results = {}
    for algo in args.algos:
        logger.info("  Running %s …", algo)
        results[algo] = run_algorithm(
            algo, G, src_node, dst_node, cost_fn, heuristic=args.heuristic
        )

    # ── Print summary table ───────────────────────────────────────────────────
    from visualization.analysis import print_summary, save_report
    print_summary(results)

    # ── Save report ───────────────────────────────────────────────────────────
    prefix = f"{args.src}_to_{args.dst}".replace(" ", "_").replace("(", "").replace(")", "")
    saved  = save_report(results, prefix=prefix)
    print("\nSaved artifacts:")
    for kind, path in saved.items():
        print(f"  {kind:<12} → {path}")

    # ── Generate map ──────────────────────────────────────────────────────────
    if not args.no_map:
        from visualization.map_plot import plot_comparison, save_map
        print("\nGenerating map …")
        fmap = plot_comparison(
            G, results,
            source_name=args.src,
            target_name=args.dst,
        )
        map_path = save_map(fmap, f"{prefix}_map.html")
        print(f"  Map → {map_path}")
        print("  Open the HTML file in a browser to view the interactive map.")

    print("\n" + "═" * 60)
    print("  Pipeline complete.")
    print("  Launch the UI:  streamlit run app/streamlit_app.py")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
