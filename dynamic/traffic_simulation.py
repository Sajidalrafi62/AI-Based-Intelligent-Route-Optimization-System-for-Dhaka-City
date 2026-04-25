"""
Module 3 — Synthetic Dynamic Data
===================================
Simulates time-varying edge attributes on the enriched Dhaka graph.

What this module does
---------------------
Module 2 sets *static* OSM-derived baselines.
Module 3 *overrides* four of those attributes at runtime:
    traffic_level, road_condition, safety_score, gender_safety_score
It also recomputes travel_time to reflect current congestion.

Architecture — designed for live-API drop-in
--------------------------------------------

    DataProvider (ABC)          ← the single swap-point for real APIs
         │
         ├── SyntheticProvider  ← time-of-day + area rules (built here)
         └── LiveDataProvider   ← documented stub; implement 3 methods to go live

    SimulationClock             ← manages simulation time (real or accelerated)
    AreaClassifier              ← maps edge midpoints to Dhaka neighbourhoods
    DynamicUpdater              ← calls provider, writes values onto graph edges

Categorical labels (for UI display)
------------------------------------
    traffic_level      → "low" | "medium" | "high"
    road_condition     → "good" | "average" | "bad"
    safety_score       → "high" | "medium" | "low"
    gender_safety_score→ "safe" | "moderate" | "unsafe"
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import networkx as nx

# Ensure project root is on sys.path when this file is run directly
import sys as _sys, os as _os
_pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _pkg_root not in _sys.path:
    _sys.path.insert(0, _pkg_root)

from graph.enrichment import load_enriched, ENRICHED_ATTRS

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Categorical label helpers
# ──────────────────────────────────────────────────────────────────────────────

def traffic_label(v: float) -> str:
    """Convert 0-1 float to 'low' / 'medium' / 'high'."""
    if v < 0.33:  return "low"
    if v < 0.66:  return "medium"
    return "high"


def condition_label(v: float) -> str:
    """Convert 0-1 float to 'good' / 'average' / 'bad'."""
    if v > 0.67:  return "good"
    if v > 0.33:  return "average"
    return "bad"


def safety_label(v: float) -> str:
    """Convert 0-1 float to 'high' / 'medium' / 'low'."""
    if v > 0.67:  return "high"
    if v > 0.33:  return "medium"
    return "low"


def gender_safety_label(v: float) -> str:
    """Convert 0-1 float to 'safe' / 'moderate' / 'unsafe'."""
    if v > 0.67:  return "safe"
    if v > 0.33:  return "moderate"
    return "unsafe"


# ──────────────────────────────────────────────────────────────────────────────
# Dhaka area definitions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AreaProfile:
    """
    Bounding-box neighbourhood with per-attribute bias values.
    Bias is added on top of the highway-type base score.
    """
    name:               str
    lat_lo:             float
    lat_hi:             float
    lon_lo:             float
    lon_hi:             float
    traffic_bias:       float = 0.0
    condition_bias:     float = 0.0
    safety_bias:        float = 0.0
    gender_safety_bias: float = 0.0


# Calibrated to known Dhaka neighbourhood characteristics
DHAKA_AREAS: List[AreaProfile] = [
    AreaProfile("Old Dhaka",        23.69, 23.73, 90.39, 90.43,
                traffic_bias=+0.10, condition_bias=-0.12,
                safety_bias=-0.10,  gender_safety_bias=-0.12),

    AreaProfile("Motijheel",        23.72, 23.74, 90.41, 90.44,
                traffic_bias=+0.15, condition_bias=-0.06,
                safety_bias=-0.05,  gender_safety_bias=-0.06),

    AreaProfile("Farmgate/Tejgaon", 23.75, 23.77, 90.38, 90.40,
                traffic_bias=+0.18, condition_bias=-0.05,
                safety_bias=-0.04,  gender_safety_bias=-0.05),

    AreaProfile("Dhanmondi",        23.74, 23.76, 90.36, 90.38,
                traffic_bias=+0.05, condition_bias=+0.02,
                safety_bias=+0.03,  gender_safety_bias=+0.04),

    AreaProfile("Gulshan/Banani",   23.77, 23.80, 90.40, 90.42,
                traffic_bias=+0.03, condition_bias=+0.08,
                safety_bias=+0.07,  gender_safety_bias=+0.08),

    AreaProfile("Mirpur",           23.79, 23.84, 90.34, 90.38,
                traffic_bias=+0.12, condition_bias=-0.04,
                safety_bias=-0.03,  gender_safety_bias=-0.04),

    AreaProfile("Uttara",           23.85, 23.89, 90.38, 90.41,
                traffic_bias=-0.05, condition_bias=+0.06,
                safety_bias=+0.05,  gender_safety_bias=+0.06),

    AreaProfile("Rayer Bazar",      23.76, 23.78, 90.35, 90.37,
                traffic_bias=+0.08, condition_bias=-0.08,
                safety_bias=-0.07,  gender_safety_bias=-0.09),
]

_DEFAULT_AREA = AreaProfile("Default", 0, 90, 88, 93)


class AreaClassifier:
    """Maps a (lat, lon) coordinate to its AreaProfile."""

    def classify(self, lat: float, lon: float) -> AreaProfile:
        for area in DHAKA_AREAS:
            if area.lat_lo <= lat <= area.lat_hi and area.lon_lo <= lon <= area.lon_hi:
                return area
        return _DEFAULT_AREA

    def edge_area(self, G: nx.MultiDiGraph, u: int, v: int) -> AreaProfile:
        """Classify by the midpoint of the two endpoint nodes."""
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        lat = (u_data.get("y", 23.78) + v_data.get("y", 23.78)) / 2.0
        lon = (u_data.get("x", 90.40) + v_data.get("x", 90.40)) / 2.0
        return self.classify(lat, lon)


# ──────────────────────────────────────────────────────────────────────────────
# Simulation clock
# ──────────────────────────────────────────────────────────────────────────────

class SimulationClock:
    """
    Manages a simulation timestamp that advances at a configurable speed.

    Parameters
    ----------
    start : simulation start datetime (defaults to today 07:00)
    speed : sim-seconds per real-second.  1 = real-time, 60 = 1 min per sec.
    """

    def __init__(self, start: Optional[datetime] = None, speed: float = 1.0):
        self._origin_sim  = start or datetime.now().replace(
            hour=7, minute=0, second=0, microsecond=0
        )
        self._origin_real = time.monotonic()
        self._speed       = speed
        self._offset      = timedelta(0)   # accumulated from set_time / tick

    def now(self) -> datetime:
        elapsed_sim = timedelta(
            seconds=(time.monotonic() - self._origin_real) * self._speed
        )
        return self._origin_sim + elapsed_sim + self._offset

    def hour(self) -> float:
        """Current sim time as a 24-h float, e.g. 8.5 = 08:30."""
        t = self.now()
        return t.hour + t.minute / 60.0 + t.second / 3600.0

    def tick(self, minutes: float) -> datetime:
        """Advance the clock by `minutes` of simulation time."""
        self._offset += timedelta(minutes=minutes)
        return self.now()

    def set_hour(self, hour: float) -> None:
        """Jump the simulation clock to a specific hour of the current day."""
        h = int(hour)
        m = int((hour % 1) * 60)
        target = self._origin_sim.replace(hour=h, minute=m, second=0, microsecond=0)
        self._offset      = target - self._origin_sim
        self._origin_real = time.monotonic()

    def __repr__(self) -> str:
        return (f"SimulationClock("
                f"now={self.now().strftime('%Y-%m-%d %H:%M')}, "
                f"speed={self._speed}x)")


# ──────────────────────────────────────────────────────────────────────────────
# DataProvider ABC  — the single swap-point for live APIs
# ──────────────────────────────────────────────────────────────────────────────

class DataProvider(ABC):
    """
    Abstract interface for dynamic edge-attribute data.

    To replace synthetic data with a live API
    ------------------------------------------
    1.  Subclass DataProvider.
    2.  Implement refresh() and the four get_*() methods below.
    3.  Pass your subclass to DynamicUpdater(provider=YourProvider()).
    4.  Nothing else in the project needs to change.

    Contract
    --------
    - refresh(ts) is called ONCE per update cycle.
      Use it to bulk-fetch from your API and cache results locally.
    - Each get_*() is called per edge; must return a float in [0, 1].
    - On API failure, return the edge's current Module-2 baseline
      via edge_data.get(attr) — never raise from a getter.
    """

    @abstractmethod
    def refresh(self, timestamp: datetime) -> None:
        """Pre-fetch or cache data for the given timestamp."""

    @abstractmethod
    def get_traffic_level(
        self, u: int, v: int, k: int,
        edge_data: dict, node_u: dict, node_v: dict,
        timestamp: datetime,
    ) -> float:
        """0 = free flow, 1 = gridlock."""

    @abstractmethod
    def get_road_condition(
        self, u: int, v: int, k: int,
        edge_data: dict, node_u: dict, node_v: dict,
        timestamp: datetime,
    ) -> float:
        """0 = impassable, 1 = perfect."""

    @abstractmethod
    def get_safety_score(
        self, u: int, v: int, k: int,
        edge_data: dict, node_u: dict, node_v: dict,
        timestamp: datetime,
    ) -> float:
        """0 = most dangerous, 1 = safest."""

    @abstractmethod
    def get_gender_safety_score(
        self, u: int, v: int, k: int,
        edge_data: dict, node_u: dict, node_v: dict,
        timestamp: datetime,
    ) -> float:
        """0 = least safe for women, 1 = safest."""


# ──────────────────────────────────────────────────────────────────────────────
# SyntheticProvider
# ──────────────────────────────────────────────────────────────────────────────

def _tick_noise(u: int, v: int, k: int, seed: int,
                timestamp: datetime, lo: float, hi: float) -> float:
    """
    Per-edge noise that changes every 30-minute window.
    Deterministic: same edge + same 30-min bucket → same value.
    """
    bucket = (timestamp.hour * 2 + timestamp.minute // 30)
    digest = hashlib.md5(f"{u}_{v}_{k}_{seed}_{bucket}".encode()).hexdigest()
    ratio  = int(digest[:8], 16) / 0xFFFF_FFFF
    return lo + ratio * (hi - lo)


def _peak_multiplier(hour: float) -> float:
    """
    1.0 (off-peak) → 1.45 (peak).
    Two Gaussian peaks: morning rush 08:30, evening rush 18:00.
    """
    am = math.exp(-0.5 * ((hour -  8.5) / 1.2) ** 2)
    pm = math.exp(-0.5 * ((hour - 18.0) / 1.2) ** 2)
    return 1.0 + 0.45 * max(am, pm)


def _night_reduction(hour: float) -> float:
    """Traffic drops significantly after midnight — returns multiplier ≤ 1."""
    if hour < 5.0:
        return max(0.45, 1.0 - 0.11 * (5.0 - hour))
    return 1.0


# Highway-type baselines — same calibration as Module 2 but re-declared here
# so SyntheticProvider is self-contained and not coupled to enrichment internals.
_HW_TRAFFIC   = {"motorway":0.28,"trunk":0.50,"primary":0.72,"secondary":0.65,
                  "tertiary":0.55,"residential":0.40,"unclassified":0.45,"service":0.35}
_HW_CONDITION = {"motorway":0.80,"trunk":0.72,"primary":0.65,"secondary":0.60,
                  "tertiary":0.55,"residential":0.48,"unclassified":0.43,"service":0.40}
_HW_SAFETY    = {"motorway":0.82,"trunk":0.73,"primary":0.66,"secondary":0.63,
                  "tertiary":0.60,"residential":0.68,"unclassified":0.53,"service":0.55}
_HW_GENDER    = {"motorway":0.62,"trunk":0.66,"primary":0.74,"secondary":0.70,
                  "tertiary":0.66,"residential":0.56,"unclassified":0.46,"service":0.40}

_WEATHER_DELTA = {
    "dry":   {"traffic": 0.00, "condition": 0.00},
    "rain":  {"traffic":+0.12, "condition":-0.15},
    "flood": {"traffic":+0.25, "condition":-0.30},
}


def _hw(data: dict) -> str:
    hw = data.get("highway", "unclassified")
    return hw[0] if isinstance(hw, list) else str(hw)


class SyntheticProvider(DataProvider):
    """
    Fully deterministic synthetic data provider.

    Parameters
    ----------
    weather : 'dry' | 'rain' | 'flood' — applies global condition/traffic deltas
    """

    def __init__(self, weather: str = "dry"):
        self.weather      = weather
        self._area_cache: Dict[Tuple[int, int], AreaProfile] = {}
        self._G: Optional[nx.MultiDiGraph] = None   # injected by DynamicUpdater
        self._classifier  = AreaClassifier()

    def _area(self, u: int, v: int) -> AreaProfile:
        key = (u, v)
        if key not in self._area_cache and self._G is not None:
            self._area_cache[key] = self._classifier.edge_area(self._G, u, v)
        return self._area_cache.get(key, _DEFAULT_AREA)

    def refresh(self, timestamp: datetime) -> None:
        pass   # no external calls needed in synthetic mode

    def get_traffic_level(self, u, v, k, edge_data, node_u, node_v,
                           timestamp) -> float:
        hour  = timestamp.hour + timestamp.minute / 60.0
        base  = _HW_TRAFFIC.get(_hw(edge_data), 0.50)
        area  = self._area(u, v)
        wd    = _WEATHER_DELTA.get(self.weather, {}).get("traffic", 0.0)
        noise = _tick_noise(u, v, k, seed=0, timestamp=timestamp, lo=-0.10, hi=0.10)
        val   = base * _peak_multiplier(hour) * _night_reduction(hour) \
                + area.traffic_bias + wd + noise
        return min(1.0, max(0.0, val))

    def get_road_condition(self, u, v, k, edge_data, node_u, node_v,
                            timestamp) -> float:
        # Slow weekly drift: condition dips mid-week (accumulated wear),
        # recovers slightly at weekend (proxy for maintenance cycles).
        day         = timestamp.weekday()          # 0 = Mon
        weekly_drift = -0.03 * math.sin(math.pi * day / 6.0)

        base  = _HW_CONDITION.get(_hw(edge_data), 0.50)
        area  = self._area(u, v)
        wd    = _WEATHER_DELTA.get(self.weather, {}).get("condition", 0.0)
        noise = _tick_noise(u, v, k, seed=1, timestamp=timestamp, lo=-0.07, hi=0.07)
        val   = base + area.condition_bias + weekly_drift + wd + noise
        return min(1.0, max(0.05, val))

    def get_safety_score(self, u, v, k, edge_data, node_u, node_v,
                          timestamp) -> float:
        hour     = timestamp.hour + timestamp.minute / 60.0
        is_night = not (6.0 <= hour <= 20.0)
        night_p  = 0.08 if is_night else 0.0

        base  = _HW_SAFETY.get(_hw(edge_data), 0.58)
        area  = self._area(u, v)
        noise = _tick_noise(u, v, k, seed=2, timestamp=timestamp, lo=-0.05, hi=0.05)
        val   = base + area.safety_bias - night_p + noise
        return min(1.0, max(0.05, val))

    def get_gender_safety_score(self, u, v, k, edge_data, node_u, node_v,
                                  timestamp) -> float:
        hour     = timestamp.hour + timestamp.minute / 60.0
        is_night = not (6.0 <= hour <= 20.0)

        # Lighting tag is the strongest static signal
        lit = str(edge_data.get("lit", "")).lower()
        if lit in ("yes", "24/7"):
            light_mod = +0.12
        elif lit == "no":
            light_mod = -0.15
        else:
            # No tag: apply night penalty when lighting status is unknown
            light_mod = -0.10 if is_night else 0.0

        night_p      = 0.13 if is_night else 0.0
        hw           = _hw(edge_data)
        isolation_p  = 0.08 if hw in ("service", "unclassified") else 0.0

        base  = _HW_GENDER.get(hw, 0.53)
        area  = self._area(u, v)
        noise = _tick_noise(u, v, k, seed=3, timestamp=timestamp, lo=-0.05, hi=0.05)
        val   = base + area.gender_safety_bias + light_mod - night_p - isolation_p + noise
        return min(1.0, max(0.05, val))


# ──────────────────────────────────────────────────────────────────────────────
# LiveDataProvider stub
# ──────────────────────────────────────────────────────────────────────────────

class LiveDataProvider(DataProvider):
    """
    Stub for future live-API integration.

    To go live, subclass this and override the three _fetch_*_batch methods.
    Everything else (the ABC contract, fallback logic) is already wired up.

    Suggested data sources for Dhaka
    ---------------------------------
    Traffic   : TomTom Traffic Flow API / Google Maps Roads API /
                OpenStreetMap-based HERE API
    Condition : BRTA crowdsourced pothole reports / civic-authority feeds
    Safety    : Bangladesh Police open-data / OSM amenity proximity layers

    Example skeleton
    ----------------
    class TomTomProvider(LiveDataProvider):
        def _fetch_traffic_batch(self, ts):
            resp = requests.get(TOMTOM_URL, params={...})
            return {(u,v,k): float for each segment in resp.json()}
    """

    def __init__(self, api_key: str = ""):
        self._api_key         = api_key
        self._traffic_cache:   Dict[Tuple[int, int, int], float] = {}
        self._condition_cache: Dict[Tuple[int, int, int], float] = {}
        self._safety_cache:    Dict[Tuple[int, int, int], float] = {}

    def refresh(self, timestamp: datetime) -> None:
        self._traffic_cache   = self._fetch_traffic_batch(timestamp)   or {}
        self._condition_cache = self._fetch_condition_batch(timestamp) or {}
        self._safety_cache    = self._fetch_safety_batch(timestamp)    or {}

    # ── override these in your concrete subclass ────────────────────────────

    def _fetch_traffic_batch(self, ts: datetime) -> Optional[Dict]:
        raise NotImplementedError

    def _fetch_condition_batch(self, ts: datetime) -> Optional[Dict]:
        raise NotImplementedError

    def _fetch_safety_batch(self, ts: datetime) -> Optional[Dict]:
        raise NotImplementedError

    # ── getters fall back to Module-2 baseline on cache miss ─────────────────

    def get_traffic_level(self, u, v, k, edge_data, node_u, node_v, timestamp):
        return self._traffic_cache.get((u, v, k),
               edge_data.get("traffic_level", 0.5))

    def get_road_condition(self, u, v, k, edge_data, node_u, node_v, timestamp):
        return self._condition_cache.get((u, v, k),
               edge_data.get("road_condition", 0.5))

    def get_safety_score(self, u, v, k, edge_data, node_u, node_v, timestamp):
        return self._safety_cache.get((u, v, k),
               edge_data.get("safety_score", 0.5))

    def get_gender_safety_score(self, u, v, k, edge_data, node_u, node_v, timestamp):
        return self._safety_cache.get((u, v, k),
               edge_data.get("gender_safety_score", 0.5))


# ──────────────────────────────────────────────────────────────────────────────
# DynamicUpdater
# ──────────────────────────────────────────────────────────────────────────────

DYNAMIC_ATTRS = ("traffic_level", "road_condition", "safety_score", "gender_safety_score")


class DynamicUpdater:
    """
    Ties together a graph, a clock, and a provider.
    Calling update() rewrites DYNAMIC_ATTRS on every edge for the current
    simulation timestamp and recomputes travel_time from updated traffic.

    Usage
    -----
        updater  = DynamicUpdater(G, clock, SyntheticProvider())
        snapshot = updater.update()          # one tick
        history  = updater.run(steps=48, minutes_per_step=30)
    """

    def __init__(self,
                 G:        nx.MultiDiGraph,
                 clock:    SimulationClock,
                 provider: DataProvider):
        self.G        = G
        self.clock    = clock
        self.provider = provider
        # Give SyntheticProvider a graph reference for area lookups
        if hasattr(provider, "_G"):
            provider._G = G

    def update(self, advance_minutes: float = 0.0) -> "Snapshot":
        """
        Recompute all dynamic attributes at the current (or advanced) time.
        Returns a Snapshot and writes values onto G in-place.
        """
        if advance_minutes:
            self.clock.tick(advance_minutes)

        ts = self.clock.now()
        self.provider.refresh(ts)

        edge_values: Dict[Tuple[int, int, int], Dict[str, float]] = {}

        for u, v, k, data in self.G.edges(keys=True, data=True):
            nu = self.G.nodes[u]
            nv = self.G.nodes[v]

            tl = self.provider.get_traffic_level(u, v, k, data, nu, nv, ts)
            rc = self.provider.get_road_condition(u, v, k, data, nu, nv, ts)
            ss = self.provider.get_safety_score(u, v, k, data, nu, nv, ts)
            gs = self.provider.get_gender_safety_score(u, v, k, data, nu, nv, ts)

            # Recompute travel_time: free-flow × congestion multiplier (1–3×)
            speed_ms  = max(float(data.get("speed_kph", 30.0)) * 1000 / 3600, 0.1)
            free_flow = float(data.get("length", 0.0)) / speed_ms
            tt        = free_flow * (1.0 + 2.0 * tl)

            attrs = {
                "traffic_level":        tl,
                "road_condition":       rc,
                "safety_score":         ss,
                "gender_safety_score":  gs,
                "travel_time":          tt,
            }
            data.update(attrs)
            edge_values[(u, v, k)] = attrs

        return Snapshot(timestamp=ts, edge_values=edge_values)

    def run(self,
            steps: int = 48,
            minutes_per_step: float = 30.0,
            verbose: bool = True) -> List["Snapshot"]:
        """
        Simulate multiple time steps; returns a list of Snapshot objects.

        Parameters
        ----------
        steps            : number of update cycles
        minutes_per_step : simulation minutes between cycles
        verbose          : print one-line progress per step
        """
        snapshots: List[Snapshot] = []
        hours = steps * minutes_per_step / 60.0
        logger.info("Running simulation: %d steps × %.0f min = %.1f hours",
                    steps, minutes_per_step, hours)

        for i in range(steps):
            advance = minutes_per_step if i > 0 else 0.0
            snap    = self.update(advance_minutes=advance)
            snapshots.append(snap)

            if verbose:
                sample = next(iter(snap.edge_values.values()))
                print(
                    f"  step {i+1:>3}/{steps}  "
                    f"{snap.timestamp.strftime('%H:%M')}  "
                    f"traffic={traffic_label(sample['traffic_level']):<6}  "
                    f"cond={condition_label(sample['road_condition']):<7}  "
                    f"safety={safety_label(sample['safety_score'])}"
                )

        return snapshots


# ──────────────────────────────────────────────────────────────────────────────
# Snapshot
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Snapshot:
    """
    Immutable record of dynamic attribute values captured at one timestamp.
    Can be replayed onto any graph instance with apply_to().
    """
    timestamp:   datetime
    edge_values: Dict[Tuple[int, int, int], Dict[str, float]]

    def apply_to(self, G: nx.MultiDiGraph) -> None:
        """Write this snapshot's values back onto graph edges in-place."""
        for (u, v, k), attrs in self.edge_values.items():
            if G.has_edge(u, v, k):
                G[u][v][k].update(attrs)

    def mean(self, attr: str) -> float:
        vals = [v[attr] for v in self.edge_values.values() if attr in v]
        return sum(vals) / len(vals) if vals else 0.0

    def summary(self) -> str:
        ts_str = self.timestamp.strftime("%Y-%m-%d %H:%M")
        lines  = [f"Snapshot @ {ts_str}  ({len(self.edge_values):,} edges)"]
        for attr in DYNAMIC_ATTRS:
            vals = [v[attr] for v in self.edge_values.values() if attr in v]
            if vals:
                lines.append(
                    f"  {attr:<24}  mean={sum(vals)/len(vals):.3f}"
                    f"  min={min(vals):.3f}  max={max(vals):.3f}"
                )
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Snapshot persistence
# ──────────────────────────────────────────────────────────────────────────────

_HERE          = os.path.dirname(os.path.abspath(__file__))
_ROOT          = os.path.dirname(_HERE)
SNAPSHOTS_PKL  = os.path.join(_ROOT, "data", "snapshots.pkl")


def save_snapshots(snapshots: List[Snapshot]) -> None:
    os.makedirs(os.path.dirname(SNAPSHOTS_PKL), exist_ok=True)
    with open(SNAPSHOTS_PKL, "wb") as f:
        pickle.dump(snapshots, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved %d snapshots → %s", len(snapshots), SNAPSHOTS_PKL)


def load_snapshots() -> List[Snapshot]:
    with open(SNAPSHOTS_PKL, "rb") as f:
        return pickle.load(f)


def snapshot_at_hour(snapshots: List[Snapshot], target_hour: float) -> Optional[Snapshot]:
    """Return the snapshot whose timestamp is closest to target_hour."""
    if not snapshots:
        return None
    return min(snapshots,
               key=lambda s: abs(s.timestamp.hour + s.timestamp.minute / 60.0
                                 - target_hour))


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────────────────

def build_simulation(
    weather:    str   = "dry",
    start_hour: float = 7.0,
    speed:      float = 1.0,
    G: Optional[nx.MultiDiGraph] = None,
) -> DynamicUpdater:
    """
    One-liner to create a ready-to-use DynamicUpdater.

    Parameters
    ----------
    weather    : 'dry' | 'rain' | 'flood'
    start_hour : simulation start time as 24-h float (e.g. 7.5 = 07:30)
    speed      : clock speed multiplier
    G          : pre-loaded graph (loaded from cache if None)

    Example
    -------
        updater  = build_simulation(weather="rain", start_hour=8.5)
        snapshot = updater.update()
    """
    if G is None:
        G = load_enriched()

    h = int(start_hour)
    m = int((start_hour % 1) * 60)
    start = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)

    clock    = SimulationClock(start=start, speed=speed)
    provider = SyntheticProvider(weather=weather)
    return DynamicUpdater(G, clock, provider)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=== Module 3 — Synthetic Dynamic Data ===\n")

    sim = build_simulation(weather="dry", start_hour=7.0)

    # Simulate a full 24-hour day in 48 × 30-min steps
    print("Simulating 24-hour day (48 × 30-min steps) ...\n")
    snapshots = sim.run(steps=48, minutes_per_step=30, verbose=True)

    # Print summary at key hours
    print("\n── Hourly highlights ──────────────────────────────────────────")
    for h in [7, 9, 12, 18, 21, 23]:
        snap = snapshot_at_hour(snapshots, h)
        if snap:
            print(snap.summary())
            print()

    save_snapshots(snapshots)
    print("Module 3 complete.")
