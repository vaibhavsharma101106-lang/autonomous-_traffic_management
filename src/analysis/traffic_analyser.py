"""
Module 3: Traffic Analysis
IT 207 Human Intelligence Project

Implements lane-wise traffic density estimation, queue length calculation,
traffic state classification, and rolling history management.

Human Intelligence Mapping — Selective Attention + Working Memory:
    Selective Attention:  Only vehicles within defined ROI lanes are counted,
                          focusing computational resources on relevant regions
                          (Broadbent, 1958 — filter model of attention).
    Working Memory:       A rolling observation buffer (MemoryBuffer) retains
                          recent traffic observations, enabling temporal context
                          for prediction — analogous to Baddeley's model of the
                          visuospatial sketchpad retaining recent scene states.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.utils import (
    CONFIG, classify_traffic_state, insert_traffic_observation,
    state_to_color_hex,
)


# ── Rolling memory buffer ─────────────────────────────────────────────────────
class MemoryBuffer:
    """
    Rolling circular buffer for recent traffic observations.

    Inspired by Baddeley's (2000) working memory model: just as human
    working memory holds a limited window of recent perceptual information
    for active processing, this buffer maintains the most recent N observations
    per lane, enabling temporal feature extraction for prediction.
    """

    def __init__(self, maxlen: int = 60):
        self._maxlen = maxlen
        self._buffers: Dict[str, deque] = {}

    def push(self, lane_id: str, observation: Dict[str, Any]):
        if lane_id not in self._buffers:
            self._buffers[lane_id] = deque(maxlen=self._maxlen)
        self._buffers[lane_id].append(observation)

    def get(self, lane_id: str, n: Optional[int] = None) -> List[Dict]:
        buf = list(self._buffers.get(lane_id, []))
        return buf[-n:] if n else buf

    def to_dataframe(self, lane_id: str) -> pd.DataFrame:
        data = self.get(lane_id)
        return pd.DataFrame(data) if data else pd.DataFrame()

    def mean(self, lane_id: str, field: str, window: int = 6) -> float:
        records = self.get(lane_id, window)
        vals = [r[field] for r in records if field in r]
        return float(np.mean(vals)) if vals else 0.0

    def lanes(self) -> List[str]:
        return list(self._buffers)

    def __len__(self):
        return sum(len(b) for b in self._buffers.values())


# ── Traffic analyser ──────────────────────────────────────────────────────────
class TrafficAnalyser:
    """
    Computes lane-wise traffic metrics from raw vehicle detection/tracking data.

    Capabilities:
        - Vehicle count per lane (in-frame and cumulative unique)
        - Traffic density [0–1] normalised by ROI area
        - Queue length estimate (metres, approximate)
        - Traffic state: FREE / MODERATE / CONGESTED / SEVERE
        - Approximate speed indicator (if positional data available)
        - Rolling observation history (MemoryBuffer)
        - Database persistence
    """

    ASSUMED_VEHICLE_SPACING_M = 5.0    # average stop-to-stop spacing
    ASSUMED_LANE_LENGTH_M     = 100.0  # estimated lane length in metres

    def __init__(self,
                 roi_config: Optional[Dict] = None,
                 db_conn=None,
                 history_window: int = 60):

        self.roi_config = roi_config or CONFIG["analysis"].get("roi_default", {})
        self.db_conn    = db_conn
        self.memory     = MemoryBuffer(maxlen=history_window)
        self._roi_areas = self._compute_roi_areas()
        self._state_history: List[Dict] = []
        logger.info(f"TrafficAnalyser initialised — lanes: {list(self.roi_config)}")

    # ── ROI area calculation ───────────────────────────────────────────────
    def _compute_roi_areas(self) -> Dict[str, float]:
        areas = {}
        for lane, pts in self.roi_config.items():
            arr = np.array(pts, np.float32)
            areas[lane] = float(cv2_poly_area(arr))
        return areas

    # ── Main analysis entry point ─────────────────────────────────────────
    def analyse(self,
                lane_counts: Dict[str, int],
                class_totals: Optional[Dict[str, int]] = None,
                timestamp: Optional[str] = None,
                source: str = "video",
                track_positions: Optional[Dict[int, Tuple[int, int]]] = None) -> Dict[str, Any]:
        """
        Compute full traffic metrics for the current frame/step.

        Args:
            lane_counts:     Dict lane_id → current vehicle count in ROI.
            class_totals:    Dict class_name → total count across all lanes.
            timestamp:       ISO-8601 timestamp (defaults to now).
            source:          Data source label ('video' / 'synthetic' / 'sumo').
            track_positions: Dict track_id → (cx, cy) for speed estimation.

        Returns:
            Dict with per-lane metrics and aggregate summary.
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        if class_totals is None:
            class_totals = {}

        lane_metrics: Dict[str, Dict] = {}

        for lane_id, count in lane_counts.items():
            density      = self._estimate_density(lane_id, count)
            queue_len_m  = self._estimate_queue_length(count)
            state        = classify_traffic_state(count)
            avg_speed    = self._estimate_speed(lane_id)

            metric = {
                "timestamp":     timestamp,
                "source":        source,
                "lane_id":       lane_id,
                "vehicle_count": count,
                "density":       round(density, 4),
                "queue_length":  round(queue_len_m, 2),
                "traffic_state": state,
                "avg_speed":     round(avg_speed, 2),
                "cars":          class_totals.get("car", 0),
                "buses":         class_totals.get("bus", 0),
                "trucks":        class_totals.get("truck", 0),
                "motorcycles":   class_totals.get("motorcycle", 0),
                "state_color":   state_to_color_hex(state),
            }
            lane_metrics[lane_id] = metric

            # Push to working-memory buffer
            self.memory.push(lane_id, metric)

            # Persist to database
            if self.db_conn is not None:
                try:
                    insert_traffic_observation(self.db_conn, metric)
                except Exception as e:
                    logger.warning(f"DB insert failed: {e}")

        aggregate = self._compute_aggregate(lane_metrics, class_totals, timestamp)
        self._state_history.append(aggregate)

        return {
            "lane_metrics": lane_metrics,
            "aggregate":    aggregate,
            "timestamp":    timestamp,
        }

    # ── Density estimation ────────────────────────────────────────────────
    def _estimate_density(self, lane_id: str, count: int) -> float:
        """
        Estimate occupancy density as fraction of ROI capacity.
        Uses pixel area of the ROI as a proxy for lane capacity.
        """
        area = self._roi_areas.get(lane_id, 1.0)
        # Approximate: each vehicle occupies ~3500 px² at typical overhead view
        vehicle_px = 3500.0
        capacity   = max(1, area / vehicle_px)
        return min(1.0, count / capacity)

    # ── Queue length estimation ───────────────────────────────────────────
    def _estimate_queue_length(self, count: int) -> float:
        """
        Approximate queue length in metres.
        Assumes average stop-to-stop vehicle spacing of 5 m.
        Disclaimer: without camera calibration this is indicative only.
        """
        return count * self.ASSUMED_VEHICLE_SPACING_M

    # ── Speed indicator ───────────────────────────────────────────────────
    def _estimate_speed(self, lane_id: str) -> float:
        """
        Heuristic speed indicator based on traffic state history.
        Note: without camera calibration, this is NOT a physical speed
        measurement and should not be reported as such.
        """
        recent = self.memory.get(lane_id, 3)
        if not recent:
            return 0.0
        avg_count = np.mean([r["vehicle_count"] for r in recent])
        thresholds = CONFIG["analysis"]["density_thresholds"]
        if avg_count <= thresholds["free"]:
            return 60.0     # approx km/h free-flow
        elif avg_count <= thresholds["moderate"]:
            return 40.0
        elif avg_count <= thresholds["congested"]:
            return 20.0
        return 10.0         # severe congestion

    # ── Aggregate metrics ─────────────────────────────────────────────────
    def _compute_aggregate(self, lane_metrics: Dict, class_totals: Dict,
                            timestamp: str) -> Dict[str, Any]:
        counts  = [m["vehicle_count"] for m in lane_metrics.values()]
        states  = [m["traffic_state"] for m in lane_metrics.values()]
        queues  = [m["queue_length"]  for m in lane_metrics.values()]
        speeds  = [m["avg_speed"]     for m in lane_metrics.values()]

        # Worst-case state across lanes
        order = {"FREE": 0, "MODERATE": 1, "CONGESTED": 2, "SEVERE": 3}
        worst_state = max(states, key=lambda s: order.get(s, 0), default="FREE")

        return {
            "timestamp":            timestamp,
            "total_vehicles":       sum(counts),
            "worst_state":          worst_state,
            "avg_density":          float(np.mean([m["density"] for m in lane_metrics.values()])),
            "total_queue_length_m": float(sum(queues)),
            "avg_speed_kmh":        float(np.mean(speeds)) if speeds else 0.0,
            "class_totals":         class_totals,
            "lane_count":           len(lane_metrics),
        }

    # ── Reporting ─────────────────────────────────────────────────────────
    def get_recent_dataframe(self, lane_id: str, n: int = 60) -> pd.DataFrame:
        """Return recent observations for a lane as a DataFrame."""
        return self.memory.to_dataframe(lane_id)

    def get_history_dataframe(self) -> pd.DataFrame:
        if not self._state_history:
            return pd.DataFrame()
        return pd.DataFrame(self._state_history)

    def congestion_summary(self) -> Dict[str, float]:
        """Percentage of time spent in each traffic state (all lanes, all time)."""
        all_states = []
        for lane_id in self.memory.lanes():
            all_states.extend([r["traffic_state"] for r in self.memory.get(lane_id)])
        if not all_states:
            return {s: 0.0 for s in ["FREE", "MODERATE", "CONGESTED", "SEVERE"]}
        total = len(all_states)
        from collections import Counter
        counts = Counter(all_states)
        return {s: round(counts.get(s, 0) / total * 100, 1)
                for s in ["FREE", "MODERATE", "CONGESTED", "SEVERE"]}


# ── Helper: polygon area without cv2 import ───────────────────────────────────
def cv2_poly_area(pts: np.ndarray) -> float:
    """Shoelace formula for polygon area in pixels²."""
    n = len(pts)
    if n < 3:
        return 1.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


# ── Synthetic time-series generator ──────────────────────────────────────────
def generate_synthetic_time_series(
        n_hours: int = 24,
        step_seconds: int = 30,
        lanes: Optional[List[str]] = None,
        seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic traffic time-series for model training.

    Pattern includes:
        - Morning peak (7–9 AM)
        - Evening peak (5–7 PM)
        - Night trough
        - Random noise
        - Day-of-week effect
        - Incident spikes

    Used as training data when UA-DETRAC or AI City Challenge data
    is not available due to access/licensing constraints.
    """
    rng = np.random.default_rng(seed)
    if lanes is None:
        lanes = ["lane1", "lane2"]

    steps_per_hour = 3600 // step_seconds
    total_steps    = n_hours * steps_per_hour
    records        = []

    base_time = pd.Timestamp("2025-01-06 06:00:00")  # Monday

    for step in range(total_steps):
        ts = base_time + pd.Timedelta(seconds=step * step_seconds)
        hour  = ts.hour
        dow   = ts.dayofweek  # 0=Mon

        # Base hourly profile (vehicles per observation step)
        if 7 <= hour <= 9:
            base = 22 - abs(hour + ts.minute / 60 - 8.0) * 6
        elif 12 <= hour <= 13:
            base = 14
        elif 17 <= hour <= 19:
            base = 20 - abs(hour + ts.minute / 60 - 18.0) * 5
        elif 23 <= hour or hour < 5:
            base = 2
        else:
            base = 8

        # Weekend reduction
        if dow >= 5:
            base *= 0.65

        base = max(0.0, base)

        for lane_idx, lane in enumerate(lanes):
            # Lane-specific variation
            lane_factor = 1.0 if lane_idx == 0 else rng.uniform(0.7, 1.3)
            count = max(0, int(rng.poisson(base * lane_factor)))
            # Rare incident spike
            if rng.random() < 0.003:
                count = min(35, count + int(rng.integers(8, 20)))

            records.append({
                "timestamp":     ts.isoformat(),
                "source":        "synthetic",
                "lane_id":       lane,
                "vehicle_count": count,
                "density":       min(1.0, count / 30.0),
                "queue_length":  count * 5.0,
                "traffic_state": classify_traffic_state(count),
                "cars":          max(0, int(count * rng.uniform(0.55, 0.70))),
                "buses":         max(0, int(count * rng.uniform(0.05, 0.12))),
                "trucks":        max(0, int(count * rng.uniform(0.05, 0.10))),
                "motorcycles":   max(0, int(count * rng.uniform(0.10, 0.20))),
                "avg_speed":     0.0,
            })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} synthetic observations "
                f"({n_hours}h × {len(lanes)} lanes, step={step_seconds}s)")
    return df
