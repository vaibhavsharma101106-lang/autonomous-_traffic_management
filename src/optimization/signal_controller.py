"""
Module 5: Adaptive Traffic Signal Optimization
IT 207 Human Intelligence Project

Implements three signal controllers:
    A. Fixed-time controller (baseline)
    B. Rule-based adaptive controller
    C. AI-assisted adaptive controller (weighted demand scoring)

Human Intelligence Mapping — Decision-Making:
    Human decision-making involves evaluating multiple competing criteria
    before choosing an action (Kahneman, 2003). The adaptive controller
    mirrors this process: it evaluates current demand, queue length, waiting
    time, and predicted future load for each lane, weighs them according to
    a configured priority vector, and selects the optimal green duration —
    subject to fairness and safety constraints.

    This is analogous to how a traffic officer at an intersection mentally
    assesses which lane has the most urgent need, balancing fairness to all
    directions, and allocating green time proportionally.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.utils import CONFIG, classify_traffic_state


# ── Phase and signal state definitions ───────────────────────────────────────
PHASE_GREEN  = "GREEN"
PHASE_YELLOW = "YELLOW"
PHASE_RED    = "RED"
PHASE_ALL_RED = "ALL_RED"


@dataclass
class LaneState:
    """Current state of a single traffic lane / approach."""
    lane_id:       str
    vehicle_count: int   = 0
    queue_length:  float = 0.0   # metres
    waiting_time:  float = 0.0   # seconds (accumulated since last green)
    predicted_count: float = 0.0 # from prediction module
    current_phase: str  = PHASE_RED
    green_duration: float = 30.0
    cycles_since_green: int = 0


@dataclass
class SignalCycle:
    """Represents one complete signal cycle across all lanes."""
    cycle_number: int
    assignments: Dict[str, float]  # lane_id → green_duration (s)
    total_cycle_time: float
    controller_type: str
    reasoning: Dict[str, str] = field(default_factory=dict)


# ── Fixed-time controller ─────────────────────────────────────────────────────
class FixedTimeController:
    """
    Traditional fixed-time signal controller.

    Allocates equal green time to all lanes in round-robin fashion.
    Used as the performance baseline.
    """

    def __init__(self, lanes: List[str], green_per_lane: float = 30.0):
        cfg = CONFIG["optimization"]
        self.lanes          = lanes
        self.green_per_lane = max(cfg["min_green"],
                                  min(green_per_lane, cfg["max_green"]))
        self.yellow         = cfg["yellow_duration"]
        self.all_red        = cfg["all_red_duration"]
        self.cycle_num      = 0
        self._current_idx   = 0

    def compute(self, lane_states: Dict[str, LaneState]) -> SignalCycle:
        self.cycle_num += 1
        assignments = {lane: self.green_per_lane for lane in self.lanes}
        total = (self.green_per_lane + self.yellow + self.all_red) * len(self.lanes)
        return SignalCycle(
            cycle_number=self.cycle_num,
            assignments=assignments,
            total_cycle_time=total,
            controller_type="fixed",
            reasoning={lane: f"Fixed {self.green_per_lane}s" for lane in self.lanes},
        )

    @property
    def name(self) -> str:
        return "Fixed-Time Controller"


# ── Rule-based adaptive controller ───────────────────────────────────────────
class RuleBasedController:
    """
    Rule-based adaptive signal controller.

    Applies heuristic rules to scale green time proportionally to
    lane vehicle counts and waiting time penalties. Respects minimum /
    maximum green durations and fairness constraints.
    """

    def __init__(self, lanes: List[str]):
        cfg = CONFIG["optimization"]
        self.lanes     = lanes
        self.min_green = cfg["min_green"]
        self.max_green = cfg["max_green"]
        self.yellow    = cfg["yellow_duration"]
        self.all_red   = cfg["all_red_duration"]
        self.cycle_num = 0
        self._total_budget = cfg.get("cycle_max", 240)

    def compute(self, lane_states: Dict[str, LaneState]) -> SignalCycle:
        self.cycle_num += 1
        scores    = {}
        reasoning = {}

        for lane_id in self.lanes:
            state = lane_states.get(lane_id, LaneState(lane_id=lane_id))
            # Score = vehicles + queue-time penalty + starvation prevention
            score = state.vehicle_count
            if state.waiting_time > 60:
                score += (state.waiting_time - 60) * 0.1
            if state.cycles_since_green > 3:
                score *= (1 + 0.15 * state.cycles_since_green)
            scores[lane_id]    = max(0.01, score)
            reasoning[lane_id] = (f"count={state.vehicle_count}, "
                                   f"wait={state.waiting_time:.0f}s, "
                                   f"score={score:.1f}")

        total_score = sum(scores.values())
        # Phase budget: total - overhead
        overhead = (self.yellow + self.all_red) * len(self.lanes)
        green_budget = max(
            self.min_green * len(self.lanes),
            self._total_budget - overhead
        )

        assignments = {}
        for lane_id in self.lanes:
            raw = (scores[lane_id] / total_score) * green_budget
            assignments[lane_id] = float(
                max(self.min_green, min(self.max_green, raw))
            )

        total = sum(assignments.values()) + overhead
        return SignalCycle(
            cycle_number=self.cycle_num,
            assignments=assignments,
            total_cycle_time=total,
            controller_type="rule_based",
            reasoning=reasoning,
        )

    @property
    def name(self) -> str:
        return "Rule-Based Adaptive Controller"


# ── AI-assisted adaptive controller ──────────────────────────────────────────
class AIAdaptiveController:
    """
    AI-Assisted Adaptive Traffic Signal Controller.

    Combines current demand, queue pressure, waiting-time urgency, and
    predicted future demand into a weighted multi-criteria score.
    Green time is allocated proportionally to lane scores, respecting
    hard safety and fairness constraints.

    Human Intelligence Mapping — Decision-Making:
        Like a human traffic officer who mentally weighs multiple factors
        before waving a lane through, this controller:
        1. EVALUATES  — demand, queue, waiting time, predicted load
        2. WEIGHS     — configurable priority vector (w_demand, w_wait, w_fair)
        3. DECIDES    — optimal green duration per lane
        4. CONSTRAINS — enforces minimum green (safety), maximum green (fairness),
                        and starvation prevention (no lane waits indefinitely)
    """

    def __init__(self, lanes: List[str],
                 demand_weight: Optional[float] = None,
                 wait_weight:   Optional[float] = None,
                 fairness_weight: Optional[float] = None):
        cfg = CONFIG["optimization"]
        self.lanes    = lanes
        self.w_demand = cfg["demand_weight"] if demand_weight is None else demand_weight
        self.w_wait   = cfg["wait_weight"] if wait_weight is None else wait_weight
        self.w_fair   = cfg["fairness_weight"] if fairness_weight is None else fairness_weight
        self.min_green = cfg["min_green"]
        self.max_green = cfg["max_green"]
        self.yellow   = cfg["yellow_duration"]
        self.all_red  = cfg["all_red_duration"]
        self.max_wait_penalty = cfg.get("max_wait_penalty", 120)
        self.cycle_budget = cfg.get("cycle_max", 240)
        self.cycle_num = 0
        self._history: List[SignalCycle] = []

    def compute(self, lane_states: Dict[str, LaneState],
                predictions: Optional[Dict[str, float]] = None) -> SignalCycle:
        """
        Compute optimal green allocations for this cycle.

        Args:
            lane_states:  Current per-lane states.
            predictions:  Optional dict lane_id → predicted_count (next 5 min).
        """
        self.cycle_num += 1
        if predictions is None:
            predictions = {}

        scores    = {}
        reasoning = {}

        for lane_id in self.lanes:
            state = lane_states.get(lane_id, LaneState(lane_id=lane_id))
            pred  = predictions.get(lane_id, float(state.vehicle_count))

            # --- Demand component ---
            # Blend current count with predicted future demand
            demand_score = 0.6 * state.vehicle_count + 0.4 * pred

            # --- Waiting-time urgency ---
            wait_score = min(state.waiting_time / 30.0, 4.0)
            # Emergency override: if wait exceeds threshold, force high priority
            if state.waiting_time > self.max_wait_penalty:
                wait_score = 5.0

            # --- Queue pressure ---
            queue_score = state.queue_length / max(
                CONFIG["analysis"].get("queue_depth_m", 5.0) * 10, 1
            )

            # --- Starvation prevention (fairness) ---
            starvation = state.cycles_since_green * 0.2

            # --- Weighted composite ---
            composite = (
                self.w_demand * demand_score
                + self.w_wait  * wait_score
                + self.w_fair  * starvation
                + 0.1          * queue_score
            )
            scores[lane_id] = max(0.01, composite)
            reasoning[lane_id] = (
                f"demand={demand_score:.1f}, wait_urgency={wait_score:.2f}, "
                f"queue={queue_score:.2f}, starvation={starvation:.2f} → "
                f"score={composite:.3f}"
            )

        # Normalise and allocate green time
        total_score = sum(scores.values())
        overhead    = (self.yellow + self.all_red) * len(self.lanes)
        green_budget = max(
            self.min_green * len(self.lanes),
            self.cycle_budget - overhead
        )

        assignments = {}
        for lane_id in self.lanes:
            raw_alloc = (scores[lane_id] / total_score) * green_budget
            # Enforce min/max
            clamped = float(max(self.min_green,
                                min(self.max_green, raw_alloc)))
            assignments[lane_id] = clamped

        total_cycle = sum(assignments.values()) + overhead
        cycle = SignalCycle(
            cycle_number=self.cycle_num,
            assignments=assignments,
            total_cycle_time=total_cycle,
            controller_type="ai_assisted",
            reasoning=reasoning,
        )
        self._history.append(cycle)
        logger.debug(f"[AI Controller] Cycle {self.cycle_num}: {assignments}")
        return cycle

    @property
    def name(self) -> str:
        return "AI-Assisted Adaptive Controller"

    def history_df(self):
        """Return cycle history as a DataFrame."""
        import pandas as pd
        rows = []
        for c in self._history:
            for lane, duration in c.assignments.items():
                rows.append({
                    "cycle": c.cycle_number,
                    "lane":  lane,
                    "green": duration,
                    "total_cycle": c.total_cycle_time,
                    "controller": c.controller_type,
                })
        return pd.DataFrame(rows)


# ── Signal simulator (software only) ─────────────────────────────────────────
class SignalSimulator:
    """
    Software-only traffic signal simulator.

    Advances a simulated intersection through multiple signal cycles,
    accumulating waiting times and queue lengths for each controller type.
    Provides data for comparing fixed vs. rule-based vs. AI controllers.
    """

    def __init__(self, lanes: List[str],
                 controller: Any,
                 arrival_rates: Optional[Dict[str, float]] = None,
                 saturation_flow: float = 0.5,
                 seed: int = 42):
        """
        Args:
            lanes:            List of lane IDs.
            controller:       Controller instance (Fixed / RuleBased / AI).
            arrival_rates:    Vehicles/second per lane (default 0.2 each).
            saturation_flow:  Vehicles departing per second of green per lane.
        """
        self.lanes            = lanes
        self.controller       = controller
        self.saturation_flow  = saturation_flow
        if arrival_rates is None:
            arrival_rates = {ln: 0.2 for ln in lanes}
        self.arrival_rates    = arrival_rates
        self.rng              = np.random.default_rng(seed)

        # Per-lane state
        self.lane_states: Dict[str, LaneState] = {
            ln: LaneState(lane_id=ln) for ln in lanes
        }
        self.total_waiting  = {ln: 0.0 for ln in lanes}
        self.total_departed = {ln: 0   for ln in lanes}
        self.total_arrivals_by_lane = {ln: 0 for ln in lanes}
        self._service_credit = {ln: 0.0 for ln in lanes}
        self.cycle_count    = 0
        self.elapsed_seconds = 0
        self.total_arrivals = 0
        self.queue_vehicle_seconds = 0.0
        self.lane_queue_vehicle_seconds = {ln: 0.0 for ln in lanes}
        self.congested_lane_seconds = 0

    def run(self, n_cycles: int = 100,
            duration_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Simulate a fixed cycle count or stop after a shared duration."""
        while (self.cycle_count < n_cycles if duration_seconds is None
               else self.elapsed_seconds < duration_seconds):
            cycle = self.controller.compute(self.lane_states)
            self.cycle_count += 1
            # Execute serialized signal phases second by second. Arrivals are
            # sampled from the same seeded Poisson process for every run.
            yellow = int(CONFIG["optimization"]["yellow_duration"])
            all_red = int(CONFIG["optimization"]["all_red_duration"])
            active = {lane: max(0, int(round(cycle.assignments.get(lane, 30.0))))
                      for lane in self.lanes}
            phase_lengths = {lane: active[lane] + yellow + all_red for lane in self.lanes}
            cycle_duration = sum(phase_lengths.values())
            green_start = {}
            cursor = 0
            for lane in self.lanes:
                green_start[lane] = cursor
                cursor += phase_lengths[lane]

            for offset in range(cycle_duration):
                for lane_id in self.lanes:
                    state = self.lane_states[lane_id]
                    arrivals = int(self.rng.poisson(max(0.0, self.arrival_rates.get(lane_id, 0.0))))
                    state.vehicle_count += arrivals
                    self.total_arrivals += arrivals
                    self.total_arrivals_by_lane[lane_id] += arrivals
                    in_green = green_start[lane_id] <= offset < green_start[lane_id] + active[lane_id]
                    if in_green:
                        self._service_credit[lane_id] += self.saturation_flow
                    available = int(self._service_credit[lane_id])
                    departed = min(state.vehicle_count, available) if in_green else 0
                    self._service_credit[lane_id] -= departed
                    state.vehicle_count -= departed
                    self.total_departed[lane_id] += departed
                    if state.vehicle_count > 0:
                        self.queue_vehicle_seconds += state.vehicle_count
                        self.lane_queue_vehicle_seconds[lane_id] += state.vehicle_count
                        self.total_waiting[lane_id] += state.vehicle_count
                    if state.vehicle_count > CONFIG["analysis"]["density_thresholds"]["moderate"]:
                        self.congested_lane_seconds += 1
                    state.current_phase = PHASE_GREEN if in_green else PHASE_RED
                self.elapsed_seconds += 1
                if duration_seconds is not None and self.elapsed_seconds >= duration_seconds:
                    break

            for lane_id in self.lanes:
                state = self.lane_states[lane_id]
                green_dur = active[lane_id]
                state.queue_length = state.vehicle_count * 5.0
                state.waiting_time += cycle_duration - green_dur
                if self.total_departed[lane_id] > 0:
                    state.cycles_since_green = 0
                    state.waiting_time = 0.0
                else:
                    state.cycles_since_green += 1
                state.green_duration = green_dur

        # Compute aggregate metrics
        return self._compute_metrics(self.cycle_count)

    def _compute_metrics(self, n_cycles: int) -> Dict[str, Any]:
        completed = sum(self.total_departed.values())
        avg_wait = self.queue_vehicle_seconds / max(completed, 1)
        avg_queue = (self.queue_vehicle_seconds /
                     max(self.elapsed_seconds * len(self.lanes), 1) * 5.0)
        throughput = sum(self.total_departed.values()) * 3600 / max(self.elapsed_seconds, 1)

        # Approximate congestion rate: fraction of lanes with count > threshold
        congestion_rate = (self.congested_lane_seconds /
                           max(self.elapsed_seconds * len(self.lanes), 1))

        return {
            "controller":        getattr(self.controller, "name", "unknown"),
            "n_cycles":          n_cycles,
            "simulation_seconds": self.elapsed_seconds,
            "avg_waiting_time":  round(float(avg_wait), 2),
            "avg_queue_length":  round(float(avg_queue), 2),
            "throughput":        round(float(throughput), 2),
            "congestion_rate":   round(congestion_rate, 3),
            "lane_waiting":      {ln: round(self.lane_queue_vehicle_seconds[ln] / max(self.total_departed[ln], 1), 2)
                                  for ln in self.lanes},
            "lane_departed":     dict(self.total_departed),
        }


# ── Utility: run comparison experiment ───────────────────────────────────────
def run_controller_comparison(lanes: List[str],
                               arrival_rates: Dict[str, float],
                               n_cycles: int = 100,
                               seed: int = 42,
                               duration_seconds: Optional[int] = None) -> Dict[str, Dict]:
    """
    Run all three controllers under identical traffic conditions and
    return their performance metrics for direct comparison.
    """
    import pandas as pd

    controllers = {
        "fixed":      FixedTimeController(lanes, green_per_lane=30.0),
        "rule_based": RuleBasedController(lanes),
        "ai_assisted": AIAdaptiveController(lanes),
    }

    results = {}
    for name, ctrl in controllers.items():
        sim     = SignalSimulator(lanes, ctrl, arrival_rates=arrival_rates, seed=seed)
        metrics = sim.run(n_cycles=n_cycles, duration_seconds=duration_seconds)
        results[name] = metrics
        logger.info(f"[{name}] wait={metrics['avg_waiting_time']}s, "
                    f"queue={metrics['avg_queue_length']}m, "
                    f"throughput={metrics['throughput']} veh/hour")

    return results
