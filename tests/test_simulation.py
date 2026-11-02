"""Regression checks for the software traffic simulator and controllers."""

import unittest

from src.optimization.signal_controller import (
    AIAdaptiveController,
    LaneState,
    SignalSimulator,
    run_controller_comparison,
)


class SimulationTests(unittest.TestCase):
    def test_ai_controller_respects_green_bounds(self):
        lanes = ["north", "south", "east", "west"]
        controller = AIAdaptiveController(lanes)
        states = {
            lane: LaneState(lane, vehicle_count=count, waiting_time=wait,
                            queue_length=count * 5.0)
            for lane, count, wait in zip(lanes, [30, 12, 5, 1], [90, 30, 10, 5])
        }
        cycle = controller.compute(states)
        self.assertEqual(set(cycle.assignments), set(lanes))
        self.assertTrue(all(10 <= seconds <= 90
                            for seconds in cycle.assignments.values()))

    def test_fixed_duration_and_seeded_repeatability(self):
        lanes = ["north", "south", "east", "west"]
        rates = {lane: 0.04 for lane in lanes}
        first = run_controller_comparison(lanes, rates, seed=15,
                                          duration_seconds=900)
        second = run_controller_comparison(lanes, rates, seed=15,
                                           duration_seconds=900)
        self.assertEqual(first, second)
        for metrics in first.values():
            self.assertEqual(metrics["simulation_seconds"], 900)
            self.assertGreater(metrics["throughput"], 0)
            self.assertGreaterEqual(metrics["avg_queue_length"], 0)
            self.assertGreaterEqual(metrics["congestion_rate"], 0)
            self.assertLessEqual(metrics["congestion_rate"], 1)

    def test_arrivals_and_departures_are_accounted_for(self):
        lanes = ["lane1", "lane2"]
        controller = AIAdaptiveController(lanes)
        simulation = SignalSimulator(
            lanes, controller, arrival_rates={"lane1": 0.1, "lane2": 0.05},
            seed=4,
        )
        result = simulation.run(duration_seconds=600)
        self.assertEqual(result["simulation_seconds"], 600)
        self.assertGreater(sum(result["lane_departed"].values()), 0)
        self.assertLessEqual(
            sum(result["lane_departed"].values()), simulation.total_arrivals
        )


if __name__ == "__main__":
    unittest.main()
