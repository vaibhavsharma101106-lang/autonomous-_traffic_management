"""
SUMO Traffic Simulation Integration
IT 207 Human Intelligence Project

Generates SUMO network and route files, runs simulations via TraCI,
and evaluates signal controllers under controlled synthetic scenarios.

Note: Requires SUMO to be installed and SUMO_HOME environment variable set.
If SUMO is not installed, the module falls back to the software-only
SignalSimulator in src/optimization/signal_controller.py.
"""

from __future__ import annotations

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.utils import CONFIG, PROJECT_ROOT

SUMO_HOME = os.environ.get("SUMO_HOME", "")
SUMO_AVAILABLE = False

if SUMO_HOME:
    tools_path = Path(SUMO_HOME) / "tools"
    if tools_path.exists():
        sys.path.insert(0, str(tools_path))
        try:
            import traci
            import sumolib
            SUMO_AVAILABLE = True
            logger.info(f"SUMO available at {SUMO_HOME}")
        except ImportError:
            logger.warning("SUMO_HOME set but traci/sumolib not importable.")
else:
    logger.warning("SUMO_HOME not set — SUMO simulation unavailable. "
                   "Using software-only fallback.")


# ── SUMO network generator ────────────────────────────────────────────────────
class SUMONetworkGenerator:
    """
    Generates a minimal SUMO intersection network for simulation.
    Creates .nod.xml, .edg.xml, .net.xml, .rou.xml, and .sumocfg files.
    """

    def __init__(self, output_dir: Optional[str] = None):
        if output_dir is None:
            output_dir = str(PROJECT_ROOT / CONFIG["paths"]["sumo_config"])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_intersection(self,
                               scenario_name: str = "intersection",
                               vehicle_rate_vph: int = 600,
                               duration_s: int = 3600) -> str:
        """
        Generate a 4-way signalised intersection network.
        Returns path to .sumocfg file.
        """
        scenario_dir = self.output_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        self._write_nodes(scenario_dir)
        self._write_edges(scenario_dir)
        self._write_connections(scenario_dir)
        self._build_network(scenario_dir, scenario_name)
        self._write_routes(scenario_dir, scenario_name,
                           vehicle_rate_vph, duration_s)
        cfg_path = self._write_config(scenario_dir, scenario_name, duration_s)
        logger.info(f"SUMO scenario generated: {cfg_path}")
        return str(cfg_path)

    def _write_nodes(self, d: Path):
        content = """<?xml version="1.0" encoding="UTF-8"?>
<nodes>
    <node id="center" x="0"    y="0"    type="traffic_light"/>
    <node id="north"  x="0"    y="200"  type="priority"/>
    <node id="south"  x="0"    y="-200" type="priority"/>
    <node id="east"   x="200"  y="0"    type="priority"/>
    <node id="west"   x="-200" y="0"    type="priority"/>
</nodes>
"""
        (d / "nodes.nod.xml").write_text(content)

    def _write_edges(self, d: Path):
        content = """<?xml version="1.0" encoding="UTF-8"?>
<edges>
    <edge id="N_C" from="north"  to="center" numLanes="2" speed="13.89"/>
    <edge id="S_C" from="south"  to="center" numLanes="2" speed="13.89"/>
    <edge id="E_C" from="east"   to="center" numLanes="2" speed="13.89"/>
    <edge id="W_C" from="west"   to="center" numLanes="2" speed="13.89"/>
    <edge id="C_N" from="center" to="north"  numLanes="2" speed="13.89"/>
    <edge id="C_S" from="center" to="south"  numLanes="2" speed="13.89"/>
    <edge id="C_E" from="center" to="east"   numLanes="2" speed="13.89"/>
    <edge id="C_W" from="center" to="west"   numLanes="2" speed="13.89"/>
</edges>
"""
        (d / "edges.edg.xml").write_text(content)

    def _write_connections(self, d: Path):
        content = """<?xml version="1.0" encoding="UTF-8"?>
<connections>
    <connection from="N_C" to="C_S" fromLane="0" toLane="0"/>
    <connection from="N_C" to="C_E" fromLane="0" toLane="0"/>
    <connection from="N_C" to="C_W" fromLane="1" toLane="1"/>
    <connection from="S_C" to="C_N" fromLane="0" toLane="0"/>
    <connection from="S_C" to="C_E" fromLane="0" toLane="0"/>
    <connection from="S_C" to="C_W" fromLane="1" toLane="1"/>
    <connection from="E_C" to="C_W" fromLane="0" toLane="0"/>
    <connection from="E_C" to="C_N" fromLane="0" toLane="0"/>
    <connection from="E_C" to="C_S" fromLane="1" toLane="1"/>
    <connection from="W_C" to="C_E" fromLane="0" toLane="0"/>
    <connection from="W_C" to="C_N" fromLane="0" toLane="0"/>
    <connection from="W_C" to="C_S" fromLane="1" toLane="1"/>
</connections>
"""
        (d / "connections.con.xml").write_text(content)

    def _build_network(self, d: Path, name: str):
        if not SUMO_AVAILABLE:
            logger.warning("Skipping netconvert — SUMO not available.")
            (d / f"{name}.net.xml").write_text(
                "<!-- SUMO not installed — placeholder -->"
            )
            return
        cmd = [
            "netconvert",
            "--node-files", str(d / "nodes.nod.xml"),
            "--edge-files", str(d / "edges.edg.xml"),
            "--connection-files", str(d / "connections.con.xml"),
            "--output-file", str(d / f"{name}.net.xml"),
            "--no-warnings", "true",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception as e:
            logger.warning(f"netconvert failed: {e}")
            (d / f"{name}.net.xml").write_text(
                "<!-- netconvert failed — placeholder -->"
            )

    def _write_routes(self, d: Path, name: str,
                       vph: int, duration_s: int):
        """Generate Poisson-distributed vehicle arrivals."""
        rng  = np.random.default_rng(42)
        step_s = 1.0
        total_steps = duration_s

        routes_xml = '<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n'
        routes_xml += '''    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.89"/>
    <vType id="bus" accel="1.2" decel="3.0" sigma="0.5" length="12" maxSpeed="11.0"/>
    <vType id="truck" accel="1.0" decel="3.0" sigma="0.5" length="8" maxSpeed="10.0"/>
'''
        # Route definitions
        routes_xml += '''    <route id="NS" edges="N_C C_S"/>
    <route id="SN" edges="S_C C_N"/>
    <route id="EW" edges="E_C C_W"/>
    <route id="WE" edges="W_C C_E"/>
    <route id="NE" edges="N_C C_E"/>
    <route id="SW" edges="S_C C_W"/>
'''
        route_ids = ["NS", "SN", "EW", "WE", "NE", "SW"]
        veh_id = 0
        prob   = vph / 3600.0

        for t in range(total_steps):
            for route in route_ids:
                if rng.random() < prob / len(route_ids):
                    vtype = rng.choice(["car", "car", "car", "bus", "truck"],
                                       p=[0.75, 0.10, 0.05, 0.05, 0.05])
                    # Flatten choice to single string
                    vtype_str = str(vtype)
                    routes_xml += (f'    <vehicle id="v{veh_id}" type="{vtype_str}" '
                                   f'route="{route}" depart="{t}" />\n')
                    veh_id += 1

        routes_xml += "</routes>\n"
        (d / f"{name}.rou.xml").write_text(routes_xml)
        logger.info(f"Generated {veh_id} vehicles for scenario '{name}' "
                    f"({vph} vph, {duration_s}s)")

    def _write_config(self, d: Path, name: str, duration_s: int) -> Path:
        cfg = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{name}.net.xml"/>
        <route-files value="{name}.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{duration_s}"/>
        <step-length value="1.0"/>
    </time>
    <report>
        <no-warnings value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>
"""
        cfg_path = d / f"{name}.sumocfg"
        cfg_path.write_text(cfg)
        return cfg_path


# ── TraCI-based SUMO runner ────────────────────────────────────────────────────
class SUMORunner:
    """
    Runs a SUMO simulation via TraCI and collects per-step metrics.
    Falls back to SignalSimulator if SUMO is not installed.
    """

    def __init__(self, cfg_path: str, controller_type: str = "ai_assisted",
                 gui: bool = False):
        self.cfg_path        = cfg_path
        self.controller_type = controller_type
        self.gui             = gui
        self._metrics: List[Dict] = []

    def run(self, duration_s: int = 3600, warm_up_s: int = 300) -> Dict[str, Any]:
        if not SUMO_AVAILABLE:
            return self._fallback_run()

        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd    = [sumo_binary, "-c", self.cfg_path,
                       "--no-step-log", "true",
                       "--waiting-time-memory", "1000"]
        try:
            traci.start(sumo_cmd)
            return self._run_with_traci(duration_s, warm_up_s)
        except Exception as e:
            logger.error(f"SUMO TraCI error: {e}")
            try:
                traci.close()
            except Exception:
                pass
            return self._fallback_run()

    def _run_with_traci(self, duration_s: int, warm_up_s: int) -> Dict[str, Any]:
        from src.optimization.signal_controller import (
            FixedTimeController, RuleBasedController, AIAdaptiveController,
            LaneState,
        )
        lanes = ["lane1", "lane2", "lane3", "lane4"]
        ctrl_map = {
            "fixed":       FixedTimeController(lanes),
            "rule_based":  RuleBasedController(lanes),
            "ai_assisted": AIAdaptiveController(lanes),
        }
        controller = ctrl_map.get(self.controller_type,
                                   ctrl_map["ai_assisted"])

        total_wait  = 0.0
        total_veh   = 0
        step        = 0
        cycle_step  = 0
        cycle_dur   = 120  # default cycle length seconds

        while traci.simulation.getTime() < duration_s:
            traci.simulationStep()
            step += 1
            cycle_step += 1

            # Collect waiting time from SUMO
            try:
                vehicles  = traci.vehicle.getIDList()
                step_wait = sum(traci.vehicle.getWaitingTime(v) for v in vehicles)
                total_wait += step_wait
                total_veh  += len(vehicles)
            except Exception:
                pass

            if cycle_step >= cycle_dur:
                # Build LaneState objects from TraCI data
                lane_states = {}
                for i, edge_id in enumerate(["N_C", "S_C", "E_C", "W_C"], 1):
                    try:
                        count = traci.edge.getLastStepVehicleNumber(edge_id)
                        wait  = traci.edge.getWaitingTime(edge_id)
                    except Exception:
                        count, wait = 0, 0.0
                    lane_id = f"lane{i}"
                    lane_states[lane_id] = LaneState(
                        lane_id=lane_id,
                        vehicle_count=count,
                        waiting_time=float(wait),
                        queue_length=count * 5.0,
                    )

                cycle = controller.compute(lane_states)
                # Apply green durations via TraCI TLS (simplified)
                try:
                    tls_ids = traci.trafficlight.getIDList()
                    if tls_ids:
                        # Set all-green phase (simplified — real deployment uses proper phases)
                        traci.trafficlight.setPhase(tls_ids[0], 0)
                except Exception:
                    pass

                cycle_step = 0

        try:
            traci.close()
        except Exception:
            pass

        n_steps = max(step - warm_up_s, 1)
        return {
            "controller":       self.controller_type,
            "avg_waiting_time": round(total_wait / max(total_veh, 1), 2),
            "total_vehicles":   total_veh,
            "duration_s":       duration_s,
            "source":           "sumo_traci",
        }

    def _fallback_run(self) -> Dict[str, Any]:
        """Use software-only simulator when SUMO is unavailable."""
        logger.info("Using software-only SignalSimulator fallback.")
        from src.optimization.signal_controller import (
            FixedTimeController, RuleBasedController, AIAdaptiveController,
            SignalSimulator,
        )
        lanes = ["lane1", "lane2"]
        ctrl_map = {
            "fixed":       FixedTimeController(lanes),
            "rule_based":  RuleBasedController(lanes),
            "ai_assisted": AIAdaptiveController(lanes),
        }
        controller = ctrl_map.get(self.controller_type, ctrl_map["ai_assisted"])
        arrival    = {ln: 0.3 for ln in lanes}
        sim        = SignalSimulator(lanes, controller, arrival_rates=arrival)
        metrics    = sim.run(n_cycles=100)
        metrics["source"] = "software_fallback"
        return metrics


# ── Experiment orchestrator ───────────────────────────────────────────────────
def run_sumo_experiments(output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Run reproducible software simulations for all configured scenarios and
    controllers. SUMO output is not claimed by this experiment routine; the
    existing SUMO network lacks a verified signal phase plan.
    """
    from src.optimization.signal_controller import run_controller_comparison

    scenarios = CONFIG["experiments"]["scenarios"]
    lanes = ["lane1", "lane2", "lane3", "lane4"]
    rows: List[Dict] = []
    for scenario_index, scenario in enumerate(scenarios):
        total_rate = scenario["vehicle_rate_vph"] / 3600.0
        if scenario["name"] == "uneven_demand":
            shares = [0.55, 0.25, 0.15, 0.05]
        else:
            shares = [0.25] * len(lanes)
        arrival_rates = {lane: total_rate * share
                         for lane, share in zip(lanes, shares)}
        comparison = run_controller_comparison(
            lanes, arrival_rates, seed=42 + scenario_index,
            duration_seconds=int(scenario["duration_s"]),
        )
        for controller, metrics in comparison.items():
            rows.append({
                "scenario": scenario["name"],
                "controller": controller,
                "avg_waiting_time_s": metrics["avg_waiting_time"],
                "avg_queue_length_m_per_approach": metrics["avg_queue_length"],
                "throughput_vehicles_per_hour": metrics["throughput"],
                "congestion_rate": metrics["congestion_rate"],
                "simulation_seconds": metrics["simulation_seconds"],
                "source": "software_simulation",
                "seed": 42 + scenario_index,
            })

    df = pd.DataFrame(rows)
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "experiment_results.csv"
        df.to_csv(out_path, index=False)
    return df
