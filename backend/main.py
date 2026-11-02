"""
FastAPI Backend — AI Traffic Management System
IT 207 Human Intelligence Project

Provides REST API endpoints for:
- Video upload and processing
- Real-time traffic analysis
- Congestion predictions
- Signal optimization recommendations
- Experiment results
- Database queries
"""

from __future__ import annotations

import io
import json
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import CONFIG, PROJECT_ROOT, init_database, get_traffic_history
from src.analysis.traffic_analyser import generate_synthetic_time_series, TrafficAnalyser
from src.prediction.predictor import (
    RandomForestPredictor, XGBoostPredictor, LSTMPredictor,
    train_all_models, evaluate_model
)
from src.optimization.signal_controller import (
    FixedTimeController, RuleBasedController, AIAdaptiveController,
    LaneState, run_controller_comparison,
)

# ── Application setup ─────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Traffic Management API",
    description="IT 207 Human Intelligence Project — Traffic Analysis & Optimization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
DB_CONN = init_database()
UPLOAD_DIR  = PROJECT_ROOT / "data" / "raw" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR  = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Lazy-loaded models
_detector  = None
_tracker   = None
_analyser  = None
_xgb_model = None
_lstm_model = None
_rf_model   = None


def get_detector():
    global _detector
    if _detector is None:
        from src.detection.detector import VehicleDetector
        _detector = VehicleDetector()
    return _detector


def get_tracker():
    global _tracker
    if _tracker is None:
        from src.tracking.tracker import VehicleTracker
        _tracker = VehicleTracker()
    return _tracker


def get_analyser():
    global _analyser
    if _analyser is None:
        _analyser = TrafficAnalyser(db_conn=DB_CONN)
    return _analyser


def _load_or_train_models():
    global _xgb_model, _rf_model, _lstm_model
    model_dir = PROJECT_ROOT / "models" / "prediction"

    # Try loading persisted models
    xgb_path  = str(model_dir / "xgboost")
    rf_path   = str(model_dir / "random_forest")
    lstm_path = str(model_dir / "lstm")

    try:
        _xgb_model = XGBoostPredictor()
        _xgb_model.load(xgb_path)
        _rf_model = RandomForestPredictor()
        _rf_model.load(rf_path)
        _lstm_model = LSTMPredictor()
        _lstm_model.load(lstm_path)
        logger.info("Loaded persisted prediction models.")
    except Exception:
        logger.warning("Saved models not found — will train on-demand.")


@app.on_event("startup")
async def startup():
    _load_or_train_models()
    logger.info("API server started.")


# ── Health & status ───────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "AI Traffic Management API v1.0.0", "status": "running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "xgboost": _xgb_model is not None and _xgb_model.trained,
            "lstm":    _lstm_model is not None and _lstm_model.trained,
            "rf":      _rf_model is not None and _rf_model.trained,
        },
    }


# ── Video processing ──────────────────────────────────────────────────────────
class VideoProcessRequest(BaseModel):
    sample_rate: int = 5
    max_frames: Optional[int] = 200
    enable_tracking: bool = True


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a traffic video for processing."""
    video_id  = str(uuid.uuid4())[:8]
    suffix    = Path(file.filename).suffix or ".mp4"
    dest_path = UPLOAD_DIR / f"{video_id}{suffix}"

    with open(dest_path, "wb") as f:
        content = await file.read()
        f.write(content)

    logger.info(f"Video uploaded: {dest_path} ({len(content)/1024:.1f} KB)")
    return {
        "video_id":    video_id,
        "filename":    file.filename,
        "path":        str(dest_path),
        "size_bytes":  len(content),
    }


@app.post("/api/video/analyse/{video_id}")
def analyse_video(video_id: str,
                  sample_rate: int = Query(5),
                  max_frames: int = Query(100)):
    """Run detection + tracking on an uploaded video."""
    # Find uploaded file
    matches = list(UPLOAD_DIR.glob(f"{video_id}*"))
    if not matches:
        raise HTTPException(404, f"Video {video_id} not found")

    video_path  = matches[0]
    output_path = OUTPUT_DIR / f"{video_id}_annotated.mp4"

    detector = get_detector()
    analyser = get_analyser()

    from src.detection.detector import process_video_frames
    results = process_video_frames(
        str(video_path), detector,
        sample_rate=sample_rate,
        max_frames=max_frames,
        output_path=str(output_path),
    )

    # Aggregate per-frame analysis
    all_lane_data: Dict[str, List[int]] = {}
    for r in results:
        analysis = analyser.analyse(
            lane_counts=r["lane_counts"],
            class_totals=r["class_totals"],
            source="video",
        )
        for lane, m in analysis["lane_metrics"].items():
            all_lane_data.setdefault(lane, []).append(m["vehicle_count"])

    lane_summary = {
        lane: {
            "avg_count": round(float(np.mean(counts)), 1),
            "max_count": int(np.max(counts)),
            "total_observations": len(counts),
        }
        for lane, counts in all_lane_data.items()
    }

    return {
        "video_id":      video_id,
        "frames_analysed": len(results),
        "avg_latency_ms": detector.avg_latency_ms,
        "lane_summary":  lane_summary,
        "aggregate":     analyser.congestion_summary(),
        "output_video":  str(output_path) if output_path.exists() else None,
    }


# ── Traffic data ──────────────────────────────────────────────────────────────
@app.get("/api/traffic/history")
def get_history(lane_id: Optional[str] = None, limit: int = 200):
    """Retrieve recent traffic observations."""
    df = get_traffic_history(DB_CONN, lane_id=lane_id, limit=limit)
    return {"records": df.to_dict(orient="records"), "count": len(df)}


@app.get("/api/traffic/summary")
def traffic_summary():
    """Return current traffic state summary."""
    analyser = get_analyser()
    return {
        "congestion_summary": analyser.congestion_summary(),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/traffic/synthetic")
def generate_synthetic(n_hours: int = Query(24), seed: int = Query(42)):
    """Generate and store synthetic traffic data."""
    df = generate_synthetic_time_series(n_hours=n_hours, seed=seed)
    # Store in database
    inserted = 0
    from src.utils import insert_traffic_observation
    for _, row in df.iterrows():
        insert_traffic_observation(DB_CONN, row.to_dict())
        inserted += 1
    # Save to CSV
    csv_path = PROJECT_ROOT / "data" / "synthetic" / "traffic_series.csv"
    df.to_csv(csv_path, index=False)
    return {
        "message": f"Generated {inserted} synthetic observations",
        "hours": n_hours,
        "csv_path": str(csv_path),
    }


# ── Models ────────────────────────────────────────────────────────────────────
@app.post("/api/models/train")
def train_models(n_hours: int = Query(72)):
    """Train all prediction models on synthetic + stored data."""
    global _xgb_model, _rf_model, _lstm_model

    # Load or generate data
    df = get_traffic_history(DB_CONN, limit=10000)
    if len(df) < 100:
        synth = generate_synthetic_time_series(n_hours=n_hours, seed=42)
        df    = pd.concat([df, synth], ignore_index=True)

    df = df[df["lane_id"] == "lane1"].reset_index(drop=True)

    if len(df) < 50:
        raise HTTPException(400, "Insufficient data for training.")

    n    = len(df)
    n_tr = int(n * 0.70)
    n_va = int(n * 0.15)
    train_df = df.iloc[:n_tr]
    val_df   = df.iloc[n_tr:n_tr + n_va]
    test_df  = df.iloc[n_tr + n_va:]

    results = train_all_models(train_df, val_df, test_df)
    _xgb_model  = results["XGBoost"]["model"]
    _rf_model   = results["RandomForest"]["model"]
    _lstm_model = results["LSTM"]["model"]

    comp = results.get("test_comparison")
    comparison = comp.to_dict() if comp is not None and not comp.empty else {}

    return {
        "status": "trained",
        "train_samples": len(train_df),
        "val_samples":   len(val_df),
        "test_samples":  len(test_df),
        "test_comparison": comparison,
    }


@app.post("/api/predict/{lane_id}")
def predict_traffic(lane_id: str,
                    model_name: str = Query("xgboost"),
                    horizon: int = Query(6)):
    """Predict future vehicle counts for a lane."""
    df = get_traffic_history(DB_CONN, lane_id=lane_id, limit=200)
    if df.empty:
        synth = generate_synthetic_time_series(n_hours=2, seed=99)
        df    = synth[synth["lane_id"] == lane_id]

    model_map = {
        "xgboost": _xgb_model,
        "lstm":    _lstm_model,
        "rf":      _rf_model,
    }
    model = model_map.get(model_name)
    if model is None or not getattr(model, "trained", False):
        raise HTTPException(400, f"Model '{model_name}' not trained. Call /api/models/train first.")

    try:
        predictions = model.predict(df)
        preds_list  = [max(0, float(p)) for p in predictions[-horizon:]]
        states      = [
            "FREE" if p <= 5 else
            "MODERATE" if p <= 15 else
            "CONGESTED" if p <= 25 else "SEVERE"
            for p in preds_list
        ]
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    return {
        "lane_id":    lane_id,
        "model":      model_name,
        "horizon":    horizon,
        "predictions": preds_list,
        "states":      states,
        "timestamp":  datetime.now().isoformat(),
    }


# ── Signal optimization ───────────────────────────────────────────────────────
class SignalOptimizationRequest(BaseModel):
    lanes: List[str] = ["lane1", "lane2"]
    lane_counts: Dict[str, int] = {}
    waiting_times: Dict[str, float] = {}
    queue_lengths: Dict[str, float] = {}
    controller_type: str = "ai_assisted"


@app.post("/api/optimize/signal")
def optimize_signal(req: SignalOptimizationRequest):
    """Compute optimal signal durations for the given lane states."""
    ctrl_map = {
        "fixed":       FixedTimeController(req.lanes),
        "rule_based":  RuleBasedController(req.lanes),
        "ai_assisted": AIAdaptiveController(req.lanes),
    }
    controller = ctrl_map.get(req.controller_type, ctrl_map["ai_assisted"])

    lane_states = {}
    for lane in req.lanes:
        lane_states[lane] = LaneState(
            lane_id=lane,
            vehicle_count=req.lane_counts.get(lane, 0),
            waiting_time=req.waiting_times.get(lane, 0.0),
            queue_length=req.queue_lengths.get(lane, 0.0),
        )

    cycle = controller.compute(lane_states)
    return {
        "cycle_number":      cycle.cycle_number,
        "controller":        req.controller_type,
        "assignments":       cycle.assignments,
        "total_cycle_time":  cycle.total_cycle_time,
        "reasoning":         cycle.reasoning,
    }


@app.post("/api/optimize/compare")
def compare_controllers(
    vehicle_rate: int = Query(600),
    n_cycles: int = Query(100)
):
    """Compare all three controllers under identical conditions."""
    lanes         = ["lane1", "lane2", "lane3", "lane4"]
    arrival_rates = {ln: vehicle_rate / 3600.0 for ln in lanes}
    results       = run_controller_comparison(lanes, arrival_rates, n_cycles)
    return results


# ── Experiments ───────────────────────────────────────────────────────────────
@app.post("/api/experiments/run")
def run_experiments():
    """Run the full SUMO experiment battery."""
    from src.simulation.sumo_runner import run_sumo_experiments
    results_dir = str(PROJECT_ROOT / "experiments" / "results")
    df = run_sumo_experiments(results_dir)
    csv_path = Path(results_dir) / "sumo_experiments.csv"
    df.to_csv(csv_path, index=False)
    return {
        "message": "Experiments complete",
        "rows":    len(df),
        "results": df.to_dict(orient="records"),
    }


@app.get("/api/experiments/results")
def get_experiment_results():
    """Return previously computed experiment results."""
    csv_path = PROJECT_ROOT / "experiments" / "results" / "sumo_experiments.csv"
    if not csv_path.exists():
        return {"results": [], "message": "No results yet. Call POST /api/experiments/run"}
    df = pd.read_csv(csv_path)
    return {"results": df.to_dict(orient="records"), "rows": len(df)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=CONFIG["api"]["host"], port=CONFIG["api"]["port"])
