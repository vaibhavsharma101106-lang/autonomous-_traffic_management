"""Leakage-aware lane-arrival forecasting with explicit data provenance.

The target is the number of vehicles first tracked in a lane during the next
60 seconds. Training inputs are aggregates of per-video observations; synthetic
benchmark data is intentionally separate from recorded-video data.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

FEATURE_COLUMNS = [
    "lane_code", "arrivals_lag_1", "arrivals_lag_2", "arrivals_lag_3",
    "arrivals_mean_3", "queue_length_m", "hour_sin", "hour_cos",
]
TARGET_COLUMN = "arrivals_next_60s"
MODEL_VERSION = "1.0.0"


def build_arrival_dataset(observations: pd.DataFrame, interval_seconds: int = 5,
                          horizon_seconds: int = 60) -> pd.DataFrame:
    """Convert persisted observations into per-lane, future-arrival samples.

    Each analysis job is treated as one recording session. The future target
    covers the next complete horizon; partial horizons at a clip boundary are
    discarded. The dataset builder never combines adjacent uploads.
    """
    required = {"analysis_job_id", "timestamp", "lane_id", "arrival_count",
                "queue_length", "tracking_enabled", "source"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Traffic observations are missing columns: {sorted(missing)}")
    if interval_seconds <= 0 or horizon_seconds <= 0 or horizon_seconds % interval_seconds:
        raise ValueError("horizon_seconds must be a positive multiple of interval_seconds")
    if observations.empty:
        return pd.DataFrame(columns=[*FEATURE_COLUMNS, TARGET_COLUMN, "session_id",
                                     "session_start", "sample_time", "target_end", "lane_id", "source"])

    horizon_steps = horizon_seconds // interval_seconds
    records = observations.copy()
    records = records[records["analysis_job_id"].notna()]
    records = records[records["tracking_enabled"].astype(bool)]
    records["timestamp"] = pd.to_datetime(records["timestamp"], errors="coerce")
    records = records.dropna(subset=["timestamp", "lane_id", "source"])
    records["arrival_count"] = pd.to_numeric(records["arrival_count"], errors="coerce").fillna(0).clip(lower=0)
    records["queue_length"] = pd.to_numeric(records["queue_length"], errors="coerce").fillna(0).clip(lower=0)
    lanes = sorted(records["lane_id"].astype(str).unique())
    lane_codes = {lane: index for index, lane in enumerate(lanes)}
    rows = []

    for (session_id, lane_id), group in records.groupby(["analysis_job_id", "lane_id"], sort=False):
        group = group.sort_values("timestamp").set_index("timestamp")
        interval = f"{interval_seconds}s"
        series = group.resample(interval, origin="start").agg(
            arrivals=("arrival_count", "sum"), queue_length_m=("queue_length", "mean"),
            source=("source", "first"),
        )
        if len(series) <= horizon_steps + 3:
            continue
        series["arrivals"] = series["arrivals"].fillna(0).astype(float)
        series["queue_length_m"] = series["queue_length_m"].ffill().fillna(0.0)
        arrivals = series["arrivals"].to_numpy()
        times = series.index
        source = str(group["source"].iloc[0])
        session_start = times[0].isoformat()
        for index in range(3, len(series) - horizon_steps):
            future = arrivals[index + 1:index + horizon_steps + 1]
            if len(future) != horizon_steps:
                continue
            hour = times[index].hour + times[index].minute / 60
            rows.append({
                "lane_code": lane_codes[str(lane_id)],
                "arrivals_lag_1": arrivals[index],
                "arrivals_lag_2": arrivals[index - 1],
                "arrivals_lag_3": arrivals[index - 2],
                "arrivals_mean_3": float(np.mean(arrivals[index - 2:index + 1])),
                "queue_length_m": float(series["queue_length_m"].iloc[index]),
                "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
                "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
                TARGET_COLUMN: float(np.sum(future)),
                "session_id": str(session_id),
                "session_start": session_start,
                "sample_time": times[index].isoformat(),
                "target_end": times[index + horizon_steps].isoformat(),
                "lane_id": str(lane_id),
                "source": source,
            })

    return pd.DataFrame(rows, columns=[*FEATURE_COLUMNS, TARGET_COLUMN, "session_id",
                                      "session_start", "sample_time", "target_end", "lane_id", "source"])


def chronological_split(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Split by recording sessions; for a single session, purge the horizon gap."""
    if dataset.empty:
        raise ValueError("No supervised arrival samples are available.")
    ordered = dataset.sort_values(["session_start", "target_end"]).reset_index(drop=True)
    sessions = ordered.groupby("session_id")["session_start"].min().sort_values().index.tolist()
    if len(sessions) >= 4:
        train_n = max(1, int(len(sessions) * 0.70))
        val_n = max(1, int(round(len(sessions) * 0.15)))
        if train_n + val_n >= len(sessions):
            train_n = len(sessions) - 2
            val_n = 1
        train_sessions = sessions[:train_n]
        val_sessions = sessions[train_n:train_n + val_n]
        test_sessions = sessions[train_n + val_n:]
        train = ordered[ordered.session_id.isin(train_sessions)].copy()
        val = ordered[ordered.session_id.isin(val_sessions)].copy()
        test = ordered[ordered.session_id.isin(test_sessions)].copy()
        split = {"method": "recording_session", "train_sessions": train_sessions,
                 "validation_sessions": val_sessions, "test_sessions": test_sessions}
    elif len(sessions) == 1:
        sample_times = sorted(ordered["sample_time"].unique().tolist())
        if len(sample_times) < 4:
            raise ValueError("A single recording needs at least four time intervals after target construction.")
        train_pos = min(max(1, int(len(sample_times) * 0.70)), len(sample_times) - 2)
        val_pos = min(max(train_pos + 1, int(len(sample_times) * 0.85)), len(sample_times) - 1)
        train_end = sample_times[train_pos]
        val_end = sample_times[val_pos]
        train = ordered[ordered["target_end"] < train_end].copy()
        val = ordered[(ordered["sample_time"] >= train_end) &
                      (ordered["sample_time"] < val_end) &
                      (ordered["target_end"] < val_end)].copy()
        test = ordered[ordered["sample_time"] >= val_end].copy()
        split = {"method": "chronological_with_horizon_purge", "session_id": sessions[0],
                 "train_boundary": train_end, "test_boundary": val_end}
    else:
        raise ValueError("At least four recording sessions are required for leakage-safe train/validation/test splits.")
    if min(len(train), len(val), len(test)) < 1:
        raise ValueError("The chronological split produced an empty partition; collect longer recordings or more sessions.")
    if set(train.session_id) & set(test.session_id) and split["method"] == "recording_session":
        raise AssertionError("Recording-session leakage detected in train/test split.")
    return train, val, test, {**split, "rows": {"train": len(train), "validation": len(val), "test": len(test)}}


def generate_synthetic_observations(session_count: int = 12, minutes_per_session: int = 30,
                                    interval_seconds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Generate labeled synthetic benchmark observations, never production data."""
    if session_count < 4 or minutes_per_session < 2 or interval_seconds <= 0:
        raise ValueError("Use at least four sessions of two minutes or more and a positive interval.")
    rng = np.random.default_rng(seed)
    steps = minutes_per_session * 60 // interval_seconds
    start = datetime(2026, 1, 1, 5, 0, 0)
    rows = []
    for session in range(session_count):
        session_start = start + timedelta(days=session, hours=(session * 3) % 18)
        for lane_index, lane in enumerate(("lane1", "lane2")):
            demand = rng.uniform(0.15, 0.8)
            queue = 0
            for step in range(steps):
                hour = (session_start.hour + step * interval_seconds / 3600) % 24
                rush = 1.8 if 7 <= hour <= 9 or 16 <= hour <= 19 else 1.0
                demand = float(np.clip(0.75 * demand + 0.25 * rng.uniform(0.1, 0.9), 0.05, 1.0))
                arrivals = int(rng.poisson(demand * rush * (1.0 + 0.15 * lane_index)))
                queue = max(0, queue + arrivals - int(rng.poisson(0.7)))
                rows.append({
                    "analysis_job_id": f"synthetic-session-{session:03d}",
                    "timestamp": (session_start + timedelta(seconds=step * interval_seconds)).isoformat(),
                    "lane_id": lane,
                    "arrival_count": arrivals,
                    "queue_length": queue * 5.0,
                    "tracking_enabled": True,
                    "source": "synthetic",
                })
    return pd.DataFrame(rows)


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    return {"mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan")}


def train_arrival_model(observations: pd.DataFrame, artifact_dir: str | Path,
                        interval_seconds: int = 5, horizon_seconds: int = 60,
                        n_estimators: int = 80, random_state: int = 42) -> Dict[str, Any]:
    """Train XGBoost and RF, compare with persistence, and save provenance."""
    dataset = build_arrival_dataset(observations, interval_seconds, horizon_seconds)
    if dataset.empty:
        raise ValueError("No tracked arrival samples span the requested forecast horizon.")
    sources = sorted(dataset.source.unique().tolist())
    if len(sources) != 1:
        raise ValueError(f"Do not mix data sources in a training run: {sources}")
    if sources[0] not in {"video", "synthetic"}:
        raise ValueError(f"Unsupported dataset source: {sources[0]}")
    train, validation, test, split_info = chronological_split(dataset)
    x_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    x_val, y_val = validation[FEATURE_COLUMNS], validation[TARGET_COLUMN]
    x_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    xgb_model = XGBRegressor(n_estimators=n_estimators, max_depth=4, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, min_child_weight=2,
                             objective="reg:squarederror", eval_metric="rmse",
                             tree_method="hist", n_jobs=2, random_state=random_state)
    xgb_model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=10,
                                     min_samples_leaf=2, n_jobs=2,
                                     random_state=random_state)
    rf_model.fit(x_train, y_train)

    persistence_pred = test["arrivals_lag_1"].to_numpy() * (horizon_seconds / interval_seconds)
    predictions = {
        "persistence": persistence_pred,
        "random_forest": rf_model.predict(x_test),
        "xgboost": xgb_model.predict(x_test),
    }
    metrics = {name: _regression_metrics(y_test.to_numpy(), pred)
               for name, pred in predictions.items()}
    finite_metrics = all(np.isfinite(value) for model in metrics.values()
                         for value in model.values())
    inference_ready = bool(sources[0] == "video" and len(test) >= 30 and finite_metrics and
                           metrics["xgboost"]["mae"] <= metrics["persistence"]["mae"])
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    xgb_model.save_model(str(output / "xgboost_arrivals.json"))
    joblib.dump(rf_model, output / "random_forest_baseline.joblib")
    dataset.to_csv(output / "supervised_dataset.csv", index=False)
    dataset_hash = hashlib.sha256(dataset.to_csv(index=False).encode("utf-8")).hexdigest()
    metadata = {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now().isoformat(),
        "status": "ready_for_inference" if inference_ready else "trained_but_not_ready",
        "inference_ready": inference_ready,
        "data_source": sources[0],
        "target": TARGET_COLUMN,
        "horizon_seconds": horizon_seconds,
        "aggregation_interval_seconds": interval_seconds,
        "features": FEATURE_COLUMNS,
        "lanes": sorted(dataset.lane_id.unique().tolist()),
        "lane_codes": {lane: index for index, lane in enumerate(sorted(dataset.lane_id.unique().tolist()))},
        "training_config": {
            "xgboost": {"n_estimators": n_estimators, "max_depth": 4,
                        "learning_rate": 0.05, "min_child_weight": 2,
                        "subsample": 0.9, "colsample_bytree": 0.9},
            "random_forest": {"n_estimators": n_estimators, "max_depth": 10,
                              "min_samples_leaf": 2},
            "random_state": random_state,
        },
        "dataset_sha256": dataset_hash,
        "dataset_rows": len(dataset),
        "recording_sessions": int(dataset.session_id.nunique()),
        "split": split_info,
        "metrics": metrics,
        "readiness_rule": "Only recorded-video XGBoost models with at least 30 holdout samples, finite metrics and MAE no worse than persistence are ready for inference.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def predict_latest_arrivals(model: XGBRegressor, observations: pd.DataFrame,
                            metadata: Dict[str, Any], interval_seconds: int = 5) -> pd.DataFrame:
    """Forecast the next 60 seconds from each latest recorded lane window."""
    if metadata.get("data_source") != "video" or not metadata.get("inference_ready"):
        raise RuntimeError("Forecasts require an evaluated, inference-ready recorded-video model.")
    required = {"analysis_job_id", "timestamp", "lane_id", "arrival_count",
                "queue_length", "tracking_enabled", "source"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Traffic observations are missing columns: {sorted(missing)}")
    records = observations[(observations["source"] == "video") &
                           observations["tracking_enabled"].astype(bool)].copy()
    records["timestamp"] = pd.to_datetime(records["timestamp"], errors="coerce")
    records = records.dropna(subset=["timestamp", "analysis_job_id", "lane_id"])
    records = records.sort_values("timestamp")
    lane_codes = metadata.get("lane_codes", {})
    rows = []
    for (job_id, lane_id), group in records.groupby(["analysis_job_id", "lane_id"]):
        lane_id = str(lane_id)
        if lane_id not in lane_codes:
            continue
        group = group.sort_values("timestamp").set_index("timestamp")
        series = group.resample(f"{interval_seconds}s", origin="start").agg(
            arrivals=("arrival_count", "sum"), queue_length_m=("queue_length", "mean"))
        if len(series) < 3:
            continue
        series["arrivals"] = series["arrivals"].fillna(0).astype(float)
        series["queue_length_m"] = series["queue_length_m"].ffill().fillna(0.0)
        point = series.index[-1]
        history = series["arrivals"].to_numpy()
        hour = point.hour + point.minute / 60
        rows.append({
            "lane_id": lane_id,
            "analysis_job_id": str(job_id),
            "last_observation_time": point.isoformat(),
            "forecast_horizon_seconds": int(metadata["horizon_seconds"]),
            **{
                "lane_code": lane_codes[lane_id],
                "arrivals_lag_1": history[-1],
                "arrivals_lag_2": history[-2],
                "arrivals_lag_3": history[-3],
                "arrivals_mean_3": float(np.mean(history[-3:])),
                "queue_length_m": float(series["queue_length_m"].iloc[-1]),
                "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
                "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
            },
        })
    if not rows:
        return pd.DataFrame(columns=["lane_id", "analysis_job_id", "forecast_horizon_seconds", "predicted_arrivals"])
    features = pd.DataFrame(rows)
    features["predicted_arrivals"] = np.maximum(model.predict(features[FEATURE_COLUMNS]), 0.0)
    features["predicted_arrivals"] = features["predicted_arrivals"].round(2)
    return features.drop(columns=FEATURE_COLUMNS)


def load_ready_arrival_model(artifact_dir: str | Path) -> Tuple[XGBRegressor, Dict[str, Any]]:
    """Load only a recorded-video model that passed the holdout readiness gate."""
    directory = Path(artifact_dir)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No trained model metadata found at {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("data_source") != "video" or not metadata.get("inference_ready"):
        raise RuntimeError("Model unavailable for inference: artifact is synthetic or failed the holdout readiness gate.")
    model = XGBRegressor()
    model.load_model(str(directory / "xgboost_arrivals.json"))
    return model, metadata
