"""Tests for real/synthetic-separated 60-second arrival forecasting."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.prediction.traffic_arrival import (
    TARGET_COLUMN,
    build_arrival_dataset,
    chronological_split,
    generate_synthetic_observations,
    load_ready_arrival_model,
    predict_latest_arrivals,
    train_arrival_model,
)


def make_observations(sessions=1, steps=300):
    rows = []
    start = pd.Timestamp("2026-02-01T08:00:00")
    for session in range(sessions):
        for lane in ("lane1", "lane2"):
            for step in range(steps):
                rows.append({
                    "analysis_job_id": f"video-{session}",
                    "timestamp": (start + pd.Timedelta(days=session, seconds=5 * step)).isoformat(),
                    "lane_id": lane,
                    "arrival_count": int((step + session) % 3 == 0),
                    "queue_length": float((step + session) % 5 * 5),
                    "tracking_enabled": True,
                    "source": "video",
                })
    return pd.DataFrame(rows)


class ArrivalForecastTests(unittest.TestCase):
    def test_target_uses_only_next_full_sixty_seconds(self):
        dataset = build_arrival_dataset(make_observations(steps=30))
        row = dataset.sort_values("sample_time").iloc[0]
        self.assertEqual(row[TARGET_COLUMN], sum(int(i % 3 == 0) for i in range(4, 16)))
        self.assertEqual(row["source"], "video")

    def test_single_recording_split_purges_horizon_crossing_rows(self):
        dataset = build_arrival_dataset(make_observations(steps=300))
        train, validation, test, details = chronological_split(dataset)
        self.assertEqual(details["method"], "chronological_with_horizon_purge")
        self.assertLess(pd.to_datetime(train.target_end).max(), pd.to_datetime(validation.sample_time).min())
        self.assertLess(pd.to_datetime(validation.target_end).max(), pd.to_datetime(test.sample_time).min())

    def test_multiple_recordings_are_split_by_session(self):
        dataset = build_arrival_dataset(make_observations(sessions=5, steps=100))
        train, validation, test, details = chronological_split(dataset)
        self.assertEqual(details["method"], "recording_session")
        self.assertFalse(set(train.session_id) & set(validation.session_id))
        self.assertFalse(set(train.session_id) & set(test.session_id))
        self.assertEqual(len(train) + len(validation) + len(test), len(dataset))

    def test_inference_uses_recent_recorded_video_features_only(self):
        class FakeRegressor:
            def predict(self, features):
                self.columns = list(features.columns)
                return [3.5] * len(features)

        observations = make_observations(sessions=1, steps=4)
        metadata = {"data_source": "video", "inference_ready": True,
                    "horizon_seconds": 60, "lane_codes": {"lane1": 0, "lane2": 1}}
        forecasts = predict_latest_arrivals(FakeRegressor(), observations, metadata)
        self.assertEqual(len(forecasts), 2)
        self.assertTrue((forecasts.predicted_arrivals == 3.5).all())
        synthetic = observations.copy()
        synthetic["source"] = "synthetic"
        with self.assertRaises(RuntimeError):
            predict_latest_arrivals(FakeRegressor(), synthetic,
                                    {**metadata, "data_source": "synthetic"})

    def test_synthetic_training_saves_provenance_but_cannot_unlock_inference(self):
        observations = generate_synthetic_observations(session_count=6, minutes_per_session=10, seed=7)
        with tempfile.TemporaryDirectory() as directory:
            metadata = train_arrival_model(observations, directory, n_estimators=10)
            self.assertEqual(metadata["data_source"], "synthetic")
            self.assertFalse(metadata["inference_ready"])
            self.assertEqual(set(metadata["metrics"]), {"xgboost", "random_forest", "persistence"})
            self.assertTrue((Path(directory) / "xgboost_arrivals.json").is_file())
            self.assertTrue((Path(directory) / "supervised_dataset.csv").is_file())
            stored = json.loads((Path(directory) / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["dataset_sha256"], metadata["dataset_sha256"])
            with self.assertRaises(RuntimeError):
                load_ready_arrival_model(directory)


if __name__ == "__main__":
    unittest.main()
