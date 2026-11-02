"""Focused tests for recorded-video validation, tracking flow and job persistence."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.detection.detector import process_video_frames, validate_video_file
from src.analysis.traffic_analyser import TrafficAnalyser
from src.utils import create_analysis_job, get_analysis_job, get_analysis_jobs, get_traffic_history, init_database, update_analysis_job


class FakeDetector:
    def detect(self, frame):
        return {"detections": [{"class_name": "car"}], "lane_counts": {"lane1": 1},
                "class_totals": {"car": 1}, "frame_annot": frame.copy(), "latency_ms": 1.0}


class FakeTracker:
    def __init__(self):
        self.frames = 0
        self.reset_called = False

    def reset_for_video(self):
        self.frames = 0
        self.reset_called = True

    def track_frame(self, frame):
        self.frames += 1
        return {"tracks": [{"track_id": 7, "class_name": "car", "lanes": ["lane1"]}],
                "lane_counts": {"lane1": 1}, "class_totals": {"car": 1},
                "frame_annot": frame.copy()}


class VideoPipelineTests(unittest.TestCase):
    def make_video(self, directory, name="clip.avi", count=5):
        path = Path(directory) / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
        for n in range(count):
            writer.write(np.full((360, 640, 3), n * 20, dtype=np.uint8))
        writer.release()
        return path

    def test_video_is_validated_and_first_sample_is_processed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_video(directory)
            metadata = validate_video_file(str(source))
            self.assertEqual((metadata["width"], metadata["height"]), (640, 360))
            progress = []
            results = process_video_frames(str(source), FakeDetector(), sample_rate=3,
                max_inference_dim=320, progress_callback=lambda done, expected: progress.append((done, expected)))
            self.assertEqual([r["frame_index"] for r in results], [1, 4])
            self.assertEqual(progress[-1], (2, 2))
            self.assertEqual(results[0]["source_width"], 640)

    def test_tracker_is_used_and_annotated_output_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_video(directory)
            output = Path(directory) / "annotated.mp4"
            tracker = FakeTracker()
            results = process_video_frames(str(source), FakeDetector(), sample_rate=1,
                max_frames=2, output_path=str(output), tracker=tracker)
            self.assertEqual(tracker.frames, 2)
            self.assertTrue(tracker.reset_called)
            self.assertEqual(results[0]["detections"][0]["track_id"], 7)
            self.assertEqual([r["arrivals_by_lane"]["lane1"] for r in results], [1, 0])
            self.assertEqual(results[-1]["unique_track_count"], 1)
            cap = cv2.VideoCapture(str(output))
            try:
                readable, frame = cap.read()
                self.assertTrue(readable)
                self.assertEqual(frame.shape[1], 640)
            finally:
                cap.release()

    def test_analysis_job_round_trip_and_validation_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            db = init_database(str(Path(directory) / "jobs.sqlite"))
            source = self.make_video(directory)
            metadata = validate_video_file(str(source))
            create_analysis_job(db, "job-1", source.name, str(source), source.stat().st_size, metadata)
            update_analysis_job(db, "job-1", status="completed", class_totals_json={"car": 3}, metrics_json={"frames": 3})
            job = get_analysis_job(db, "job-1")
            self.assertEqual(job["class_totals"], {"car": 3})
            self.assertEqual(job["metrics"]["frames"], 3)
            self.assertEqual(len(get_analysis_jobs(db)), 1)
            db.close()
            with self.assertRaises(ValueError):
                validate_video_file(str(source), max_size_bytes=1)
            with self.assertRaises(ValueError):
                validate_video_file(str(Path(directory) / "bad.txt"))

    def test_video_time_and_lane_arrivals_are_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            db = init_database(str(Path(directory) / "observations.sqlite"))
            analyser = TrafficAnalyser(db_conn=db)
            timestamp = "2026-10-02T12:00:05"
            analyser.analyse({"lane1": 2}, {"car": 2}, timestamp=timestamp,
                             analysis_job_id="job-video", arrival_counts={"lane1": 1},
                             lane_class_totals={"lane1": {"car": 1}}, tracking_enabled=True)
            history = get_traffic_history(db, source="video")
            self.assertEqual(len(history), 1)
            self.assertEqual(history.iloc[0]["timestamp"], timestamp)
            self.assertEqual(history.iloc[0]["analysis_job_id"], "job-video")
            self.assertEqual(int(history.iloc[0]["arrival_count"]), 1)
            self.assertEqual(int(history.iloc[0]["cars"]), 1)
            self.assertTrue(bool(history.iloc[0]["tracking_enabled"]))
            db.close()

    def test_existing_database_gets_analytics_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy_path = Path(directory) / "legacy.sqlite"
            legacy = sqlite3.connect(legacy_path)
            legacy.execute("""CREATE TABLE traffic_observations (
                id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, source TEXT NOT NULL,
                lane_id TEXT NOT NULL, vehicle_count INTEGER NOT NULL DEFAULT 0,
                density REAL NOT NULL DEFAULT 0, queue_length REAL NOT NULL DEFAULT 0,
                traffic_state TEXT NOT NULL DEFAULT 'FREE', cars INTEGER DEFAULT 0,
                buses INTEGER DEFAULT 0, trucks INTEGER DEFAULT 0,
                motorcycles INTEGER DEFAULT 0, avg_speed REAL DEFAULT 0
            )""")
            legacy.commit()
            legacy.close()
            db = init_database(str(legacy_path))
            columns = {row[1] for row in db.execute("PRAGMA table_info(traffic_observations)")}
            self.assertIn("analysis_job_id", columns)
            self.assertIn("arrival_count", columns)
            self.assertIn("tracking_enabled", columns)
            db.close()


if __name__ == "__main__":
    unittest.main()
